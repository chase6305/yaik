import os.path
import random
from contextlib import contextmanager
import solver.build_equations as build_equations
from fk.robots import RobotDescription
import fk.robot_models as robot_models
import solver.degenerate_analyse.numerical_check_data as numerical_check_data
from solver.solve_non_degenerate import solve_main_branch
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.build_variable_branch import build_solved_variable_branches
from codegen.skeleton.tree_serialize import save_ik_tree, read_skeleton_tree
from typing import Optional, Set
import sympy as sp
import numpy as np
import attr


@attr.s
class RunIKOption(object):
    use_all_intersection_pair_axis_equation: bool = False
    try_intersecting_axis_equation: bool = True
    use_polynomial_solver_in_main_branch: bool = True
    random_seed: Optional[int] = attr.ib(default=None)
    max_num_nodes: int = attr.ib(default=50)


def check_robot(robot: RobotDescription) -> bool:
    """
    Everything in robot.dh_params should be declared in either unknown or robot.symbolic_parameters,
    and everything in robot.symbolic_parameters must be valued or bounded.
    The Unknown should contain bound by themselves.
    """
    errors = []

    def finite(value):
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError, OverflowError):
            return False

    def valid_bounds(bounds):
        try:
            lower, upper = bounds
            return finite(lower) and finite(upper) and float(lower) <= float(upper)
        except (TypeError, ValueError):
            return False

    unknown_symbols = [unknown.symbol for unknown in robot.unknowns]
    if any(not isinstance(symbol, sp.Symbol) for symbol in unknown_symbols):
        print('Invalid robot description: joint variables must be symbols')
        return False
    unknown_set = set(unknown_symbols)
    parameter_symbols = robot.symbolic_parameters | set(robot.parameters_value) | set(robot.parameters_bound)
    if {symbol.name for symbol in unknown_symbols} & {str(symbol) for symbol in parameter_symbols}:
        errors.append('joint symbols must not also be declared as parameters')
    if not unknown_symbols or len({symbol.name for symbol in unknown_symbols}) != len(unknown_symbols):
        errors.append('joint symbols must be nonempty and unique')
    if len(robot.dh_params) != len(robot.unknowns):
        errors.append('each DH entry must correspond to one joint')

    declared = unknown_set | robot.symbolic_parameters
    undeclared = set()
    for dh in robot.dh_params:
        for value in dh.flatten():
            expression = sp.sympify(value)
            undeclared.update(expression.free_symbols - declared)
            if not expression.free_symbols and not finite(expression):
                errors.append('DH constants must be finite real numbers')
    if undeclared:
        errors.append('undeclared DH parameters: ' + ', '.join(sorted(map(str, undeclared))))

    for unknown in robot.unknowns:
        if not valid_bounds((unknown.lower_bound, unknown.upper_bound)):
            errors.append('invalid joint bounds: ' + str(unknown.symbol))
        if not (unknown.is_revolute or unknown.is_translational):
            errors.append('unsupported joint type: ' + str(unknown.symbol))
    for unknown, dh in zip(robot.unknowns, robot.dh_params):
        joint_parameter = dh.theta if unknown.is_revolute else dh.d
        if joint_parameter != unknown.symbol:
            errors.append('joint must match its DH theta/d symbol: ' + str(unknown.symbol))
        fixed_entries = (dh.alpha, dh.a, dh.d if unknown.is_revolute else dh.theta)
        if any(sp.sympify(value).free_symbols & unknown_set for value in fixed_entries):
            errors.append('fixed DH entries must not depend on joints: ' + str(unknown.symbol))

    offsets = robot.get_unknown_offset()
    if offsets is not None:
        try:
            valid_offsets = len(offsets) == len(robot.unknowns) and all(map(finite, offsets))
        except TypeError:
            valid_offsets = False
        if not valid_offsets:
            errors.append('joint offsets must contain one finite value per joint')
    frame_parameters = robot.symbolic_parameters | set(robot.parameters_value)
    for transform in (robot.get_pre_transform_sp(), robot.get_post_transform_sp()):
        if transform is None:
            continue
        if not isinstance(transform, sp.MatrixBase) or transform.shape != (4, 4):
            errors.append('base/tool transforms must be 4x4 symbolic matrices')
            continue
        if transform.free_symbols - frame_parameters:
            errors.append('base/tool transforms contain undeclared parameters or joint variables')
        if any(not value.free_symbols and not finite(value) for value in transform):
            errors.append('base/tool transform constants must be finite real numbers')
    for parameter in robot.symbolic_parameters:
        if parameter not in robot.parameters_value and parameter not in robot.parameters_bound:
            errors.append('missing parameter value or bounds: ' + str(parameter))
    for parameter, value in robot.parameters_value.items():
        if not finite(value):
            errors.append('invalid parameter value: ' + str(parameter))
    for parameter, bounds in robot.parameters_bound.items():
        if not valid_bounds(bounds):
            errors.append('invalid parameter bounds: ' + str(parameter))

    fixed = robot.unknown_as_parameter_more_dof
    if len(set(fixed)) != len(fixed) or not set(fixed).issubset(unknown_set):
        errors.append('redundant parameters must be distinct joint symbols')
    if len(unknown_set - set(fixed)) > 6:
        errors.append('fix enough redundant joints to leave at most six unknowns')
    if errors:
        print('Invalid robot description: ' + '; '.join(errors))
    return not errors


@contextmanager
def offline_random_seed(seed):
    if seed is None:
        yield
        return
    if not isinstance(seed, (int, np.integer)):
        raise ValueError('random_seed must be an integer')
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        np.random.seed(int(seed))
        random.seed(int(seed))
        print('Offline generation random seed:', seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def run_ik(
        robot: RobotDescription,
        numerical_test_cases_path: Optional[str] = None,
        run_ik_option: Optional[RunIKOption] = None):
    option = run_ik_option if run_ik_option is not None else RunIKOption()
    with offline_random_seed(option.random_seed):
        return _run_ik(robot, numerical_test_cases_path, option)


def resume_ik_tree(tree: SkeletonTree, random_seed=None, max_num_nodes=50):
    """Extend a saved tree without rerunning or replacing its main branch."""
    if not check_robot(tree.robot):
        raise ValueError('invalid robot model in the saved IK tree')
    with offline_random_seed(random_seed):
        build_solved_variable_branches(tree, max_num_nodes=max_num_nodes)
    return tree


def _run_ik(
        robot: RobotDescription,
        numerical_test_cases_path: Optional[str] = None,
        run_ik_option: Optional[RunIKOption] = None):
    # Check the robot
    robot_ok = check_robot(robot)
    if not robot_ok:
        print('Robot input checking failed. Do nothing and return.')
        return

    # Make the option if not available
    if run_ik_option is None:
        run_ik_option = RunIKOption()

    # Build the equations
    ik_equation_input = build_equations.build_ik_equations(robot, run_ik_option.use_all_intersection_pair_axis_equation)

    # Generate numerical test cases
    if numerical_test_cases_path is None:
        numerical_test_cases = numerical_check_data.generate_test_cases(ik_equation_input)
        test_cases_save_path = robot.name + '_numerical_test.yaml'
        numerical_check_data.save_test_cases(numerical_test_cases, test_cases_save_path)
    else:
        numerical_test_cases = numerical_check_data.load_test_cases(numerical_test_cases_path)

    # Logging
    print('We have {n} numerical test cases'.format(n=len(numerical_test_cases)))

    # invoke the solver
    variable_solutions = solve_main_branch(
        robot,
        ik_equation_input,
        numerical_test_cases,
        try_intersecting_axis_equation=run_ik_option.try_intersecting_axis_equation,
        use_polynomial_solver=run_ik_option.use_polynomial_solver_in_main_branch)
    tree = SkeletonTree(variable_solutions, robot, ik_equation_input)

    # Save the non-degenerate branch
    non_degenerate_save_path = robot.name + '_main.yaml'
    save_ik_tree(tree, non_degenerate_save_path)

    # Build the variable branch, and do it again
    build_solved_variable_branches(tree, max_num_nodes=run_ik_option.max_num_nodes)
    save_path = robot.name + '_ik.yaml'
    save_ik_tree(tree, save_path)


def solve_from_yaml(robot_yaml_path: str, test_case_path: Optional[str], run_ik_option: Optional[RunIKOption] = None):
    import yaml
    assert os.path.exists(robot_yaml_path)
    with open(robot_yaml_path, 'r') as read_stream:
        data_map = yaml.load(read_stream, Loader=yaml.CLoader)
    loaded_robot = RobotDescription.load_from_dict(data_map)
    run_ik(loaded_robot, test_case_path, run_ik_option)


def run_robot_from_script():
    robot_to_solve = robot_models.ur10_urdf_robot()
    test_case_path = None
    option = RunIKOption()
    option.use_all_intersection_pair_axis_equation = True
    option.try_intersecting_axis_equation = True
    # test_case_path = './gallery/test_data/franka_panda_numerical_test.yaml'
    # test_case_path = test_case_path if os.path.exists(test_case_path) else None
    run_ik(robot_to_solve, test_case_path, option)


def run_robot_ik_from_yaml():
    robot_path = './gallery/spherical_wrist_six_axis/spherical_wrist_six_axis.yaml'
    test_case_path = None
    option = RunIKOption()
    option.use_all_intersection_pair_axis_equation = True
    option.try_intersecting_axis_equation = True
    option.use_polynomial_solver_in_main_branch = False
    solve_from_yaml(robot_path, test_case_path, option)


def extend_main_branch_tree():
    main_branch_tree_path = './gallery/arm_robo/arm_robo_main.yaml'
    tree = read_skeleton_tree(main_branch_tree_path)
    rbt_name = tree.robot.name

    # Build the variable branch, and do it again
    build_solved_variable_branches(tree, max_num_nodes=50)
    save_path = rbt_name + '_ik.yaml'
    save_ik_tree(tree, save_path)


if __name__ == '__main__':
    import sys
    from yaik import main
    raise SystemExit(main(sys.argv[1:], command='generate'))
