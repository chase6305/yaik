"""Narrow-family witnesses and critical-parameter code generation."""

from pathlib import Path
import math
import shutil
import subprocess
import tempfile
from unittest import TestCase, skipUnless

import numpy as np
import sympy as sp

from codegen.cpp_codegen.codegen_cpp import generate_code_cpp
from codegen.skeleton.build_equation_branch import build_equation_zero_branches
from codegen.skeleton.build_equation_branch import _provably_nonzero
from codegen.skeleton.tree_serialize import TreeSerializer
from python_run_import import trig_family_representatives, try_solve_linear_type2_specific_rows
from solver.test_equation_branches import coaxial_tree, scope_for
from solver.solved_variable import SolutionMethod
from solver.solved_variable import SolutionDegenerateRecord


def precision_tree():
    from codegen.skeleton.skeleton_tree import SkeletonTree
    from fk.robots import RobotDescription, DHEntry
    from solver.equation_types import Unknown, UnknownType
    from solver.test_equation_branches import explicit
    z, pz = sp.symbols('z Pz')
    model = RobotDescription('precision_witness')
    model.unknowns = [Unknown(z, UnknownType.Translational.name)]
    model.dh_params = [DHEntry(0, 0, z, 0)]
    # Squared-position equations can give both signs. Both are accepted by
    # the legacy loose FK tolerance here, but only +Pz is the exact IK witness.
    return SkeletonTree([explicit(z, [-pz, pz])], model)


def guarded_seed_tree(auxiliary=False):
    from codegen.skeleton.skeleton_tree import SkeletonTree
    from fk.robots import RobotDescription, DHEntry
    from solver.equation_types import Unknown
    from solver.solved_variable import VariableSolution
    from solver.test_equation_branches import explicit
    x, extra, p = sp.symbols('x extra p')
    model = RobotDescription('guarded_seed_aux' if auxiliary else 'guarded_seed_root')
    model.unknowns = [Unknown(x)]
    model.dh_params = [DHEntry(0, 0, 0, x)]
    model.symbolic_parameters = {p}
    model.parameters_bound = {p: (0., 1.)}
    solution = VariableSolution.make_explicit_solution(
        extra if auxiliary else x, [1/p], SolutionMethod.FamilyRepresentative.name,
        argument_valid_checkers=[p > 0],
        degenerate_record=SolutionDegenerateRecord.record_always_non_degenerate())
    solution.internal_solution.set_family_critical_equations([(1/p, 0, 0)], (-math.pi, math.pi), p > 0)
    return SkeletonTree([explicit(x, [sp.S.Zero]), solution] if auxiliary else [solution], model)


def critical_tree(guarded=False):
    tree = coaxial_tree()
    build_equation_zero_branches(tree)
    family = next(node.solution for node in tree.node_list if node.is_solution_node and
                  node.solution.solution_method == SolutionMethod.FamilyRepresentative.name)
    left, right = .123456789, .123456789+1e-11
    if guarded:
        p = sp.Symbol('p')
        tree.robot.symbolic_parameters.add(p)
        tree.robot.parameters_bound[p] = (0., 1.)
        family.internal_solution.set_family_critical_equations([(1/p, 0, 0)], (-math.pi, math.pi), p > 0)
    else:
        family.internal_solution.set_family_critical_equations([
            (-math.sin(left), math.cos(left), 0),
            (-math.sin(right), math.cos(right), 0)], (-math.pi, math.pi))
    return tree


class TestCriticalFamilies(TestCase):
    def test_disabled_critical_cuts_retain_seed_guards_in_the_numerical_visitor(self):
        from codegen.skeleton.visitor_numerical_solve import SkeletonNumericalSolution, step_from_explicit_solution
        tree = guarded_seed_tree()
        solution = next(node.solution for node in tree.node_list if node.is_solution_node)
        p = next(iter(tree.robot.symbolic_parameters))
        current = SkeletonNumericalSolution([0.], [-1])
        self.assertEqual(step_from_explicit_solution(current, {}, solution, 0, {p: 0.}), [])
        output = step_from_explicit_solution(current, {}, solution, 0, {p: 1.})
        self.assertTrue(output)
        scope = scope_for(tree)
        self.assertEqual(scope['guarded_seed_root_ik_solve_raw'](np.eye(4), 0.), [])

    def test_optional_guarded_family_does_not_erase_a_complete_prefix(self):
        tree = guarded_seed_tree(auxiliary=True)
        scope = scope_for(tree)
        output = scope['guarded_seed_aux_ik_solve_raw'](np.eye(4), 0.)
        np.testing.assert_allclose([q.ravel() for q in output], [[0.]])
    def test_offset_axes_keep_the_position_aware_shoulder_solver(self):
        from codegen.skeleton.family_critical import enrich_spherical_shoulders
        from codegen.skeleton.skeleton_tree import SkeletonTree
        from fk.robots import RobotDescription, DHEntry
        from solver.equation_types import Unknown
        from solver.solved_variable import VariableSolution
        from solver.solved_variable_impl import LinearSinCosType_2_SolutionImpl
        from solver.test_equation_branches import explicit
        x, y, z = sp.symbols('x y z')
        for a1, d1, a2 in ((0, 0, 0), (.1, 0, 0), (0, .1, 0), (0, 0, .1)):
            model = RobotDescription('shoulder_geometry')
            model.unknowns = [Unknown(symbol) for symbol in (x, y, z)]
            model.dh_params = [DHEntry(0, 0, 0, x), DHEntry(-sp.pi/2, a1, d1, y),
                               DHEntry(-sp.pi/2, a2, 0, z)]
            solution = VariableSolution(LinearSinCosType_2_SolutionImpl(z, sp.eye(4), [(0, 1, 2)]))
            tree = SkeletonTree([solution, explicit(y, [sp.S.Zero]), explicit(x, [sp.S.Zero])], model)
            self.assertEqual(enrich_spherical_shoulders(tree), int(not any((a1, d1, a2))))

    def test_nonzero_pruning_uses_exact_bounds_and_does_not_sample_joint_values(self):
        angle, length = sp.symbols('angle length')
        record = SolutionDegenerateRecord.record_all_equations([length+sp.sin(angle)])
        self.assertTrue(_provably_nonzero(record, {length: 2.0}))
        self.assertFalse(_provably_nonzero(record, {length: .5}))
    def test_all_rows_recover_a_joint_angle_from_a_weak_preselected_residual(self):
        angle, epsilon = .431, 1e-8
        matrix = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.],
                           [1., 1., 1+epsilon*math.cos(angle), 1-epsilon*math.sin(angle)],
                           [0., 0., math.cos(angle), -math.sin(angle)]])
        for scale in (1., 1e-200, 1e200):
            roots = try_solve_linear_type2_specific_rows(matrix*scale, 0, 1, 2)
            self.assertIsNotNone(roots)
            self.assertLess(min(abs(math.remainder(root-angle, 2*math.pi)) for root in roots), 1e-12)
    def test_preserves_cells_narrower_than_a_sampling_step(self):
        left, right = .123456789, .123456789+1e-11
        equations = [(-math.sin(value), math.cos(value), 0) for value in (left, right)]
        output = trig_family_representatives(equations, -math.pi, math.pi)
        self.assertTrue(any(left < value < right for value in output))
        self.assertEqual(output[0], -math.pi)
        self.assertEqual(output[-1], math.pi)
        self.assertLessEqual(len(output), 4*len(equations)+3)

    def test_near_tangent_cuts_do_not_collapse_or_underflow(self):
        for small in (1e-12, 1e-200):
            output = trig_family_representatives([(1., small, -1.)], -10*small, 10*small)
            self.assertIn(0., output)
            self.assertIn(2*math.atan2(small, 1.), output)
        for scale in (1e-200, 1e200):
            small = scale*1e-12
            output = trig_family_representatives([(scale, small, -scale)], -1e-11, 1e-11)
            self.assertIn(0., output)
            self.assertIn(2*math.atan2(small/scale, 1.), output)

    def test_large_joint_origins_preserve_phase_when_adding_turns(self):
        for target in (1000061.5465403113, 999999989.1289062, 1000000000029.6497,
                       100000000000060.98, 1000000000000060.6, -1000000000035.2318):
            rows = [(-math.sin(target), math.cos(target), 0.)]
            output = trig_family_representatives(rows, target-1., target+1.)
            self.assertIn(target, output)
            self.assertLessEqual(len(output), 7)

    def test_nonfinite_and_constant_equations_do_not_corrupt_seed_candidates(self):
        for coefficient in range(3):
            for invalid in (math.nan, math.inf, -math.inf):
                row = [1., 1., 1.]
                row[coefficient] = invalid
                output = trig_family_representatives([row, (0, 0, 1), (0, 0, 0)], -1, 1, [.2])
                self.assertEqual(output, [-1., 0., .2, 1.])
        for bounds in ((1, -1), (-4, 4), (0, math.inf)):
            with self.assertRaises(ValueError):
                trig_family_representatives([], *bounds)

    def test_serialization_retains_critical_cuts_and_domain_guards(self):
        for guarded in (False, True):
            tree = critical_tree(guarded)
            serializer = TreeSerializer(tree.num_nodes, tree.robot)
            tree.preorder_visit(serializer)
            loaded = TreeSerializer.load_skeleton_tree(serializer.serialize_dict())
            scope = scope_for(loaded)
            q = np.array([.2, -.3, .6])
            parameters = (0.,) if guarded else ()
            target = scope['coaxial_family_fk'](q, *parameters)
            candidates = scope['coaxial_family_ik_solve'](target, *parameters)
            self.assertTrue(candidates)
            if guarded:
                self.assertEqual(len(candidates), 33)
            else:
                self.assertTrue(any(.123456789 < candidate[0, 0] < .123456789+1e-11
                                    for candidate in candidates))
            for candidate in candidates:
                np.testing.assert_allclose(scope['coaxial_family_fk'](candidate, *parameters), target, atol=1e-12)

    @skipUnless(shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file(),
                'Requires g++ and Eigen3')
    def test_cpp_critical_cells_and_inactive_guard(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='yaik-critical-') as temporary:
            temporary = Path(temporary)
            tree, guarded = critical_tree(), critical_tree(True)
            guarded.robot.name = 'guarded_family'
            generate_code_cpp(tree, str(temporary/'critical.cpp'), use_safe_operator=True)
            generate_code_cpp(guarded, str(temporary/'guarded.cpp'), use_safe_operator=True)
            generate_code_cpp(precision_tree(), str(temporary/'precision.cpp'), use_safe_operator=True)
            generate_code_cpp(guarded_seed_tree(auxiliary=True), str(temporary/'optional.cpp'), use_safe_operator=True)
            (temporary/'test.cpp').write_text('''#define main critical_demo
#include "critical.cpp"
#undef main
#define main guarded_demo
#include "guarded.cpp"
#undef main
#define main precision_demo
#include "precision.cpp"
#undef main
#define main optional_demo
#include "optional.cpp"
#undef main
#include <cassert>
#include "gallery/marvin_m6/solver.hpp"
int main() {
    assert(guarded_seed_aux_ik::computeIK(Eigen::Matrix4d::Identity(), 0.0).size() == 1);
    const auto precision_target = precision_witness_ik::computeFK({{2e-7}});
    const auto precise = precision_witness_ik::computeIK(precision_target);
    assert(std::any_of(precise.begin(), precise.end(), [](const auto& q) { return q[0] == 2e-7; }));
    for (double target : {1000061.5465403113, 999999989.1289062, 1000000000029.6497,
                          100000000000060.98, 1000000000000060.6, -1000000000035.2318}) {
        const std::vector<std::array<double, 3>> rows{{{-std::sin(target), std::cos(target), 0}}};
        for (const auto& cuts : {yaik_cpp::trigFamilyRepresentatives(rows, target-1, target+1),
                                marvin_m6::detail::familyParameters(rows, target-1, target+1)}) {
            assert(std::find(cuts.begin(), cuts.end(), target) != cuts.end());
            assert(cuts.size() <= 7);
        }
    }
    for (double small : {1e-12, 1e-200}) {
        const std::vector<std::array<double, 3>> coefficients{{{1, small, -1}}};
        const double root = 2*std::atan2(small, 1.0);
        for (const auto& cuts : {yaik_cpp::trigFamilyRepresentatives(coefficients, -10*small, 10*small),
                                marvin_m6::detail::familyParameters(coefficients, -10*small, 10*small)}) {
            assert(std::find(cuts.begin(), cuts.end(), 0.0) != cuts.end());
            assert(std::find(cuts.begin(), cuts.end(), root) != cuts.end());
        }
    }
    std::vector<std::array<double, 1>> near_limit{{{1+1.5e-8}}, {{1-1e-10}}, {{1-2e-10}}};
    const auto classes = yaik_cpp::jointLimitClasses<1>(near_limit, {{-1}}, {{1}}, {{false}});
    yaik_cpp::removeDuplicateWithinClasses<1>(near_limit, 1e-6, classes);
    assert(near_limit.size() == 2);
    assert(near_limit[0][0] > 1 && near_limit[1][0] < 1);
    const double angle = .431, epsilon = 1e-8;
    Eigen::Matrix<double, 4, 4> weak;
    weak << 1, 0, 1, 0, 0, 1, 0, 1,
        1, 1, 1+epsilon*std::cos(angle), 1-epsilon*std::sin(angle),
        0, 0, std::cos(angle), -std::sin(angle);
    for (double scale : {1.0, 1e-200, 1e200}) {
        const Eigen::Matrix<double, 4, 4> scaled = weak*scale;
        double first, second;
        assert(yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<4>(scaled, 0, 1, 2, first, second));
        assert(std::min(std::abs(std::remainder(first-angle, 2*M_PI)),
                        std::abs(std::remainder(second-angle, 2*M_PI))) < 1e-12);
    }
    std::array<double, 3> q{{.2, -.3, .6}};
    const auto pose = coaxial_family_ik::computeFK(q);
    const auto candidates = coaxial_family_ik::computeIK(pose);
    bool found = false;
    for (const auto& candidate : candidates) {
        if (candidate[0] > .123456789 && candidate[0] < .123456789+1e-11) found = true;
        assert((coaxial_family_ik::computeFK(candidate)-pose).cwiseAbs().maxCoeff() < 1e-12);
    }
    assert(found);
    assert(guarded_family_ik::computeIK(pose, 0.0).size() == 33);
}
''')
            compiled = subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(root), '-I', '/usr/include/eigen3',
                            str(temporary/'test.cpp'), '-o', str(temporary/'test')],
                           capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            subprocess.run([str(temporary/'test')], check=True)
