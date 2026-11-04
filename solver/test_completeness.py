"""Check branch certificates against changed geometry and actual path guards."""

from pathlib import Path
from unittest import TestCase

import sympy as sp
import yaml

from codegen.skeleton.branch_certificates import certify_unreachable_branch
from codegen.skeleton.completeness import audit_completeness
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.tree_node import EquationAllZeroDispatcherNode
from codegen.skeleton.tree_serialize import TreeSerializer
from fk.fk_equations import inv_Px, inv_Py, inv_Pz
from fk.robots import DHEntry, RobotDescription
from solver.equation_types import Unknown
from solver.solved_variable import SolutionDegenerateRecord
from solver.test_equation_branches import explicit


def annulus_tree():
    x, y, z, length = sp.symbols('x y z length')
    model = RobotDescription('annulus')
    model.unknowns = [Unknown(symbol) for symbol in (x, y, z)]
    model.dh_params = [DHEntry(0, length, 0, x), DHEntry(0, sp.Rational(1, 2), 0, y),
                       DHEntry(0, 0, 0, z)]
    model.symbolic_parameters = {length}
    model.parameters_value = {length: 2.}
    xy = SolutionDegenerateRecord.record_all_equations([inv_Px, inv_Py])
    zz = SolutionDegenerateRecord.record_all_equations([inv_Pz])
    tree = SkeletonTree([explicit(x, [sp.S.Zero], xy), explicit(y, [sp.S.Zero], zz),
                         explicit(z, [sp.S.Zero])], model)
    tree.append_branched_solution([explicit(x, [sp.S.Zero]), explicit(y, [sp.S.Zero], zz),
                                   explicit(z, [sp.S.Zero])], tree.root, 1)
    return tree


class TestCompleteness(TestCase):
    def test_only_true_path_conjuncts_bound_the_inverse_position(self):
        tree = annulus_tree()
        main, singular = tree.node_list[2], tree.node_list[7]
        self.assertIsInstance(main, EquationAllZeroDispatcherNode)
        self.assertIsNone(certify_unreachable_branch(tree, main))
        certificate = certify_unreachable_branch(tree, singular)
        self.assertEqual(certificate['radius_lower_bound'], '3/2')
        self.assertGreater(sp.Rational(certificate['strict_squared_margin']), 0)
        self.assertFalse(certificate['floating_point_evaluation_certified'])
        report = audit_completeness(tree)
        self.assertEqual(len(report['unresolved_branches']), 1)
        self.assertEqual(len(report['certified_unreachable_branches']), 1)
        self.assertFalse(report['certified_complete'])

    def test_proof_is_recomputed_after_model_or_guard_changes(self):
        tree = annulus_tree()
        singular = tree.node_list[7]
        self.assertIsNotNone(certify_unreachable_branch(tree, singular))
        length = next(iter(tree.robot.symbolic_parameters))
        for invalid in (.5, float('inf'), float('nan')):
            tree.robot.parameters_value[length] = invalid
            self.assertIsNone(certify_unreachable_branch(tree, singular))
        del tree.robot.parameters_value[length]
        tree.robot.parameters_bound[length] = (0., 2.)
        self.assertIsNone(certify_unreachable_branch(tree, singular))
        tree.robot.parameters_value[length] = 2.
        singular._degenerate_checker = sp.Abs(inv_Pz) <= 2
        self.assertIsNone(certify_unreachable_branch(tree, singular))

    def test_disjunctions_and_unrelated_symbols_do_not_supply_bounds(self):
        tree = annulus_tree()
        singular = tree.node_list[7]
        tree.root._degenerate_checker = (sp.Abs(inv_Px) <= 1e-6) | (sp.Abs(inv_Py) <= 1e-6)
        self.assertIsNone(certify_unreachable_branch(tree, singular))
        tree.root._degenerate_checker = (sp.Abs(inv_Px) <= 1e-6) & (sp.Abs(sp.Symbol('other')) <= 1e-6)
        self.assertIsNone(certify_unreachable_branch(tree, singular))

    def test_complex_or_unbound_rotations_cannot_use_norm_preservation(self):
        for invalid in (sp.I, sp.I*sp.Symbol('x'), sp.Symbol('unbound')):
            tree = annulus_tree()
            tree.robot.dh_params[0].theta = invalid
            self.assertIsNone(certify_unreachable_branch(tree, tree.node_list[7]))

    def test_auxiliary_frames_must_be_verified_rigid_before_using_inverse_position(self):
        from fk.robots import RobotAuxiliaryData
        tree = annulus_tree()
        tree.robot.auxiliary_data = RobotAuxiliaryData()
        tree.robot.auxiliary_data.pre_transform_sp = sp.eye(4)
        self.assertIsNotNone(certify_unreachable_branch(tree, tree.node_list[7]))
        for invalid in (2, -1, sp.Float('0.999999999')):
            tree.robot.auxiliary_data.pre_transform_sp[0, 0] = invalid
            self.assertIsNone(certify_unreachable_branch(tree, tree.node_list[7]))

    def test_roundtrip_preserves_the_certificate_without_trusting_saved_proofs(self):
        tree = annulus_tree()
        serializer = TreeSerializer(tree.num_nodes, tree.robot)
        tree.preorder_visit(serializer)
        data = serializer.serialize_dict()
        before = audit_completeness(tree)
        after = audit_completeness(TreeSerializer.load_skeleton_tree(data))
        self.assertEqual(before['certified_unreachable_branches'], after['certified_unreachable_branches'])
        self.assertFalse(after['certified_complete'])

    def test_marvin_missing_branch_is_geometrically_empty_but_ik_is_not_certified(self):
        root = Path(__file__).resolve().parents[1]
        for side in ('left', 'right'):
            path = root/'gallery'/'marvin_m6'/f'marvin_m6_{side}_arm_ik.yaml'
            tree = TreeSerializer.load_skeleton_tree(yaml.safe_load(path.read_text()))
            report = audit_completeness(tree)
            self.assertFalse(report['unresolved_branches'])
            self.assertFalse(report['certified_complete'])
            self.assertEqual(len(report['certified_unreachable_branches']), 1)
            proof = report['certified_unreachable_branches'][0]['certificate']
            self.assertGreater(proof['radius_lower_bound_m'], .0089)
            self.assertEqual(proof['arithmetic'], 'exact_rational')
