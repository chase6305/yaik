"""Singular tangent-half-angle charts must not crash or erase old witnesses."""

from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import TestCase, mock, skipUnless
import warnings

import numpy as np

from codegen.cpp_codegen.codegen_cpp import CodeGenerationCpp
from codegen.py_codegen.codegen_py import CodeGenerationPython
from codegen.skeleton.tree_serialize import read_skeleton_tree
from python_run_import import compute_solution_from_tanhalf_LME


ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads(Path(__file__).with_name('runtime_cases.json').read_text())


class TestGeneralRuntime(TestCase):
    def test_linear_matrix_polynomial_recovers_finite_and_infinite_angle_roots(self):
        b = np.random.default_rng(7).normal(size=(6, 9))
        expected = 2*math.atan(.25)
        for scale in (1., 1e-200, 1e200):
            with self.subTest(scale=scale), warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter('always')
                output = compute_solution_from_tanhalf_LME(np.zeros_like(b), scale*b, -.25*scale*b)
                self.assertTrue(output)
                self.assertLessEqual(len(output), 48)
                self.assertLess(min(abs(value-expected) for value in output), 1e-12)
                self.assertLess(min(abs(abs(value)-math.pi) for value in output), 1e-12)
                for value in output:
                    self.assertLess(min(abs(value-expected), abs(abs(value)-math.pi)), 1e-10)
                self.assertEqual(emitted, [])

    def test_invalid_and_unresolved_matrices_return_an_empty_list(self):
        base = np.random.default_rng(7).normal(size=(6, 9))
        zero = np.zeros_like(base)
        self.assertEqual(compute_solution_from_tanhalf_LME(zero, zero, zero), [])
        self.assertEqual(compute_solution_from_tanhalf_LME(base, zero, base), [])
        for bad in (np.full((6, 9), np.nan), np.full((6, 9), np.inf),
                    np.ones((5, 9)), base.astype(complex)+1j):
            with self.subTest(shape=bad.shape):
                self.assertEqual(compute_solution_from_tanhalf_LME(bad, zero, zero), [])
        with mock.patch('python_run_import.np.linalg.inv', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                compute_solution_from_tanhalf_LME(base, zero, base)

    def test_subnormal_coefficients_are_solved_without_overflowing_an_inverse(self):
        base = np.random.default_rng(7).normal(size=(6, 9))*1e-310
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            output = compute_solution_from_tanhalf_LME(base, np.zeros_like(base), -.0625*base)
        for expected in (-2*math.atan(.25), 2*math.atan(.25)):
            self.assertLess(min(abs(value-expected) for value in output), 1e-10)
        self.assertEqual(emitted, [])

    def test_coupled_angles_use_correct_ratios_and_handle_zero_and_infinite_tangents(self):
        for angles in ((2*math.atan(.4), 2*math.atan(.3), 2*math.atan(-.7)),
                       (.76, 0., -1.2), (.76, .58, 0.),
                       (.76, math.pi, -1.2), (.76, .58, math.pi),
                       (math.pi, .58, -1.2)):
            with self.subTest(angles=angles):
                rng = np.random.default_rng(91)
                a, b, c = [rng.normal(size=(6, 9)) for _ in range(3)]
                sx, cx = math.sin(angles[0]/2), math.cos(angles[0]/2)
                sy, cy = math.sin(angles[1]/2), math.cos(angles[1]/2)
                sz, cz = math.sin(angles[2]/2), math.cos(angles[2]/2)
                monomials = np.kron([sy*sy, sy*cy, cy*cy], [sz*sz, sz*cz, cz*cz])
                residual = (a*sx*sx+b*sx*cx+c*cx*cx)@monomials
                # Enforce a known homogeneous root without dividing by tan(pi/2).
                coefficient, weight = (a, sx*sx) if abs(sx) > abs(cx) else (c, cx*cx)
                coefficient -= np.outer(residual, monomials/(weight*(monomials@monomials)))
                roots = compute_solution_from_tanhalf_LME(a, b, c, True)
                self.assertTrue(roots)
                distance = min(max(abs(math.remainder(value-reference, 2*math.pi))
                                   for value, reference in zip(root, angles)) for root in roots)
                self.assertLess(distance, 1e-6)

    def test_saved_robots_recover_singular_targets_and_keep_legacy_candidates(self):
        scopes = {}
        for case in CASES:
            name = case['robot']
            if name not in scopes:
                tree = read_skeleton_tree(str(ROOT/case['tree']))
                scope = {'__name__': 'general_runtime_regression'}
                with redirect_stdout(StringIO()):
                    exec('\n'.join(CodeGenerationPython(tree).generate_code()), scope)
                scopes[name] = scope
            scope = scopes[name]
            q = np.asarray(case['q'])
            with self.subTest(robot=name, kind=case['kind'], q=case['q']), \
                    warnings.catch_warnings(record=True) as emitted:
                target = scope[name+'_fk'](q)
                candidates = scope[name+'_ik_solve'](target)
                self.assertTrue(candidates)
                for value in candidates:
                    self.assertLess(np.max(np.abs(scope[name+'_fk'](value)-target)), 1e-4)
                self.assertEqual(emitted, [])

    @skipUnless(shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file(),
                'Requires g++ and Eigen3')
    def test_cpp_matrix_charts_buffer_compatibility_and_saved_robot_witnesses(self):
        # Exercise a semi-symbolic and a numerical-reduction tree in one binary.
        names = ('abb_crb_15000', 'denso_cvrb')
        sections = []
        for name in names:
            case = next(case for case in CASES if case['robot'] == name)
            tree = read_skeleton_tree(str(ROOT/case['tree']))
            with redirect_stdout(StringIO()):
                code = '\n'.join(CodeGenerationCpp(tree, use_safe_operator=True).generate_code())
            sections.append(code.split('// Code below for debug')[0])
        main = r'''
#include <cassert>
int main() {
    Eigen::Matrix<double, 6, 9> b;
    for (int row=0; row<6; ++row) for (int col=0; col<9; ++col)
        b(row,col)=std::sin(.7+row*1.31+col*.41)+std::cos(row*col*.73);
    const Eigen::Matrix<double, 6, 9> zero=Eigen::Matrix<double, 6, 9>::Zero();
    for (double scale : {1.,1e-200,1e200}) {
        const Eigen::Matrix<double, 6, 9> linear=b*scale, constant=-.25*linear;
        std::array<double,48> roots;
        const int count=yaik_cpp::general_6dof_internal::computeSolutionFromTanhalfLME(zero,linear,constant,&roots);
        assert(count>0 && count<=48);
        bool finite=false, infinite=false;
        for (int i=0;i<count;++i) {
            finite|=std::abs(roots[i]-2*std::atan(.25))<1e-10;
            infinite|=std::abs(std::abs(roots[i])-M_PI)<1e-10;
        }
        assert(finite && infinite);
        std::array<double,16> legacy_buffer;
        const int legacy_count=yaik_cpp::general_6dof_internal::computeSolutionFromTanhalfLME(zero,linear,constant,&legacy_buffer);
        assert(legacy_count>0 && legacy_count<=16);
    }
    {
        const Eigen::Matrix<double,6,9> a=b*1e-310, c=-.0625*a;
        std::array<double,48> roots;
        const int count=yaik_cpp::general_6dof_internal::computeSolutionFromTanhalfLME(a,zero,c,&roots);
        bool negative=false, positive=false;
        for (int i=0;i<count;++i) {
            positive|=std::abs(roots[i]-2*std::atan(.25))<1e-8;
            negative|=std::abs(roots[i]+2*std::atan(.25))<1e-8;
        }
        assert(positive && negative);
    }
'''
        matrix_case = json.loads(Path(__file__).with_name('runtime_matrix_case.json').read_text())
        main += '{\n'
        for name in ('a', 'b', 'c'):
            coefficients = ','.join(repr(value) for row in matrix_case[name] for value in row)
            main += 'Eigen::Matrix<double,6,9> '+name+'; '+name+' << '+coefficients+';\n'
        main += '''std::array<double,48> roots;
const int count=yaik_cpp::general_6dof_internal::computeSolutionFromTanhalfLME(a,b,c,&roots);
bool found=false;
for(int i=0;i<count;++i) found|=std::abs(std::abs(roots[i])-M_PI)<1e-8;
assert(found); }\n'''
        for name in names:
            for case in (case for case in CASES if case['robot'] == name):
                values = ','.join(repr(value) for value in case['q'])
                main += ('{ const std::array<double,6> q{{'+values+'}}; '
                         'const auto target='+name+'_ik::computeFK(q); '
                         'const auto output='+name+'_ik::computeIK(target); assert(!output.empty()); '
                         'for (const auto& candidate : output) assert(('
                         +name+'_ik::computeFK(candidate)-target).squaredNorm()<1e-6); }\n')
        main += '}\n'
        with tempfile.TemporaryDirectory(prefix='yaik-general-runtime-') as temporary:
            temporary = Path(temporary)
            source, executable = temporary/'test.cpp', temporary/'test'
            source.write_text('\n'.join(sections)+main)
            completed = subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(ROOT),
                                        '-I', '/usr/include/eigen3', str(source), '-o', str(executable)],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            subprocess.run([str(executable)], check=True)
