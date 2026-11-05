"""A guard must skip undefined arithmetic before evaluating its predicate."""
from pathlib import Path
import math
import shutil
import subprocess
import tempfile
from unittest import TestCase, skipUnless

import numpy as np
import sympy as sp

from codegen.cpp_codegen.codegen_cpp import CodeGenerationCpp
from codegen.py_codegen.codegen_py import CodeGenerationPython
from codegen.skeleton.guard_codegen import predicate_code
from codegen.skeleton.skeleton_tree import SkeletonTree
from fk.robots import DHEntry, RobotDescription
from solver.equation_types import Unknown, UnknownType
from solver.solved_variable import SolutionMethod, VariableSolution

ROOT = Path(__file__).resolve().parents[1]


def guard_tree(prefix=False, nested=False, safe=True):
    angle, z, p, r11 = sp.symbols('angle z p r_11')
    model = RobotDescription('guard_domain_'+str(int(prefix))+str(int(nested))+str(int(safe)))
    model.unknowns = ([Unknown(z, UnknownType.Translational.name)] if prefix else [])+[Unknown(angle)]
    model.dh_params = ([DHEntry(0, 0, z, 0)] if prefix else [])+[DHEntry(0, p, 0, angle)]
    model.symbolic_parameters = {p}
    model.parameters_bound = {p: (0., 2.)}
    denominator = p+1/p if nested else p
    argument = r11/denominator
    solution = VariableSolution.make_explicit_solution(angle, [sp.acos(argument)], SolutionMethod.ArcCos.name,
               argument_valid_checkers=[(sp.Abs(p)>1e-6)&(sp.Abs(argument)<=1)])
    solutions = [solution]
    if prefix:
        solutions.insert(0, VariableSolution.make_explicit_solution(z, [sp.S.Zero], SolutionMethod.OneVariableAlgebra.name))
    return SkeletonTree(solutions, model)


class TestGuardDomains(TestCase):
    def test_boolean_composition_checks_inner_denominators_and_preserves_defined_truth(self):
        p, q = sp.symbols('p q')
        comparisons = [(1/p > 0), (1/(p+1/p) < 2), (q > 0) | (1/p > 0),
                       (q > 0) & (1/p > 0), sp.Not(1/p > 0, evaluate=False)]
        for predicate in comparisons:
            code = predicate_code(predicate, sp.pycode, 'python')
            for parameter in (-2., -1., .5, 1., 2.):
                for value in (-1., 1.):
                    self.assertEqual(eval(code, {'p': parameter, 'q': value, 'math': math}),
                                     bool(predicate.subs({p: parameter, q: value})))
            # An undefined arm is skipped; a defined true disjunct remains true.
            self.assertEqual(eval(code, {'p': 0., 'q': -1., 'math': math}), False)
        code = predicate_code((q > 0) | (1/p > 0), sp.pycode, 'python')
        self.assertTrue(eval(code, {'p': 0., 'q': 1., 'math': math}))
        # A defined false conjunct makes the negated conjunction true even
        # when the other conjunct is undefined. Negation must not bless an
        # undefined atomic comparison by treating it as ordinary False.
        code = predicate_code(sp.Not((q > 0)&(1/p > 0), evaluate=False), sp.pycode, 'python')
        self.assertTrue(eval(code, {'p': 0., 'q': -1., 'math': math}))
        self.assertFalse(eval(code, {'p': 0., 'q': 1., 'math': math}))

    def test_generated_python_rejects_excluded_parameters_without_warnings(self):
        for prefix in (False, True):
            for nested in (False, True):
                tree = guard_tree(prefix, nested)
                scope = {'__name__': 'guard_domain_test'}
                exec('\n'.join(CodeGenerationPython(tree).generate_code()), scope)
                solve = scope[tree.robot.name+'_ik_solve_raw']
                with np.errstate(all='raise'):
                    self.assertEqual(solve(np.eye(4), 0.), [])
                    self.assertEqual(solve(np.eye(4), 1e-300), [])
                    self.assertEqual(len(solve(np.eye(4), 1.)), 1)

    @skipUnless(shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file(),
                'Requires g++ and Eigen3')
    def test_cpp_skips_undefined_arithmetic_in_safe_and_plain_modes(self):
        sections, calls = [], []
        for prefix in (False, True):
            for nested in (False, True):
                for safe in (False, True):
                    tree = guard_tree(prefix, nested, safe)
                    sections.append('\n'.join(CodeGenerationCpp(tree, use_safe_operator=safe).generate_code())
                                    .split('// Code below for debug')[0])
                    calls.append('''{
                        MODEL::RawIKWorksace workspace;
                        for(double parameter:{0.,1e-300,1.}) {
                            std::feclearexcept(FE_ALL_EXCEPT);
                            MODEL::computeRawIK(target,parameter,workspace);
                            assert(!std::fetestexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW));
                            assert(workspace.raw_ik_out.size()==(parameter==1.?1:0));
                        }
                    }'''.replace('MODEL', tree.robot.name+'_ik'))
        source = '\n'.join(sections)+'\n#include <cfenv>\n#include <cassert>\nint main(){\n'
        source += 'const Eigen::Matrix4d target=Eigen::Matrix4d::Identity();\n'+'\n'.join(calls)+'\n}\n'
        with tempfile.TemporaryDirectory(prefix='yaik-guard-domains-') as temporary:
            temporary = Path(temporary)
            (temporary/'test.cpp').write_text(source)
            result = subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(ROOT), '-I', '/usr/include/eigen3',
                                     str(temporary/'test.cpp'), '-o', str(temporary/'test')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([str(temporary/'test')], check=True)
