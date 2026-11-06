"""Independent modified-DH FK for regenerated four-joint gallery singularities."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import TestCase, skipUnless

import numpy as np
import sympy as sp

from codegen.skeleton.tree_serialize import read_skeleton_tree

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('abb_irb460_110_240', 'kawasaki_CP130L')


def model_cases(name):
    folder = ROOT/'gallery'/name
    robot = read_skeleton_tree(str(folder/(name+'_ik.yaml'))).robot
    values = [sp.sympify(value).subs(robot.parameters_value)
              for entry in robot.dh_params for value in entry.flatten()]
    evaluate = sp.lambdify([unknown.symbol for unknown in robot.unknowns], values, 'numpy')
    offsets = np.array(robot.get_unknown_offset() or [0.]*robot.n_dofs)
    lower = np.array([float(q.lower_bound) for q in robot.unknowns])-offsets
    upper = np.array([float(q.upper_bound) for q in robot.unknowns])-offsets
    pre = robot.get_pre_transform_sp()
    post = robot.get_post_transform_sp()
    pre = np.eye(4) if pre is None else np.array(pre.subs(robot.parameters_value), dtype=float)
    post = np.eye(4) if post is None else np.array(post.subs(robot.parameters_value), dtype=float)
    rng = np.random.default_rng(20261117)
    cases = []
    for index in range(1000):
        q = rng.uniform(lower, upper)
        if index % 3 == 0:
            for joint in range(len(q)):
                if rng.random() < .5:
                    q[joint] = rng.choice([lower[joint], upper[joint], min(upper[joint], max(lower[joint], 0.))])
        target = pre.copy()
        fields = evaluate(*(q+offsets))
        for start in range(0, len(fields), 4):
            alpha, a, d, theta = fields[start:start+4]
            ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
            # Rx(alpha) Tx(a) Rz(theta) Tz(d), evaluated without generated FK.
            transform = np.array([[ct, -st, 0., a], [ca*st, ca*ct, -sa, -d*sa],
                                  [sa*st, sa*ct, ca, d*ca], [0., 0., 0., 1.]])
            target = target @ transform
        cases.append((q, target @ post))
    return cases


class TestGalleryRegeneration(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {name: model_cases(name) for name in NAMES}

    def test_recorded_artifact_hashes_match(self):
        for name in NAMES:
            folder = ROOT/'gallery'/name
            report = json.loads((folder/'provenance.json').read_text())
            self.assertEqual(report['seed'], 20261125)
            self.assertEqual(report['coverage_status'], 'not_certified')
            for artifact, expected in report['artifact_sha256'].items():
                self.assertEqual(hashlib.sha256((folder/artifact).read_bytes()).hexdigest(), expected, artifact)

    def test_saved_trees_reproduce_exported_code(self):
        from codegen.cpp_codegen.codegen_cpp import CodeGenerationCpp
        from codegen.py_codegen.codegen_py import CodeGenerationPython
        for name in NAMES:
            folder = ROOT/'gallery'/name
            tree = read_skeleton_tree(str(folder/(name+'_ik.yaml')))
            for extension, generator in (('py', CodeGenerationPython(tree)),
                                         ('cpp', CodeGenerationCpp(tree, use_safe_operator=True))):
                expected = '\n'.join(line.rstrip() for line in generator.generate_code())+'\n'
                self.assertEqual((folder/(name+'_ik_generated.'+extension)).read_text(), expected)

    def test_python_recovers_all_independent_targets(self):
        for name in NAMES:
            path = ROOT/'gallery'/name/(name+'_ik_generated.py')
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fk, solve = getattr(module, name+'_fk'), getattr(module, name+'_ik_solve')
            for index, (q, target) in enumerate(self.cases[name]):
                self.assertLess(np.max(np.abs(fk(q)-target)), 1e-12, (name, index))
                output = solve(target)
                self.assertTrue(any(np.isfinite(candidate).all() and np.max(np.abs(fk(candidate)-target))<1e-4
                                    for candidate in output), (name, index))

    @skipUnless(shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file(),
                'Requires g++ and Eigen3')
    def test_cpp_recovers_all_independent_targets(self):
        sections = []
        for index, name in enumerate(NAMES):
            sections.append('#define main gallery_demo_'+str(index)+'\n#include "'+
                            str(ROOT/'gallery'/name/(name+'_ik_generated.cpp'))+'"\n#undef main\n')
        sections.append('''#include <iostream>
int main() {
    int model, index=0;
    while(std::cin >> model) {
        Eigen::Matrix4d target;
        for(int row=0;row<4;++row)for(int col=0;col<4;++col)std::cin >> target(row,col);
        const auto output=model==0 ? abb_irb460_110_240_ik::computeIK(target) : kawasaki_CP130L_ik::computeIK(target);
        bool valid=false;
        for(const auto& q:output) {
            const auto actual=model==0 ? abb_irb460_110_240_ik::computeFK(q) : kawasaki_CP130L_ik::computeFK(q);
            if((actual-target).cwiseAbs().maxCoeff()<1e-4)valid=true;
        }
        if(!valid){std::cerr << "Missing independent target " << model << " " << index;return 1;}
        ++index;
    }
}
''')
        with tempfile.TemporaryDirectory(prefix='yaik-gallery-') as temporary:
            temporary = Path(temporary)
            source, binary = temporary/'consumer.cpp', temporary/'consumer'
            source.write_text('\n'.join(sections))
            compiled = subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(ROOT), '-I', '/usr/include/eigen3',
                                       str(source), '-o', str(binary)], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            rows = [str(model)+' '+' '.join(format(float(value), '.17g') for value in target.ravel())
                    for model, name in enumerate(NAMES) for q, target in self.cases[name]]
            result = subprocess.run([str(binary)], input='\n'.join(rows)+'\n', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
