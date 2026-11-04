"""Ensure the public header links safely across translation units."""

from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import mock, skipUnless, TestCase

import numpy as np

from gallery.marvin_m6.validate import CppSolver, UrdfArm, pose_error, cases_for


HERE = Path(__file__).resolve().parent
EIGEN = Path('/usr/include/eigen3')


@skipUnless(shutil.which('g++') and (EIGEN / 'Eigen/Core').exists(),
            'C++ integration requires g++ and Eigen3')
class TestCppIntegration(TestCase):
    def test_bounded_arms_share_one_compilation_and_keep_their_own_frames(self):
        with tempfile.TemporaryDirectory(prefix='marvin-m6-shared-') as temporary:
            command = subprocess.run
            with mock.patch('gallery.marvin_m6.validate.subprocess.run', wraps=command) as compile_cpp:
                solvers = {side: CppSolver(side, temporary, bounded=True) for side in ('left', 'right')}
            self.assertEqual(compile_cpp.call_count, 1)
            q = np.array([0.3, -0.5, 0.7, -1.0, 0.4, -0.2, 0.1])
            for side, solver in solvers.items():
                arm = UrdfArm(side)
                samples = [('regular', q)]+[(label, joints) for label, joints in cases_for(arm, 0, 20261103)
                                           if label.startswith('elbow_repeated_limit/')]
                for label, joints in samples:
                    with self.subTest(side=side, case=label):
                        target = arm.fk(joints)
                        np.testing.assert_allclose(solver.fk(joints), target, atol=1e-14, rtol=0)
                        candidates = solver.ik(target, joints[4])
                        self.assertTrue(len(candidates))
                        for candidate in candidates:
                            position, rotation = pose_error(arm.fk(candidate), target)
                            self.assertLessEqual(position, 1e-7)
                            self.assertLessEqual(rotation, 1e-6)
                            self.assertIsNotNone(arm.normalize(candidate))

    def test_header_links_in_multiple_translation_units(self):
        with tempfile.TemporaryDirectory(prefix='marvin-m6-link-') as temporary:
            temporary = Path(temporary)
            (temporary / 'first.cpp').write_text('''#include "solver.hpp"
int second();
int main() { return second(); }
''')
            (temporary / 'second.cpp').write_text('''#include "solver.hpp"
int second() {
    const std::array<double, 7> q{{0.3, -0.5, 0.7, -1.0, 0.4, -0.2, 0.1}};
    for (auto side : {marvin_m6::Side::Left, marvin_m6::Side::Right}) {
        const auto target = marvin_m6::computeFK(q, side);
        const auto solutions = marvin_m6::solveIK(target, q[4], side);
        if (solutions.empty()) return 1;
    }
    return 0;
}
''')
            executable = temporary / 'check'
            subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(EIGEN),
                            '-I', str(HERE), str(temporary / 'first.cpp'),
                            str(temporary / 'second.cpp'), '-o', str(executable)],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(executable)], check=True)
