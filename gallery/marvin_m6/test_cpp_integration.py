"""Ensure the public header links safely across translation units."""

from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import skipUnless, TestCase


HERE = Path(__file__).resolve().parent
EIGEN = Path('/usr/include/eigen3')


@skipUnless(shutil.which('g++') and (EIGEN / 'Eigen/Core').exists(),
            'C++ integration requires g++ and Eigen3')
class TestCppIntegration(TestCase):
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
