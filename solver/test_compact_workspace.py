"""Compact workspace growth preserves candidates, views and value semantics."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import TestCase, skipUnless

from codegen.cpp_codegen.codegen_cpp import CodeGenerationCpp
from codegen.skeleton.codegen_base import node_input_capacity_bounds
from solver.test_codegen_safety import overlapping_branch_tree, successful_overlap_tree

ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN = shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file()


class TestCompactWorkspace(TestCase):
    @skipUnless(TOOLCHAIN, 'Requires g++ and Eigen3')
    def test_generated_reuse_matches_fresh_calls_after_valid_and_invalid_targets(self):
        tree = successful_overlap_tree()
        with redirect_stdout(StringIO()):
            code = '\n'.join(CodeGenerationCpp(tree, use_safe_operator=True).generate_code()).split('// Code below for debug')[0]
        name = tree.robot.name+'_ik'
        main = r'''#include <cassert>
int main() {
    using Model=MODEL;
    Model::RawIKWorksace workspace;
    std::vector<std::array<double,Model::robot_nq>> output;
    for(int i=0;i<200;++i) {
        Eigen::Matrix4d target=Eigen::Matrix4d::Identity();
        target(2,3)=std::sin(i)*1.5;
        const auto fresh=Model::computeIK(target);
        Model::computeIK(target,workspace,output);
        assert(!fresh.empty() && output==fresh);
        for(const auto& q:output)
            assert((Model::computeFK(q)-target).cwiseAbs().maxCoeff()<1e-7);
        target(0,0)=std::numeric_limits<double>::quiet_NaN();
        Model::computeIK(target,workspace,output);
        assert(output.empty());
    }
}
'''.replace('MODEL', name)
        with tempfile.TemporaryDirectory(prefix='yaik-reuse-') as temporary:
            temporary = Path(temporary)
            source, binary = temporary/'reuse.cpp', temporary/'reuse'
            source.write_text(code+main)
            result = subprocess.run(['g++', '-std=c++14', '-O2', '-I', str(ROOT), '-I', '/usr/include/eigen3',
                                     str(source), '-o', str(binary)], capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_input_bounds_cover_overlapping_children_without_changing_global_limit(self):
        tree = overlapping_branch_tree(False, root_count=16, overlap=True)
        bounds = node_input_capacity_bounds(tree)
        self.assertEqual(len(bounds), tree.num_nodes)
        self.assertEqual(bounds[tree.root.flatten_idx], 1)
        for child in tree.node_list[2].children:
            if child is not None:
                self.assertEqual(bounds[child.flatten_idx], 16)
        with redirect_stdout(StringIO()):
            generator = CodeGenerationCpp(tree, use_safe_operator=True)
            code = '\n'.join(generator.generate_code())
        self.assertIn('static constexpr int max_n_solutions = 32;', code)
        self.assertIn('node_index_workspace.reset(node_input_capacities, max_n_solutions);', code)
        self.assertNotIn('workspace.raw_ik_out.reserve(max_n_solutions);', code)

    @skipUnless(TOOLCHAIN, 'Requires g++ and Eigen3')
    def test_growth_copy_move_reset_and_candidate_reference_stability(self):
        source = r'''#include "yaik_cpp_common.h"
#include <cassert>
#include <type_traits>
using namespace yaik_cpp;
int main() {
    CompactNodeIndexWorkspace workspace;
    const std::array<std::size_t,3> profile{{1,0,2}};
    workspace.reset(profile,8);
    assert(workspace.num_nodes()==3);
    assert(workspace.is_input_indices_valid_for_node(0));
    assert(!workspace.is_input_indices_valid_for_node(1));
    assert(!workspace.append_index_to_node(1,-1));
    assert(workspace.append_index_to_node(2,100));
    assert(workspace.append_index_to_node(2,101));
    const auto tail=workspace.get_input_indices_for_node(2);
    for(int i=0;i<8;++i) {
        assert(workspace.append_index_to_node(1,i));
        assert(tail.size()==2 && tail[0]==100 && tail[1]==101);
    }
    assert(!workspace.append_index_to_node(1,8));
    const auto head=workspace.get_input_indices_for_node(1);
    for(int i=0;i<8;++i) assert(head[i]==i);
    const auto capacity=workspace.index_storage_capacity();
    CompactNodeIndexWorkspace copy=workspace;
    workspace.reset(profile,8);
    assert(workspace.index_storage_capacity()==capacity);
    assert(workspace.get_input_indices_for_node(1).size()==0);
    assert(copy.get_input_indices_for_node(1).size()==8);
    CompactNodeIndexWorkspace assigned;
    assigned=copy;
    copy=copy;
    CompactNodeIndexWorkspace moved=std::move(copy);
    assert(moved.get_input_indices_for_node(2)[1]==101);
    copy.reset(profile,8);
    assert(copy.get_input_indices_for_node(2).size()==0);
    copy=std::move(assigned);
    assert(copy.get_input_indices_for_node(1)[7]==7);
    copy.reset(std::array<std::size_t,1>{{2}},8);
    assert(copy.append_index_to_node(0,77));
    bool rejected=false;
    try { copy.reset(std::array<std::size_t,1>{{9}},8); }
    catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected && copy.get_input_indices_for_node(0)[0]==77);
    rejected=false;
    try { copy.reset(std::array<std::size_t,1>{{0}},std::numeric_limits<std::size_t>::max()); }
    catch(const std::length_error&) { rejected=true; }
    assert(rejected && copy.get_input_indices_for_node(0)[0]==77);
    copy.reset(std::array<std::size_t,0>{{}},8);
    assert(copy.num_nodes()==0);

    CompactSolutionQueue<2,4> queue;
    queue.reset();
    assert(queue.push_solution({{1.,2.}})==0);
    auto* first=&queue.get_solution(0);
    for(int i=1;i<4;++i) assert(queue.push_solution({{double(i+1),2.}})==i);
    assert(&queue.get_solution(0)==first && (*first)[0]==1.);
    assert(queue.push_solution({{9.,9.}})==-1 && queue.size()==4);
    queue.solutions_validity[0]=false;
    auto duplicate=queue;
    queue.reset();
    assert(duplicate.size()==4 && !duplicate.is_solution_valid(0));
    queue=duplicate;
    duplicate=duplicate;
    auto transferred=std::move(duplicate);
    duplicate.reset();
    assert(transferred.get_solution(3)[0]==4. && duplicate.size()==0);
    queue=std::move(transferred);
    assert(queue.get_solution(3)[0]==4.);
    CompactSolutionQueue<2,4> partial;
    partial.push_solution({{5.,6.}});
    auto partial_copy=partial;
    auto* stable=&partial_copy.get_solution(0);
    partial_copy.push_solution({{7.,8.}});
    assert(&partial_copy.get_solution(0)==stable && (*stable)[0]==5.);
    using Compact=RawIkWorkspaceTemplate<6,10000,7,CompactSolutionQueue<6,10000>,CompactNodeIndexWorkspace>;
    static_assert(sizeof(Compact)<1024,"large candidate queues must not live in the workspace object");
    static_assert(sizeof(RawIkWorkspaceTemplate<6,10000,7>)>480000,"legacy default layout remains available");
}
'''
        with tempfile.TemporaryDirectory(prefix='yaik-compact-') as temporary:
            temporary = Path(temporary)
            path, binary = temporary/'workspace.cpp', temporary/'workspace'
            path.write_text(source)
            result = subprocess.run(['g++', '-std=c++14', '-O1', '-I', str(ROOT), '-I', '/usr/include/eigen3',
                                     str(path), '-o', str(binary)], capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
