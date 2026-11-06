"""Empty raw paths can supply nearby seeds without relaxing original-target FK."""
from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest import TestCase, skipUnless

import numpy as np
import sympy as sp

from codegen.cpp_codegen.codegen_cpp import CodeGenerationCpp
from codegen.py_codegen.codegen_py import CodeGenerationPython
from codegen.skeleton.tree_serialize import read_skeleton_tree

ROOT=Path(__file__).resolve().parents[1]
CASES=json.loads((ROOT/'solver/fixtures/empty_raw_recovery_cases.json').read_text())['cases']
TOOLCHAIN=shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file()


def independent_fk(robot,q,parameters):
    values=dict(robot.parameters_value)
    for symbol in robot.symbolic_parameters:
        if str(symbol) in parameters:values[symbol]=parameters[str(symbol)]
    fields=[sp.sympify(value).subs(values) for dh in robot.dh_params for value in dh.flatten()]
    evaluate=sp.lambdify([unknown.symbol for unknown in robot.unknowns],fields,'numpy')
    offsets=np.array(robot.get_unknown_offset() or [0.]*robot.n_dofs,dtype=float)
    pre,post=robot.get_pre_transform_sp(),robot.get_post_transform_sp()
    target=np.eye(4) if pre is None else np.array(pre.subs(values),dtype=float)
    fields=evaluate(*(np.asarray(q).reshape(-1)+offsets))
    for first in range(0,len(fields),4):
        alpha,a,d,theta=fields[first:first+4]
        ca,sa,ct,st=math.cos(alpha),math.sin(alpha),math.cos(theta),math.sin(theta)
        target=target@np.array([[ct,-st,0,a],[ca*st,ca*ct,-sa,-d*sa],
                                [sa*st,sa*ct,ca,d*ca],[0,0,0,1.]])
    if post is not None:target=target@np.array(post.subs(values),dtype=float)
    return target


class TestEmptyRecovery(TestCase):
    def test_python_saved_empty_paths_retain_fixed_inputs_and_strict_original_fk(self):
        cache={}
        for case in CASES:
            name=case['robot']
            if name not in cache:
                tree=read_skeleton_tree(str(next((ROOT/'gallery').glob('**/'+name+'_ik.yaml'))))
                scope={'__name__':'empty_recovery_regression'}
                with redirect_stdout(StringIO()):
                    exec('\n'.join(CodeGenerationPython(tree).generate_code()),scope)
                cache[name]=(tree,scope)
            tree,scope=cache[name]
            with self.subTest(robot=name,index=case['index']):
                target=np.array(case['target'])
                np.testing.assert_allclose(independent_fk(tree.robot,case['q'],case['parameters']),target,atol=2e-14,rtol=0)
                arguments=[case['parameters'][str(symbol)] for symbol in tree.get_free_symbols()]
                output=scope[name+'_ik_solve'](target,*arguments)
                if case['recovery_expected']:self.assertTrue(output)
                for q in output:
                    self.assertLess(np.max(np.abs(independent_fk(tree.robot,q,case['parameters'])-target)),1e-7)
                    for index,joint in enumerate(tree.robot.unknowns):
                        if joint.symbol in tree.robot.unknown_as_parameter_more_dof:
                            self.assertEqual(float(q[index,0]),case['parameters'][str(joint.symbol)])

    def test_cli_can_export_legacy_paths_with_matching_runtimes(self):
        tree=ROOT/'gallery/ur_10/ur_10_ik.yaml'
        with tempfile.TemporaryDirectory(prefix='yaik-recovery-cli-') as temporary:
            output=Path(temporary)
            result=subprocess.run([sys.executable,'-B',str(ROOT/'yaik.py'),'codegen',str(tree),
                                   '--no-extra-recovery','--with-runtime','--output-dir',str(output)],
                                  capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIs(json.loads(result.stdout)['extra_recovery'],False)
            for name in ('python_run_import.py','yaik_cpp_common.h'):
                self.assertEqual((output/name).read_bytes(),(ROOT/name).read_bytes())
            scope={}
            exec((output/'ur_10_ik_generated.py').read_text(),scope)
            case=next(row for row in CASES if row['robot']=='ur_10' and row['recovery_expected'])
            self.assertEqual(scope['ur_10_ik_solve'](np.array(case['target'])),[])
            model=read_skeleton_tree(str(tree))
            with redirect_stdout(StringIO()):
                expected='\n'.join(line.rstrip() for line in CodeGenerationCpp(model,use_safe_operator=True,refine_seeds=False).generate_code())+'\n'
            self.assertEqual((output/'ur_10_ik_generated.cpp').read_text(),expected)

    @skipUnless(TOOLCHAIN,'Requires g++ and Eigen3')
    def test_cpp_empty_paths_keep_the_legacy_random_stream(self):
        sections=[];selected=[]
        for name in ('arm_robo','ur_10','pr2_r_gripper_palm'):
            tree=read_skeleton_tree(str(next((ROOT/'gallery').glob('**/'+name+'_ik.yaml'))))
            case=next(case for case in CASES if case['robot']==name and case['recovery_expected'])
            with redirect_stdout(StringIO()):
                code='\n'.join(CodeGenerationCpp(tree,use_safe_operator=True).generate_code()).split('// Code below for debug')[0]
            sections.append(code);selected.append((tree,case))
        sections.append('#include <cassert>\n#include <cstdlib>\n#include <iomanip>\nint main() {')
        for tree,case in selected:
            name=tree.robot.name;n=tree.robot.n_dofs
            matrix=','.join(format(value,'.17g') for row in case['target'] for value in row)
            args=''.join(','+repr(case['parameters'][str(symbol)]) for symbol in tree.get_free_symbols())
            geometry=''.join(','+repr(case['parameters'][str(symbol)]) for symbol in tree.get_free_symbols(False))
            sections.append('''{ using Model=NAME_ik;
Eigen::Matrix4d target;target << MATRIX;
Model::RawIKWorksace workspace;
std::vector<std::array<double,NQ>> previous;
Eigen::Matrix4d disturbance;bool ready=false;
std::srand(1);
Model::computeIKForRawTarget(target,Model::endEffectorTargetOriginalToRaw(target GEOMETRY) ARGS,workspace,previous,disturbance,ready,true);
const int expected=std::rand();std::srand(1);
const auto output=Model::computeIK(target ARGS);
assert(std::rand()==expected);
for(const auto& old:previous)assert(std::find(output.begin(),output.end(),old)!=output.end());
bool strict=false;
for(const auto& q:output)if((Model::computeFK(q GEOMETRY)-target).cwiseAbs().maxCoeff()<1e-7) {
  strict=true;for(double v:q)std::cout<<std::setprecision(17)<<v<<' ';std::cout<<'\\n';break;
}
assert(strict); }
'''.replace('NAME',name).replace('NQ',str(n)).replace('MATRIX',matrix).replace('GEOMETRY',geometry).replace('ARGS',args))
        sections.append('}')
        with tempfile.TemporaryDirectory(prefix='yaik-empty-native-') as temporary:
            source,binary=Path(temporary)/'test.cpp',Path(temporary)/'test'
            source.write_text('\n'.join(sections))
            result=subprocess.run(['g++','-std=c++14','-O2','-I',str(ROOT),'-I','/usr/include/eigen3',
                                   str(source),'-o',str(binary)],capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            lines=result.stdout.splitlines();self.assertEqual(len(lines),len(selected))
            for line,(tree,case) in zip(lines,selected):
                q=list(map(float,line.split()))
                self.assertLess(np.max(np.abs(independent_fk(tree.robot,q,case['parameters'])-case['target'])),1e-7)
