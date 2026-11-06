"""Finite raw root candidates recover strict witnesses without changing old paths."""
from contextlib import redirect_stdout
from io import StringIO
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import TestCase, skipUnless

import numpy as np
import sympy as sp

from codegen.cpp_codegen.codegen_cpp import CodeGenerationCpp
from codegen.py_codegen.codegen_py import CodeGenerationPython
from codegen.skeleton.tree_serialize import read_skeleton_tree
from python_run_import import refine_ik_seed, retain_refinement_seed

ROOT=Path(__file__).resolve().parents[1]
TOOLCHAIN=shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file()
NAME='yaskawa_mpx3500'


def parameterized_tree(tree):
    tree=copy.deepcopy(tree)
    tree.robot.name='mpx_parameterized_refinement'
    mount,shift=sp.symbols('refinement_mount refinement_shift')
    tree.robot.symbolic_parameters=set(tree.robot.symbolic_parameters)|{mount,shift}
    tree.robot.parameters_bound.update({mount:(-1.,1.),shift:(-1.,1.)})
    frame=sp.Matrix([[sp.cos(mount),-sp.sin(mount),0,shift],
                     [sp.sin(mount),sp.cos(mount),0,0],[0,0,1,0],[0,0,0,1]])
    tree.robot.auxiliary_data.pre_transform_sp=frame*tree.robot.get_pre_transform_sp()
    return tree


def parameter_frame():
    c,s=math.cos(.3),math.sin(.3)
    return np.array([[c,-s,0,-.2],[s,c,0,0],[0,0,1,0],[0,0,0,1.]])


class TestSeedRefinement(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree=read_skeleton_tree(str(ROOT/'gallery'/NAME/(NAME+'_ik.yaml')))
        cls.cases=json.loads((ROOT/'solver/fixtures/mpx3500_raw_refinement_cases.json').read_text())['cases']
        robot=cls.tree.robot
        cls.offsets=np.array(robot.get_unknown_offset() or [0.]*robot.n_dofs,dtype=float)
        fields=[sp.sympify(value).subs(robot.parameters_value) for dh in robot.dh_params for value in dh.flatten()]
        cls.evaluate=staticmethod(sp.lambdify([joint.symbol for joint in robot.unknowns],fields,'numpy'))
        pre,post=robot.get_pre_transform_sp(),robot.get_post_transform_sp()
        cls.pre=np.eye(4) if pre is None else np.array(pre.subs(robot.parameters_value),dtype=float)
        cls.post=np.eye(4) if post is None else np.array(post.subs(robot.parameters_value),dtype=float)

    @classmethod
    def independent_fk(cls,q):
        transform=cls.pre.copy()
        fields=cls.evaluate(*(np.asarray(q).reshape(-1)+cls.offsets))
        for start in range(0,len(fields),4):
            alpha,a,d,theta=fields[start:start+4]
            ca,sa,ct,st=math.cos(alpha),math.sin(alpha),math.cos(theta),math.sin(theta)
            transform=transform@np.array([[ct,-st,0,a],[ca*st,ca*ct,-sa,-d*sa],
                                         [sa*st,sa*ct,ca,d*ca],[0,0,0,1.]])
        return transform@cls.post

    def test_saved_numerical_exports_match_trees_sources_and_runtimes(self):
        from utility.artifacts import generator_source_hashes
        checked=[]
        sources=generator_source_hashes()
        for path in sorted((ROOT/'gallery').glob('**/*_ik.yaml')):
            provenance=path.parent/'provenance.json'
            report=json.loads(provenance.read_text())
            folder=provenance.parent
            tree=read_skeleton_tree(str(path))
            self.assertEqual(report.get('generator_sources_sha256',report.get('generator_overrides_sha256')),sources)
            for name,expected in report['runtime_sha256'].items():
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected)
            for name,expected in report['artifact_sha256'].items():
                self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),expected)
            for extension,generator in (('py',CodeGenerationPython(tree)),('cpp',CodeGenerationCpp(tree,use_safe_operator=True))):
                with redirect_stdout(StringIO()):
                    expected='\n'.join(line.rstrip() for line in generator.generate_code())+'\n'
                self.assertEqual((folder/(tree.robot.name+'_ik_generated.'+extension)).read_text(),expected)
            checked.append(tree.robot.name)
        self.assertIn(NAME,checked)

    def test_python_recovers_recorded_targets_and_retains_old_candidates(self):
        scope={'__name__':'raw_seed_regression'}
        with redirect_stdout(StringIO()):
            exec('\n'.join(CodeGenerationPython(self.tree).generate_code()),scope)
        for case in self.cases:
            with self.subTest(index=case['index']):
                target=np.array(case['target'])
                np.testing.assert_allclose(self.independent_fk(case['q']),target,atol=2e-14,rtol=0)
                old=scope[NAME+'_ik_solve_for_raw_target'](target,scope[NAME+'_ik_target_original_to_raw'](target))
                output=scope[NAME+'_ik_solve'](target)
                self.assertTrue(any(np.max(np.abs(self.independent_fk(q)-target))<1e-7 for q in output))
                self.assertTrue(all(any(np.array_equal(previous,q) for q in output) for previous in old))

    def test_runtime_frame_parameters_reach_the_refinement_fk(self):
        scope={'__name__':'parameterized_raw_seed_regression'}
        with redirect_stdout(StringIO()):
            exec('\n'.join(CodeGenerationPython(parameterized_tree(self.tree)).generate_code()),scope)
        frame=parameter_frame()
        for case in self.cases:
            with self.subTest(index=case['index']):
                target=frame@np.array(case['target'])
                output=scope['mpx_parameterized_refinement_ik_solve'](target,.3,-.2)
                self.assertTrue(any(np.max(np.abs(frame@self.independent_fk(q)-target))<1e-7 for q in output))

    def test_helper_preserves_fixed_coordinates_and_rejects_unreachable_nonfinite_inputs(self):
        calls=[]
        def fk(q):
            calls.append(tuple(q))
            pose=np.eye(4)
            c,s=math.cos(q[0]),math.sin(q[0])
            pose[:2,:2]=[[c,-s],[s,c]]
            pose[2,3]=q[1]
            return pose
        target=fk([.4,-0.]);calls.clear()
        seed=np.array([.1,-0.])
        result=refine_ik_seed(fk,target,seed,[False,True])
        self.assertIsNotNone(result)
        self.assertTrue(np.signbit(result[1]))
        self.assertTrue(all(math.copysign(1.,q[1])<0 for q in calls))
        np.testing.assert_array_equal(seed,[.1,-0.])
        invalid=target.copy();invalid[0,3]=.01
        self.assertIsNone(refine_ik_seed(fk,invalid,seed,[False,True]))
        for value in (math.nan,math.inf,-math.inf):
            self.assertIsNone(refine_ik_seed(fk,target,[value,-0.],[False,True]))
            invalid=target.copy();invalid[3,0]=value
            self.assertIsNone(refine_ik_seed(fk,invalid,seed,[False,True]))
        invalid=target.copy();invalid[3,0]=.01
        self.assertIsNone(refine_ik_seed(fk,invalid,seed,[False,True]))
        self.assertIsNone(refine_ik_seed(fk,target,seed,[True,True]))
        def enormous(q):
            pose=np.eye(4);pose[0,3]=q[0]*1e308;return pose
        self.assertIsNone(refine_ik_seed(enormous,np.eye(4),[1.]))
        def discontinuous(q):
            pose=np.eye(4);pose[0,3]=1. if abs(q[0])<1e-7 else math.copysign(1e308,q[0]);return pose
        self.assertIsNone(refine_ik_seed(discontinuous,np.eye(4),[0.]))
        seeds=[]
        for value,error in ((1.,4.),(2.,3.),(3.,2.),(4.,1.),(5.,.5),(5.,.5),(math.nan,0.)):
            retain_refinement_seed(seeds,[value],error)
        self.assertEqual([entry[0] for entry in seeds],[.5,1.,2.,3.])
        self.assertEqual(len(seeds),4)

    @skipUnless(TOOLCHAIN,'Requires g++ and Eigen3')
    def test_cpp_recovers_targets_preserves_candidates_and_random_stream(self):
        with redirect_stdout(StringIO()):
            code='\n'.join(CodeGenerationCpp(self.tree,use_safe_operator=True).generate_code()).split('// Code below for debug')[0]
            code+='\n'+'\n'.join(CodeGenerationCpp(parameterized_tree(self.tree),use_safe_operator=True).generate_code()).split('// Code below for debug')[0]
        main=r'''
#include <cassert>
#include <cstdlib>
#include <iostream>
int main() {
    using Model=yaskawa_mpx3500_ik;
    int count=0;
    Eigen::Matrix4d target;
    while(std::cin>>target(0,0)) {
        for(int i=1;i<16;++i)std::cin>>target(i/4,i%4);
        Model::RawIKWorksace workspace;
        std::vector<std::array<double,6>> old;
        Eigen::Matrix4d disturbance;
        bool ready=false;
        std::srand(1);
        Model::computeIKForRawTarget(target,Model::endEffectorTargetOriginalToRaw(target),workspace,old,disturbance,ready,true);
        const int expected=std::rand();
        std::srand(1);
        const auto output=Model::computeIK(target);
        assert(std::rand()==expected);
        assert(std::all_of(old.begin(),old.end(),[&](const std::array<double,6>& q){
            return std::find(output.begin(),output.end(),q)!=output.end();}));
        bool strict=false;
        for(const auto& q:output)if((Model::computeFK(q)-target).cwiseAbs().maxCoeff()<1e-7) {
            strict=true;
            for(double value:q)std::cout<<std::setprecision(17)<<value<<' ';
            std::cout<<'\n';break;
        }
        assert(strict);++count;
        Eigen::Matrix4d frame=Eigen::Matrix4d::Identity();
        frame.block<3,3>(0,0)=Eigen::AngleAxisd(.3,Eigen::Vector3d::UnitZ()).toRotationMatrix();
        frame(0,3)=-.2;
        const auto transformed=frame*target;
        const auto framed_output=mpx_parameterized_refinement_ik::computeIK(transformed,.3,-.2);
        strict=false;
        for(const auto& q:framed_output)if((mpx_parameterized_refinement_ik::computeFK(q,.3,-.2)-transformed).cwiseAbs().maxCoeff()<1e-7) {
            strict=true;
            for(double value:q)std::cout<<std::setprecision(17)<<value<<' ';
            std::cout<<'\n';break;
        }
        assert(strict);
    }
    assert(count==33);
    auto fk=[](const std::array<double,2>& q) {
        Eigen::Matrix4d pose=Eigen::Matrix4d::Identity();
        pose(2,3)=q[0];pose(0,3)=q[1];return pose;
    };
    auto seed=std::array<double,2>{{.1,-0.}};
    const auto target2=fk({{.4,-0.}});
    std::srand(3);const int expected=std::rand();std::srand(3);
    assert(yaik_cpp::refineIkSeed<2>(fk,target2,seed,{{false,true}}));
    assert(std::signbit(seed[1]) && std::rand()==expected);
    const auto previous=seed;
    auto impossible=target2;impossible(1,3)=1.;
    assert(!yaik_cpp::refineIkSeed<2>(fk,impossible,seed,{{false,true}}) && seed==previous);
    seed[0]=std::numeric_limits<double>::infinity();
    assert(!yaik_cpp::refineIkSeed<2>(fk,target2,seed,{{false,true}}));
}
'''
        with tempfile.TemporaryDirectory(prefix='yaik-raw-refinement-') as temporary:
            temporary=Path(temporary)
            source,binary=temporary/'regression.cpp',temporary/'regression'
            source.write_text('#include <iomanip>\n'+code+main)
            result=subprocess.run(['g++','-std=c++14','-O2','-I',str(ROOT),'-I','/usr/include/eigen3',
                                   str(source),'-o',str(binary)],capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stderr)
            inputs='\n'.join(' '.join(format(value,'.17g') for row in case['target'] for value in row)
                             for case in self.cases)+'\n'
            result=subprocess.run([str(binary)],input=inputs,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            returned=[list(map(float,line.split())) for line in result.stdout.splitlines()]
            self.assertEqual(len(returned),2*len(self.cases))
            frame=parameter_frame()
            for index,case in enumerate(self.cases):
                self.assertLess(np.max(np.abs(self.independent_fk(returned[2*index])-case['target'])),1e-7)
                self.assertLess(np.max(np.abs(frame@self.independent_fk(returned[2*index+1])-frame@case['target'])),1e-7)
