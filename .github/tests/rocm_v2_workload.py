"""Captured-workload HIP checks using the existing GGML test and campaign runners."""
import argparse,copy,csv,hashlib,json,math,re,statistics,subprocess
from pathlib import Path
from rocm_v2_qwen import manifest as qwen_manifest
from rocm_v2_campaign import validate_manifest
from rocm_v2_halo import summarize_models

VARIANTS={'native':('../source-baseline','baseline',False),'before':('../source-baseline','baseline',True),
          'full1':('.', 'full1',True),'full2':('../source-rows2','full2',True),'vector':('../source-vector','vector',True)}

def fixture_ids(path):
    return [line.rsplit(' ',1)[1] for line in path.read_text().splitlines() if line]

def prepare(root):
    import shutil
    helpers=Path(__file__).resolve().parent
    root.mkdir()
    subprocess.run(['git','init',str(root)],check=True)
    subprocess.run(['git','-C',str(root),'remote','add','origin','https://github.com/0xShug0/audio.cpp.git'],check=True)
    revision='f7a0982a3b70da5217535ff909eb9ec3a51dc8df'
    subprocess.run(['git','-C',str(root),'fetch','--no-tags','--depth=1','--filter=blob:none','origin',revision],check=True)
    subprocess.run(['git','-C',str(root),'sparse-checkout','set','include','src','app','docs','model_specs','tools/cmake','tests/perf','webui/native/demo_voices','external/ggml','external/cJSON','external/cpp-httplib','external/libyaml','external/llama_tokenizer','external/sentencepiece','external/utf8proc'],check=True)
    subprocess.run(['git','-C',str(root),'checkout','--detach','FETCH_HEAD'],check=True)
    (root/'source-commit.txt').write_text(revision+'\n')
    for patch in ['generic-gates.patch']:
        subprocess.run(['git','-C',str(root),'apply','--check',str(helpers/patch)],check=True)
        subprocess.run(['git','-C',str(root),'apply',str(helpers/patch)],check=True)
    baseline=root.parent/'source-baseline';shutil.copytree(root,baseline)
    subprocess.run(['git','-C',str(root),'apply','--check',str(helpers/'workload-kernels.patch')],check=True)
    subprocess.run(['git','-C',str(root),'apply',str(helpers/'workload-kernels.patch')],check=True)
    shutil.copy2(root/'external/ggml/tests/test-backend-ops.cpp',baseline/'external/ggml/tests/test-backend-ops.cpp')
    variants=[root,baseline]
    for name,patch in [('source-rows2','workload-rows2.patch'),('source-vector','workload-vector.patch')]:
        dest=root.parent/name;shutil.copytree(root,dest)
        subprocess.run(['git','-C',str(dest),'apply','--check',str(helpers/patch)],check=True)
        subprocess.run(['git','-C',str(dest),'apply',str(helpers/patch)],check=True)
        variants.append(dest)
    for directory in variants:
        with (directory/'CMakeLists.txt').open('a') as f:
            f.write('\nfind_package(Threads REQUIRED)\nadd_executable(test-backend-ops external/ggml/tests/test-backend-ops.cpp)\ntarget_link_libraries(test-backend-ops PRIVATE ggml Threads::Threads)\n')
    (root/'.github/tests').mkdir(parents=True)
    for path in helpers.glob('rocm_v2_*.py'):shutil.copy2(path,root/'.github/tests'/path.name)
    shutil.copy2(helpers/'hip_vmm_probe.cpp',root/'.github/tests/hip_vmm_probe.cpp')
    shutil.copytree(helpers/'workload-probes',root/'.github/tests/workload-probes')
    for name in ['generic-gates.patch','workload-kernels.patch','workload-rows2.patch','workload-vector.patch']:
        shutil.copy2(helpers/name,root/name)
    (root/'toolchain').mkdir()
    subprocess.run(['curl','--fail','--location','--max-time','120','--retry','2','-o','ccache.tar.xz','https://github.com/ccache/ccache/releases/download/v4.13.6/ccache-4.13.6-linux-x86_64-musl-static.tar.xz'],check=True)
    subprocess.run(['tar','-xJf','ccache.tar.xz','--strip-components=1','-C',str(root/'toolchain'),'ccache-4.13.6-linux-x86_64-musl-static/ccache'],check=True)
    (root/'inputs').mkdir()
    requests=json.loads((helpers/'workload-requests.json').read_text())
    records=json.loads((helpers/'workload-model-files.json').read_text())
    hashes=[r['sha256']+'  '+r['destination'] for r in records if r['sha256']]
    for r in records:(root/r['destination']).parent.mkdir(parents=True,exist_ok=True)
    for family,sequence in requests.items():
        path=root/'inputs'/(family+'-requests.json');path.write_text(json.dumps(sequence,ensure_ascii=False,indent=2)+'\n')
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.relative_to(root).as_posix())
    (root/'inputs/workload-requests.json').write_text(json.dumps(requests,ensure_ascii=False,indent=2)+'\n')
    (root/'inputs/model-provenance.json').write_text(json.dumps(records,indent=2)+'\n')
    (root/'inputs/input-sha256.txt').write_text('\n'.join(hashes)+'\n')
    subprocess.run(['cmake','-S',str(root/'external/ggml'),'-B',str(root/'build/debug'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Debug','-DCMAKE_CXX_STANDARD=17','-DCMAKE_CXX_FLAGS_DEBUG=-g1 -O2','-DCMAKE_C_FLAGS_DEBUG=-g1 -O2','-DBUILD_SHARED_LIBS=OFF','-DGGML_HIP=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF','-DGGML_BLAS=OFF','-DGGML_BUILD_TESTS=ON','-DGGML_BUILD_EXAMPLES=OFF'],check=True)
    subprocess.run(['cmake','--build',str(root/'build/debug'),'--target','test-backend-ops','--parallel','8'],check=True)
    with (root/'cpu-preflight.log').open('w') as log:
        subprocess.run([str(root/'build/debug/bin/test-backend-ops'),'test','-b','CPU','--output','csv','--test-file',str(root/'.github/tests/workload-probes/cpu-preflight.ops')],stdout=log,stderr=subprocess.STDOUT,check=True)
    actual=read_rows(root/'cpu-preflight.log')
    if set(actual)!=set(fixture_ids(root/'.github/tests/workload-probes/cpu-preflight.ops')):raise ValueError('CPU preflight skipped fixtures')
    for directory in variants:
        paths=[p for p in directory.rglob('*') if p.is_file() and not set(p.relative_to(directory).parts)&{'.git','build','inputs','toolchain'}]
        lines=[hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(directory).as_posix() for p in sorted(paths)]
        (directory/'workload-source-sha256.txt').write_text('\n'.join(lines)+'\n')
    patch_lines=[hashlib.sha256((helpers/n).read_bytes()).hexdigest()+'  '+n for n in ['generic-gates.patch','workload-kernels.patch','workload-rows2.patch','workload-vector.patch']]
    (root/'patch-sha256.txt').write_text('\n'.join(patch_lines)+'\n')
    (root/'campaign.json').write_text(json.dumps(manifest(root),indent=2)+'\n')
    validate_manifest(json.loads((root/'campaign.json').read_text()))
    subprocess.run(['tar','-czf','probe-source.tar.gz','--exclude=.git','--exclude=*/build',*[str(p) for p in variants]],check=True)


def manifest(root):
    plan=qwen_manifest(root); health=plan['builds'][0]
    health['commands'].insert(0,['cmake','-E','make_directory','build/debug'])
    builds=[health]
    for name in ['baseline','full1','full2','vector']:
        build=copy.deepcopy(plan['builds'][1]);build['id']=name
        if name!='full1':build['cwd']={'baseline':'../source-baseline','full2':'../source-rows2','vector':'../source-vector'}[name]
        build['commands'][0]=['-DAUDIOCPP_MODELS=breeze_tts,ming3_tts,omnivoice,qwen3_tts,irodori_tts,yue2' if x.startswith('-DAUDIOCPP_MODELS=') else x for x in build['commands'][0]]
        build['commands'][0]+=['-DBUILD_SHARED_LIBS=OFF']
        build['commands'][1].insert(build['commands'][1].index('--parallel'),'test-backend-ops')
        builds.append(build)
    fixtures=root/'.github/tests/workload-probes'
    checks=[dict(id='health',argv=['build/debug/device-health','health'])];mapping={}
    for variant,(cwd,build,selected) in VARIANTS.items():
        pattern='check-*.ops' if variant in ['native','full1'] else 'q8-check-*.ops'
        for path in sorted(fixtures.glob(pattern)):
            name=variant+'-'+path.stem
            argv=[cwd+'/build/debug/bin/test-backend-ops','test','-b','ROCm0','--output','csv','--test-file','.github/tests/workload-probes/'+path.name]
            if selected:argv+=['--backend-impl','gfx1151_v1']
            checks.append(dict(id=name,argv=argv,timeout_s=180))
            for case in fixture_ids(path):mapping[variant,case]='correctness/'+name
    timings=[]
    for repeat,order in enumerate([list(VARIANTS),list(reversed(VARIANTS))]):
        for variant in order:
            cwd,build,selected=VARIANTS[variant]
            pattern='perf-*.ops' if variant in ['native','full1'] else 'q8-perf-*.ops'
            for path in sorted(fixtures.glob(pattern)):
                argv=[cwd+'/build/debug/bin/test-backend-ops','perf','-b','ROCm0','--output','csv','--duration-ms','25','--test-file','.github/tests/workload-probes/'+path.name]
                if selected:argv+=['--backend-impl','gfx1151_v1']
                timings.append(dict(id=f'r{repeat}-{variant}-'+path.stem,argv=argv,
                    depends_on=sorted({mapping[variant,c] for c in fixture_ids(path)}),timeout_s=180))
    suites=[dict(id='correctness',gpu=True,cases=checks),dict(id='operators',gpu=True,cases=timings),
            dict(id='framework',gpu=False,timeout_s=180,cases=[dict(id='select',argv=['python3','.github/tests/rocm_v2_workload.py','--root','../results','--source','.','--select'])])]
    traces=[]
    for variant,(cwd,build,selected) in VARIANTS.items():
        argv=['/opt/rocm/bin/rocprofv3','--kernel-trace','--output-format','csv',
              '--output-directory','../results/profile/'+variant,'--',
              cwd+'/build/debug/bin/test-backend-ops','test','-b','ROCm0','--output','csv',
              '--test-file','.github/tests/workload-probes/dispatch.ops']
        if selected:argv+=['--backend-impl','gfx1151_v1']
        traces.append(dict(id=variant,argv=argv,timeout_s=180))
    suites.append(dict(id='dispatch',gpu=True,cases=traces))
    return validate_manifest(dict(version=1,builds=builds,suites=suites,health_check=plan['health_check']))

def read_rows(path):
    lines=[line for line in path.read_text().splitlines() if line.startswith('"')]
    if not lines:raise ValueError('No structured results: '+str(path))
    rows=list(csv.DictReader(lines));result={}
    for row in rows:
        match=re.search(r'\bname=([^,)]+)',row['op_params'])
        if not match:raise ValueError('Missing fixture ID: '+str(path))
        key=match[1]
        if key in result:raise ValueError('Duplicate fixture result: '+str(path)+' '+key)
        if row['supported']!='1' or row['passed']!='1':raise ValueError('Unsupported/failed fixture: '+str(path)+' '+key)
        result[key]=row
    return result

def select(root,source):
    catalog=json.loads((source/'.github/tests/workload-probes/cases.json').read_text());by_id={r['id']:r for r in catalog}
    results=json.loads((root/'summary.json').read_text());bad=[r for r in results['results'] if r['status']!='PASS']
    if bad:raise ValueError('Failed campaign commands: '+json.dumps([(r['id'],r['status']) for r in bad]))
    plan=json.loads((source/'campaign.json').read_text());expected={}
    for suite in plan['suites']:
        if suite['id'] not in ['correctness','operators']:continue
        for case in suite['cases']:
            if '--test-file' not in case['argv']:continue
            path=source/case['argv'][case['argv'].index('--test-file')+1]
            expected[suite['id'],case['id']]=set(fixture_ids(path))
    timing={};correctness={}
    for (suite,name),ids in expected.items():
        data=read_rows(root/'case'/suite/(name+'.log'))
        if set(data)!=ids:raise ValueError('Missing/extra fixture rows: '+suite+'/'+name)
        if suite=='correctness':
            variant=next(v for v in VARIANTS if name.startswith(v+'-'))
            correctness.setdefault(variant,set()).update(data)
        else:
            match=re.match(r'r(\d+)-(native|before|full1|full2|vector)-',name);variant=match[2]
            for key,row in data.items():
                value=float(row['time_us'])
                if not math.isfinite(value) or value<=0 or int(row['n_runs'])<=0:raise ValueError('Missing timing: '+name+' '+key)
                timing.setdefault(variant,{}).setdefault(key,[]).append(value)
    for variant,data in timing.items():
        if any(len(v)!=2 for v in data.values()):raise ValueError('Incomplete repetitions: '+variant)
    common=set(timing['full1']) & set(timing['native'])
    non_q8=[];operator_rows=[]
    for key in sorted(common):
        n=timing['native'][key];v=timing['full1'][key]
        loss=min(v)>max(n)*1.05
        operator_rows.append(dict(id=key,group=by_id[key]['group'],models=by_id[key]['models'],
            native_us=statistics.median(n),candidate_us=statistics.median(v),
            speedup=statistics.median(n)/statistics.median(v),clear_regression=loss))
        if key not in timing['full2'] and loss:non_q8.append(key)
    (root/'operator-comparison.json').write_text(json.dumps(operator_rows,indent=2)+'\n')
    q8=set(timing['full2']);scores={};detail=[]
    for variant in ['full1','full2','vector']:
        ratios=[];losses=[];captured_losses=[];stream=[]
        for key in sorted(q8):
            n=timing['native'][key];v=timing[variant][key]
            ratio=statistics.median(n)/statistics.median(v);ratios.append(ratio)
            real=bool(by_id[key]['models'])
            if min(v)>max(n)*1.05:
                losses.append(key)
                if real:captured_losses.append(key)
            if any('weight bank' in x for x in by_id[key]['reasons']):stream.append(ratio)
            detail.append(dict(variant=variant,id=key,native_us=statistics.median(n),candidate_us=statistics.median(v),speedup=ratio,captured=real))
        scores[variant]=dict(geomean_speedup=math.exp(statistics.mean(map(math.log,ratios))),
            captured_regressions=captured_losses,regressions=losses,streaming_geomean_speedup=math.exp(statistics.mean(map(math.log,stream))))
    ranked=sorted(scores,key=lambda v:(-len(scores[v]['regressions']),scores[v]['geomean_speedup']),reverse=True)
    winner=ranked[0]
    accepted=scores[winner]['geomean_speedup']>1.0 and not scores[winner]['regressions'] and not non_q8
    report=dict(candidate=winner,accepted=accepted,correctness_verified=True,scores=scores,non_q8_regressions=non_q8,correctness_counts={k:len(v) for k,v in correctness.items()},
                timed_counts={k:len(v) for k,v in timing.items()},samples_per_shape=2)
    (root/'selected-variant.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'q8-comparison.json').write_text(json.dumps(detail,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not accepted:raise ValueError('No candidate passes the complete operator performance gate')


def model_manifest(root,source):
    selected=json.loads((root/'selected-variant.json').read_text())
    if not selected.get('correctness_verified'):raise ValueError('Operator correctness gate did not pass')
    variant=selected['candidate'];cwd=VARIANTS[variant][0]
    requests=json.loads((source/'inputs/workload-requests.json').read_text());suites=[]
    for family in requests:
        for label in ['a1','b1','b2','a2']:
            candidate=label.startswith('b');binary=(cwd if candidate else '../source-baseline')+'/build/debug/bin/audiocpp_cli'
            model='inputs/yue2' if family=='yue2' else 'inputs/'+family+'.gguf'
            task='gen' if family=='yue2' else 'vdes' if family=='qwen3_tts' else 'tts'
            argv=[binary,'--task',task,'--family',family,'--model',model,'--backend','hip','--threads','8',
                  '--request-sequence','inputs/'+family+'-requests.json','--out-dir','../results/models/wavs/'+family+'-'+label,'--log','--metrics']
            if candidate:argv+=['--backend-impl','gfx1151_v1']
            suites.append(dict(id=family+'-'+label,gpu=True,timeout_s=900,cases=[dict(id='session',argv=argv)]))
    suites.append(dict(id='report',gpu=False,timeout_s=300,cases=[dict(id='models',argv=['python3','.github/tests/rocm_v2_workload.py','--root','../results/models','--source','.','--summarize'])]))
    return validate_manifest(dict(version=1,builds=[],suites=suites,health_check=dict(argv=['build/debug/device-health','health'],timeout_s=15)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,default=Path('.'))
    action=p.add_mutually_exclusive_group(required=True)
    for name in ['prepare','manifest','select','model-manifest','summarize']:action.add_argument('--'+name,action='store_true')
    args=p.parse_args()
    if args.prepare:prepare(args.root)
    elif args.manifest:(args.root/'campaign.json').write_text(json.dumps(manifest(args.root),indent=2)+'\n')
    elif args.select:select(args.root,args.source)
    elif args.model_manifest:(args.source/'workload-model-campaign.json').write_text(json.dumps(model_manifest(args.root,args.source),indent=2)+'\n')
    else:summarize_models(args.root,baseline_implementation='default',candidate_implementation='gfx1151_v1',request_sets=json.loads((args.source/'inputs/workload-requests.json').read_text()))

if __name__=='__main__':main()
