"""Bounded development-only CRAFT-inspired Stage 3 analysis."""
from __future__ import annotations
import csv, hashlib, io, json, math, os, sys, time, traceback, unittest, warnings
PROCESS_START=time.perf_counter()
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.stats import qmc, spearmanr
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results_paper/phase_6d_craft_stage3_v1'
BASE=ROOT/'results_paper/phase_6d_craft_stage1_v1'
B2=ROOT/'results_paper/phase_6d_craft_stage2b_v1'
CACHE=ROOT/'results_paper/splits/ao_annotation_only_v1'
CK=ROOT/'results_paper/AO_confirmation_v1/AO-3/seed_20260928/best_model.pt'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
from MTN.ao_models_v1 import build_ao_model, constant_velocity_forecast
from MTN.run_ao_v1 import _read_split, _train_only_normalization

BUDGET=1800.0; SEEDS=(20260928,20260929); SOBOL_SEED=20611
RANDOM_ORDERS=(3101,3102,3103,3104,3105)
OUTPUTS=('final_center_x','final_center_y','final_displacement_norm','cv_correction_energy')
UNITS={'final_center_x':'pixels','final_center_y':'pixels','final_displacement_norm':'pixels','cv_correction_energy':'pixels_squared'}
START=PROCESS_START

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def check_budget():
    if time.perf_counter()-START>BUDGET: raise TimeoutError('30-minute Stage 3 budget exhausted')
def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def atomic_csv(path,rows,fields=None):
    if fields is None: fields=list(rows[0]) if rows else []
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def atomic_npz(path,**arrays):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def mark(stage,**meta): atomic_json(OUT/'status.json',{'state':stage,'elapsed_seconds':time.perf_counter()-START,'budget_seconds':BUDGET,**meta})
def log(msg):
    print(msg,flush=True)
    with (OUT/'execution.log').open('a',encoding='utf-8') as f:f.write(msg+'\n');f.flush();os.fsync(f.fileno())
def select_top_tokens(coeff,fraction=.10):
    a=np.asarray(coeff).reshape(-1);n=int(math.ceil(len(a)*fraction));return np.sort(np.argsort(-a,kind='stable')[:n])
def coeff_activation(u,w): return np.einsum('...k,kf->...f',np.asarray(u),np.asarray(w))
def bh_adjust(pvalues):
    p=np.asarray(pvalues,float);q=np.full(len(p),np.nan);finite=np.flatnonzero(np.isfinite(p));n=len(finite);order=finite[np.argsort(p[finite],kind='stable')];running=1.0
    for j in range(n-1,-1,-1):
        i=order[j];running=min(running,float(p[i])*n/(j+1));q[i]=running
    return np.clip(q,0,1)
def jansen_total(y_a,y_b,y_ab):
    var=float(np.var(np.r_[y_a,y_b]))
    if not np.isfinite(var) or var<=1e-12:return np.full(y_ab.shape[0],np.nan),var
    return np.mean((y_a[None,:]-y_ab)**2,axis=1)/(2*var),var
def unit_cosine_stability(w1,w2):
    a=w1/np.maximum(np.linalg.norm(w1,axis=1,keepdims=True),1e-30);b=w2/np.maximum(np.linalg.norm(w2,axis=1,keepdims=True),1e-30)
    sim=a@b.T;i,j=linear_sum_assignment(-sim);return float(sim[i,j].mean()),float(sim[i,j].min())
def scalar_outputs(pred_raw,x_norm,mean,std):
    obs=x_norm*std+mean
    # Exact NumPy equivalent of the frozen model utility's CV anchor: last box
    # plus future steps times the two-step finite-difference velocity.
    velocity=(x_norm[:,-1,:]-x_norm[:,-3,:])/2.0
    steps=np.arange(1,46,dtype=x_norm.dtype).reshape(1,45,1)
    cv=(x_norm[:,-1:,:]+steps*velocity[:,None,:])*std+mean
    centers=(pred_raw[...,:2]+pred_raw[...,2:])/2
    last=(obs[:,-1,:2]+obs[:,-1,2:])/2
    return {'final_center_x':centers[:,-1,0],'final_center_y':centers[:,-1,1],
            'final_displacement_norm':np.linalg.norm(centers[:,-1]-last,axis=1),
            'cv_correction_energy':np.mean((pred_raw-cv)**2,axis=(1,2))}
def make_model():
    payload=torch.load(CK,map_location='cpu',weights_only=False);m=build_ao_model('AO-3').eval();m.load_state_dict(payload['model_state'],strict=True);return m
def run_tests():
    import test_stage3_helpers
    s=io.StringIO();res=unittest.TextTestRunner(stream=s,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_stage3_helpers))
    detail=s.getvalue();(OUT/'REGRESSION_TESTS.txt').write_text(detail,encoding='utf-8')
    if not res.wasSuccessful():raise AssertionError('Focused Stage 3 regression tests failed:\n'+detail)
    return {'tests_run':res.testsRun,'failures':len(res.failures),'errors':len(res.errors),'passed':res.wasSuccessful()}
class Canvas:
    """Small Pillow/vector-SVG chart surface; Pillow also exports PDF."""
    def __init__(self,title,width=900,height=560):
        self.w=width;self.h=height;self.im=Image.new('RGB',(width,height),'white');self.d=ImageDraw.Draw(self.im);self.font=ImageFont.load_default();self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'];self.text(width//2,18,title,'#111111',anchor='middle',size=18)
    def text(self,x,y,s,color='#222222',anchor='start',size=12):
        s=str(s);pil_anchor={'start':'la','middle':'mm','end':'ra'}.get(anchor,'la');self.d.text((int(x),int(y)),s,fill=color,font=self.font,anchor=pil_anchor)
        import html
        self.svg.append(f'<text x="{x}" y="{y+10}" fill="{color}" text-anchor="{anchor}" font-family="sans-serif" font-size="{size}">{html.escape(s)}</text>')
    def line(self,x1,y1,x2,y2,color='#222222',width=1):
        self.d.line((int(x1),int(y1),int(x2),int(y2)),fill=color,width=width);self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')
    def poly(self,pts,color='#2474a6',width=2):
        if len(pts)>1:self.d.line([(int(x),int(y)) for x,y in pts],fill=color,width=width);self.svg.append('<polyline fill="none" stroke="'+color+f'" stroke-width="{width}" points="'+' '.join(f'{x},{y}' for x,y in pts)+'"/>')
    def rect(self,x1,y1,x2,y2,color):
        self.d.rectangle((int(x1),int(y1),int(x2),int(y2)),fill=color);self.svg.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="{color}"/>')
    def circle(self,x,y,r,color):
        self.d.ellipse((int(x-r),int(y-r),int(x+r),int(y+r)),fill=color);self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')
    def save(self,stem,manifest,description):
        self.im.save(OUT/f'{stem}.png');self.im.convert('RGB').save(OUT/f'{stem}.pdf',format='PDF',resolution=150)
        (OUT/f'{stem}.svg').write_text(''.join(self.svg)+'</svg>',encoding='utf-8')
        for ext in ('png','svg','pdf'):manifest.append({'figure':f'{stem}.{ext}','format':ext,'description':description})

def main():
    global START
    lock=OUT/'run.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'start':time.time()}).encode());os.close(fd)
    marks={'official_test_accessed':False,'model_training':False,'activation_extraction_repeated':False}
    try:
        torch.set_num_threads(4);threadpool_limits(limits=4);np.random.seed(20260928);torch.manual_seed(20260928)
        mark('started',**marks);tests=run_tests();log(f'focused regression tests: {tests}')
        actpath=BASE/'AO3_FFN_activations.npz';bankpath=B2/'SELECTED_FULL_TRAINING_BANK.npz'
        stage1_hash={r['artifact']:r['sha256'] for r in csv.DictReader((BASE/'artifact_hashes.csv').open())}
        assert sha(actpath).lower()==stage1_hash['AO3_FFN_activations.npz'].lower()
        assert sha(BASE/'activation_audit.json').lower()==stage1_hash['activation_audit.json'].lower()
        with (B2/'ARTIFACT_HASHES.csv').open(encoding='utf-8-sig',newline='') as handle:
            bmanifest={r['artifact']:r['sha256'] for r in csv.DictReader(handle)}
        assert sha(bankpath).lower()==bmanifest['results_paper/phase_6d_craft_stage2b_v1/SELECTED_FULL_TRAINING_BANK.npz'].lower()
        audit=json.loads((BASE/'activation_audit.json').read_text());assert sha(CK).lower()==audit['checkpoint_sha256'].lower()
        train,val,splitman=_read_split(CACHE);assert splitman['official_test_sets_read']==[]
        with np.load(actpath) as z:tr_act=z['train'].astype(np.float32);va_act=z['validation'].astype(np.float32);trids=z['train_ids'].astype(str);vaids=z['validation_ids'].astype(str)
        assert np.array_equal(trids,train['sample_id'].astype(str)) and np.array_equal(vaids,val['sample_id'].astype(str))
        assert tr_act.shape==(988,15,128) and va_act.shape==(432,15,128) and tr_act.min()>=0 and va_act.min()>=0
        with np.load(bankpath) as z:seeds=z['seeds'];banks=z['components'];tc=z['train_coefficients'];vc=z['validation_coefficients']
        assert tuple(seeds.tolist())==SEEDS and banks.shape==(2,4,128) and tc.shape==(2,988,15,4) and vc.shape==(2,432,15,4)
        W=banks[0].astype(np.float64);Utr=tc[0].astype(np.float64);Uva=vc[0].astype(np.float64)
        assert all(np.isfinite(a).all() and (a>=0).all() for a in (W,Utr,Uva))
        inputs=[actpath,BASE/'activation_audit.json',bankpath,CACHE/'train.npz',CACHE/'validation.npz',CACHE/'split_manifest.json',CK]
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
        mark('inputs_verified',**marks,input_hashes=input_hashes);log('saved activation, bank, split and checkpoint hashes verified')

        # Balanced sample: 16 lexicographically first sample IDs from each GP7 video.
        vvid=val['video_id'].astype(str);videos=sorted(set(vvid));assert len(videos)==2
        selected=[]
        for v in videos:
            ix=np.flatnonzero(vvid==v);ix=ix[np.argsort(vaids[ix],kind='stable')];selected+=ix[:16].tolist()
        selected=np.asarray(selected,int);selected_set=set(selected.tolist())
        atomic_csv(OUT/'VALIDATION_SAMPLE_MANIFEST.csv',[{'validation_row':i,'video_id':vvid[i],'sample_id':str(vaids[i]),'selected':i in selected_set} for i in range(len(vaids))])

        # Predefined observation-derived motion descriptors and BH correction.
        boxes=train['observed_boxes'].astype(np.float64);centers=(boxes[:,:,:2]+boxes[:,:,2:])/2;vel=np.diff(centers,axis=1);acc=np.diff(centers,n=2,axis=1)
        angles=np.arctan2(vel[:,:,1],vel[:,:,0]);turn=np.abs(np.arctan2(np.sin(np.diff(angles,axis=1)),np.cos(np.diff(angles,axis=1))))
        valid=(np.linalg.norm(vel[:,:-1],axis=2)>1e-6)&(np.linalg.norm(vel[:,1:],axis=2)>1e-6)
        direction=np.divide((turn*valid).sum(axis=1),valid.sum(axis=1),out=np.zeros(len(centers)),where=valid.sum(axis=1)>0)
        rootarea=np.sqrt(np.maximum(boxes[:,:,2]-boxes[:,:,0],0)*np.maximum(boxes[:,:,3]-boxes[:,:,1],0))
        desc={'horizontal_displacement_px':centers[:,-1,0]-centers[:,0,0],
              'vertical_displacement_px':centers[:,-1,1]-centers[:,0,1],
              'mean_velocity_px_per_observed_step':np.linalg.norm(vel,axis=2).mean(axis=1),
              'mean_acceleration_px_per_step2':np.linalg.norm(acc,axis=2).mean(axis=1),
              'mean_direction_change_rad':direction,
              'bbox_scale_cv':np.std(rootarea,axis=1)/np.maximum(np.mean(rootarea,axis=1),1e-12)}
        assoc=[];pvals=[];unique_tracks=len(set(zip(train['video_id'].astype(str),train['track_id'].astype(str))))
        for k in range(4):
            score=Utr[:,:,k].mean(axis=1)
            for name,d in desc.items():
                rho,p=spearmanr(score,d);assoc.append({'concept':f'C{k+1}','descriptor':name,'n_windows':len(score),'unique_video_track_groups':unique_tracks,'spearman_rho':float(rho),'p_value':float(p),'bh_q_value':None,'association_flag':False,'semantic_label':'numeric only'});pvals.append(p)
        qs=bh_adjust(pvals)
        for r,q in zip(assoc,qs):r['bh_q_value']=float(q);r['association_flag']=bool(abs(r['spearman_rho'])>=.30 and q<.05)
        atomic_csv(OUT/'MOTION_ASSOCIATIONS.csv',assoc)

        # Parent top-decile token identities and one minimal earlier-layer capture on unique windows.
        top=[select_top_tokens(Utr[:,:,k]) for k in range(4)];toprows=[]
        for k,ids in enumerate(top):
            for f in ids:
                wi,ti=divmod(int(f),15);toprows.append({'concept':f'C{k+1}','train_row':wi,'sample_id':str(trids[wi]),'video_id':str(train['video_id'][wi]),'track_id':str(train['track_id'][wi]),'temporal_index_zero_based':ti,'observed_step_one_based':ti+1,'parent_coefficient':float(Utr[wi,ti,k])})
        atomic_csv(OUT/'PARENT_TOP_TOKEN_MANIFEST.csv',toprows)
        unique_windows=np.unique(np.concatenate([x//15 for x in top]));norm=_train_only_normalization(train)
        mean=torch.as_tensor(norm['mean'],dtype=torch.float32);std=torch.as_tensor(norm['std'],dtype=torch.float32)
        xtr=(torch.as_tensor(train['observed_boxes'],dtype=torch.float32)-mean)/std
        model=make_model();capt={};early=np.empty((len(unique_windows),15,128),dtype=np.float32)
        hook=model.encoder_layers[1].feedforward[1].register_forward_hook(lambda _m,_a,o:capt.__setitem__('a',o.detach().cpu().numpy().copy()))
        with torch.no_grad():
            for s in range(0,len(unique_windows),32):
                ix=unique_windows[s:s+32];_=model(xtr[ix]);early[s:s+len(ix)]=capt['a'];check_budget()
        hook.remove();assert early.min()>=0 and np.isfinite(early).all();emap={int(w):j for j,w in enumerate(unique_windows)}
        recrows=[];childW=[];childH=[];child_ok=[]
        for k,ids in enumerate(top):
            wi=ids//15;ti=ids%15;X=np.stack([early[emap[int(a)],int(b)] for a,b in zip(wi,ti)]).astype(np.float64);Ws=[];Hs=[]
            for seed in SEEDS:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning);nmf=NMF(n_components=2,init='nndsvdar',solver='cd',random_state=seed,max_iter=200,tol=1e-3);H=nmf.fit_transform(X);cw=nmf.components_
                warn=any(issubclass(z.category,ConvergenceWarning) for z in caught);msg=' | '.join(str(z.message) for z in caught if issubclass(z.category,ConvergenceWarning));err=float(np.linalg.norm(X-H@cw)/np.linalg.norm(X));Ws.append(cw);Hs.append(H)
                recrows.append({'parent_concept':f'C{k+1}','seed':seed,'selected_tokens':len(X),'earlier_layer':'encoder_layers.1.feedforward.1_relu','rank':2,'relative_fit_error':err,'zero_coefficient_fraction':float(np.mean(H==0)),'iterations':int(nmf.n_iter_),'convergence_warning':warn,'warning_text':msg})
            stab,mn=unit_cosine_stability(Ws[0],Ws[1]);ok=all(not r['convergence_warning'] and r['relative_fit_error']<=.50 for r in recrows[-2:]) and stab>=.90;child_ok.append(ok);childW.append(np.stack(Ws));childH.append(np.stack(Hs))
            recrows.append({'parent_concept':f'C{k+1}','seed':'stability','selected_tokens':len(X),'earlier_layer':'encoder_layers.1.feedforward.1_relu','rank':2,'relative_fit_error':'','zero_coefficient_fraction':'','iterations':'','convergence_warning':False,'warning_text':f'matched_mean_cosine={stab:.9f};minimum={mn:.9f};supported={ok}'})
            check_budget()
        atomic_csv(OUT/'RECURSIVE_CONCEPT_RESULTS.csv',recrows)
        atomic_npz(OUT/'RECURSIVE_CHILD_BANKS.npz',components=np.stack(childW),coefficients=np.stack(childH),parent_top_token_indices=np.stack(top),earlier_layer_window_indices=unique_windows)
        mark('recursive_and_motion_complete',**marks,input_hashes=input_hashes,recursive_supported_by_parent=child_ok)
        log(f'earlier-layer capture windows={len(unique_windows)}; recursive support={child_ok}')

        # Validate the Jansen implementation on an additive function with known indices.
        qtest=qmc.Sobol(8,scramble=True,seed=50991).random_base2(7);qa,qb=qtest[:,:4],qtest[:,4:]
        fn=lambda z:z[:,0]+2*z[:,1]+3*z[:,2]+4*z[:,3]
        ya=fn(qa);yb=fn(qb);yhy=[]
        for k in range(4):
            z=qa.copy();z[:,k]=qb[:,k];yhy.append(fn(z))
        syn,var_syn=jansen_total(ya,yb,np.stack(yhy));syn_known=np.array([1,4,9,16])/30
        syn_err=float(np.max(np.abs(syn-syn_known)));assert syn_err<=.10

        # Four independent Uniform(0,1) multipliers scale fixed rank-4 concept fields.
        xv=(torch.as_tensor(val['observed_boxes'][selected],dtype=torch.float32)-mean)/std
        mean_np=np.asarray(norm['mean'],dtype=np.float32);std_np=np.asarray(norm['std'],dtype=np.float32)
        model=make_model();Usel=Uva[selected];contrib=(Usel[:,:,:,None]*W[None,None,:,:]).astype(np.float32)
        def scalar_means(raw):
            return {k:float(np.mean(v)) for k,v in scalar_outputs(raw,xv.cpu().numpy(),mean_np,std_np).items()}
        with torch.no_grad():y_original=model(xv).cpu().numpy()*std_np+mean_np
        def fixed_forward(act):
            h=model.encoder_layers[2].feedforward[1].register_forward_hook(lambda _m,_a,o:torch.as_tensor(act,device=o.device,dtype=o.dtype))
            try:
                with torch.no_grad():return model(xv).cpu().numpy()*std_np+mean_np
            finally:h.remove()
        h_id=model.encoder_layers[2].feedforward[1].register_forward_hook(lambda _m,_a,o:o)
        with torch.no_grad():y_identity=model(xv).cpu().numpy()*std_np+mean_np
        h_id.remove();y_reconstruction=fixed_forward(contrib.sum(axis=2));y_zero=fixed_forward(np.zeros((len(selected),15,128),np.float32))
        assert float(np.max(np.abs(y_identity-y_original)))<1e-6
        base={'original':scalar_means(y_original),'identity':scalar_means(y_identity),'full_reconstruction':scalar_means(y_reconstruction),'zero_activation':scalar_means(y_zero)}
        def eval_lambdas(lambdas,chunk=16):
            vals=[]
            for pos in range(0,len(lambdas),chunk):
                ls=np.asarray(lambdas[pos:pos+chunk],np.float32);ns=len(ls)
                act=np.einsum('sk,ntkf->sntf',ls,contrib,optimize=True).reshape(ns*len(selected),15,128)
                xb=xv.repeat(ns,1,1)
                h=model.encoder_layers[2].feedforward[1].register_forward_hook(lambda _m,_a,o:torch.as_tensor(act,device=o.device,dtype=o.dtype))
                try:
                    with torch.no_grad():pred=model(xb).cpu().numpy()*std_np+mean_np
                finally:h.remove()
                pred=pred.reshape(ns,len(selected),45,4)
                for p in pred:vals.append(scalar_means(p))
                check_budget()
            return np.asarray([[r[o] for o in OUTPUTS] for r in vals])
        qdesign=qmc.Sobol(8,scramble=True,seed=SOBOL_SEED).random_base2(7);sobA,sobB=qdesign[:,:4],qdesign[:,4:]
        YA=eval_lambdas(sobA);YB=eval_lambdas(sobB);YAB=np.empty((4,128,4),float)
        for k in range(4):
            mix=sobA.copy();mix[:,k]=sobB[:,k];YAB[k]=eval_lambdas(mix)
        conv=[];importance=[];ests={}
        for oi,oname in enumerate(OUTPUTS):
            prev=None
            for n in (32,64,128):
                st,var=jansen_total(YA[:n,oi],YB[:n,oi],YAB[:,:n,oi]);ests[(oname,n)]=st
                delta=float(np.max(np.abs(st-prev))) if prev is not None and np.isfinite(st).all() and np.isfinite(prev).all() else None
                ok=bool(var>1e-12 and np.isfinite(st).all() and delta is not None and delta<=.10) if n==128 else None
                for k in range(4):conv.append({'output':oname,'units':UNITS[oname],'base_samples':n,'concept':f'C{k+1}','total_order_index':float(st[k]) if np.isfinite(st[k]) else None,'output_variance':var,'change_from_previous_budget':delta,'converged_at_128':ok,'variance_negligible':bool(var<=1e-12)})
                if n==128:
                    for k in range(4):importance.append({'output':oname,'units':UNITS[oname],'concept':f'C{k+1}','base_samples':n,'total_order_index':float(st[k]) if np.isfinite(st[k]) else None,'output_variance':var,'converged':ok,'interpretation':'relative sensitivity within reconstructed FFN activation branch'})
                prev=st
        atomic_csv(OUT/'SOBOL_CONVERGENCE.csv',conv);atomic_csv(OUT/'SOBOL_IMPORTANCE.csv',importance)
        log(f'synthetic Sobol check max_abs_error={syn_err:.6f}; N=128 convergence='+str({o:next(r['converged'] for r in importance if r['output']==o) for o in OUTPUTS}))
        mark('sobol_complete',**marks,input_hashes=input_hashes,synthetic_sobol_max_abs_error=syn_err)

        # Baseline and per-concept intervention effects; retain original/reconstruction discrepancy.
        den={o:abs(base['full_reconstruction'][o]-base['original'][o]) for o in OUTPUTS};absrows=[]
        lambdas=[('zero_activation',np.zeros(4)),('full_reconstruction',np.ones(4))]
        lambdas += [(f'C{k+1}_only',np.eye(4)[k]) for k in range(4)]
        lambdas += [(f'C{k+1}_deleted',np.ones(4)-np.eye(4)[k]) for k in range(4)]
        vals=eval_lambdas(np.stack([x[1] for x in lambdas]))
        for j,(name,_) in enumerate(lambdas):
            for oi,o in enumerate(OUTPUTS):
                value=float(vals[j,oi]);absrows.append({'intervention':name,'output':o,'units':UNITS[o],'scalar_value':value,'original_value':base['original'][o],'identity_value':base['identity'][o],'full_reconstruction_value':base['full_reconstruction'][o],'zero_activation_value':base['zero_activation'][o],'absolute_change_from_original':abs(value-base['original'][o]),'absolute_change_from_full':abs(value-base['full_reconstruction'][o]),'full_reconstruction_discrepancy':den[o]})
        for o in OUTPUTS:absrows.append({'intervention':'identity_hook','output':o,'units':UNITS[o],'scalar_value':base['identity'][o],'original_value':base['original'][o],'identity_value':base['identity'][o],'full_reconstruction_value':base['full_reconstruction'][o],'zero_activation_value':base['zero_activation'][o],'absolute_change_from_original':0.0,'absolute_change_from_full':den[o],'full_reconstruction_discrepancy':den[o]})
        atomic_csv(OUT/'ABSOLUTE_INTERVENTION_EFFECTS.csv',absrows)

        # Sobol/reverse/random insertion and deletion. Random orders use five fixed seeds.
        fidelity=[];aucs={}
        random_orders={s:np.random.default_rng(s).permutation(4) for s in (3101,3102,3103,3104,3105)}
        for oi,o in enumerate(OUTPUTS):
            st=ests[(o,128)]
            if not np.isfinite(st).all() or not next(r['converged'] for r in importance if r['output']==o):
                fidelity.append({'output':o,'units':UNITS[o],'order_method':'skipped','mode':'all','random_seed':'','step':'','active_concepts':'','reason':'Sobol variance or convergence criterion failed; no importance ranking used'})
                continue
            rankorder=np.argsort(-st,kind='stable')
            strategies=[('sobol','',rankorder),('reverse','',rankorder[::-1])]+[(f'random_{s}',s,x) for s,x in random_orders.items()]
            for label,seed,order in strategies:
                for mode in ('insertion','deletion'):
                    mask=np.zeros(4) if mode=='insertion' else np.ones(4);masks=[mask.copy()]
                    for c in order:
                        mask=mask.copy();mask[c]=1.0 if mode=='insertion' else 0.0;masks.append(mask)
                    fvals=eval_lambdas(np.stack(masks))[:,oi];curve=[]
                    for step,value in enumerate(fvals):
                        d_full=abs(float(value)-base['full_reconstruction'][o]);d_orig=abs(float(value)-base['original'][o]);denom=den[o]
                        nfull=d_full/denom if denom>1e-10 else None;norig=d_orig/denom if denom>1e-10 else None
                        row={'output':o,'units':UNITS[o],'order_method':label,'mode':mode,'random_seed':seed,'step':step,'active_concepts':';'.join(f'C{k+1}' for k,v in enumerate(masks[step]) if v>.5),'scalar_value':float(value),'distance_from_original':d_orig,'distance_from_full_reconstruction':d_full,'normalized_distance_from_original':norig,'normalized_distance_from_full':nfull,'normalization_denominator':denom}
                        fidelity.append(row);curve.append(nfull)
                    auc=float(np.sum((np.asarray(curve[:-1])+np.asarray(curve[1:]))*.125)) if den[o]>1e-10 else None;aucs[(o,label,mode)]=auc
                    fidelity.append({'output':o,'units':UNITS[o],'order_method':label,'mode':mode,'random_seed':seed,'step':'AUC','curve_auc_normalized_distance_from_full':auc,'normalization_denominator':den[o]})
            for mode in ('insertion','deletion'):
                rvals=[aucs[(o,f'random_{s}',mode)] for s in random_orders];fidelity.append({'output':o,'units':UNITS[o],'order_method':'random_mean_sd','mode':mode,'random_seed':';'.join(map(str,random_orders)),'step':'AUC_summary','curve_auc_normalized_distance_from_full_mean':float(np.mean(rvals)) if all(v is not None for v in rvals) else None,'curve_auc_normalized_distance_from_full_sd':float(np.std(rvals,ddof=1)) if all(v is not None for v in rvals) else None,'normalization_denominator':den[o]})
        atomic_csv(OUT/'FIDELITY_RESULTS.csv',fidelity)
        # Report curve-ordering evidence against reverse/random controls.
        fidelity_gate={}
        for o in OUTPUTS:
            for mode in ('insertion','deletion'):
                sa=aucs.get((o,'sobol',mode));ra=aucs.get((o,'reverse',mode));rmean=aucs.get((o,'random_mean_sd',mode))
                # random_mean_sd AUC rows are summarized directly from the five seeds.
                rv=[aucs.get((o,f'random_{s}',mode)) for s in RANDOM_ORDERS]
                rm=float(np.mean(rv)) if all(x is not None for x in rv) else None
                fidelity_gate[f'{o}_{mode}']=None if any(x is None for x in (sa,ra,rm)) else (sa<=ra-.10 and sa<=rm-.10 if mode=='insertion' else sa>=ra+.10 and sa>=rm+.10)

        # Numeric figures rendered with Pillow to avoid an uninstalled plotting dependency.
        figures=[];colors=['#c0392b','#2471a3','#239b56','#8e44ad','#d68910','#566573']
        c=Canvas('Parent concept activation profiles: balanced GP7 sample',900,500);x0,y0,x1,y1=70,410,850,90
        c.line(x0,y0,x1,y0);c.line(x0,y0,x0,y1);pvals=Uva[selected].mean(axis=0);ym=max(float(pvals.max()),1e-9)
        for k in range(4):
            pts=[(x0+j*(x1-x0)/14,y0-float(pvals[j,k])/ym*(y0-y1)) for j in range(15)];c.poly(pts,colors[k],2);c.text(70+k*90,455,f'C{k+1}',colors[k])
        c.text(420,475,'Observed step 1 → 15; mean NNLS coefficient');c.save('PARENT_CONCEPT_PROFILES',figures,'Coefficient profiles, not input attribution maps')

        examples=[];c=Canvas('Observed training paths at parent coefficient maxima',1000,760)
        for k in range(4):
            si=int(np.argmax(Utr[:,:,k].mean(axis=1)));xy=centers[si];examples.append({'concept':f'C{k+1}','train_row':si,'sample_id':str(trids[si]),'video_id':str(train['video_id'][si]),'track_id':str(train['track_id'][si]),'mean_coefficient':float(Utr[si,:,k].mean())})
            col=k%2;row=k//2;lx=50+col*490;ty=80+row*330;ww=410;hh=245;xmin,xmax=float(xy[:,0].min()),float(xy[:,0].max());ymin,ymax=float(xy[:,1].min()),float(xy[:,1].max());dx=max(xmax-xmin,1);dy=max(ymax-ymin,1)
            c.text(lx,ty,f'C{k+1} {trids[si]}');pts=[(lx+(xx-xmin)/dx*ww,ty+25+(yy-ymin)/dy*hh) for xx,yy in xy];c.poly(pts,colors[k],2)
            for px,py in pts:c.circle(px,py,3,colors[k])
        atomic_csv(OUT/'REPRESENTATIVE_TRAIN_TRAJECTORIES.csv',examples);c.save('REPRESENTATIVE_OBSERVED_TRAJECTORIES',figures,'Observed paths from highest mean parent-coefficient training windows')

        if any(child_ok):
            c=Canvas('Rank-2 recursive factors (numeric IDs only)',850,430)
            for k,ok in enumerate(child_ok):
                y=100+k*78;c.circle(180,y,10,colors[k]);c.text(145,y-23,f'C{k+1}',colors[k]);c.line(190,y,610,y,'#888888',2)
                if ok:
                    for j in range(2):yy=y+(j-.5)*26;c.circle(640,yy,8,colors[k]);c.text(660,yy-8,f'C{k+1}.{j+1}',colors[k])
                else:c.text(630,y-8,'unsupported')
            c.save('RECURSIVE_CONCEPT_HIERARCHY',figures,'Numerically screened parent-child factors; numeric labels')

        descriptors=list(desc);c=Canvas('Training-only motion associations',1040,430);cellw=145;cellh=58;xbase=270;ybase=100
        for k in range(4):
            c.text(215,ybase+k*cellh+20,f'C{k+1}')
            for j,dname in enumerate(descriptors):
                rho=next(r['spearman_rho'] for r in assoc if r['concept']==f'C{k+1}' and r['descriptor']==dname);intensity=min(255,int(80+175*abs(rho)));fill=f'#{intensity:02x}5050' if rho>=0 else f'#5050{intensity:02x}';x=xbase+j*cellw;y=ybase+k*cellh;c.rect(x,y,x+cellw-4,y+cellh-4,fill);c.text(x+cellw/2,y+20,f'{rho:.2f}','#ffffff',anchor='middle',size=12)
        for j,dname in enumerate(descriptors):c.text(xbase+j*cellw+cellw/2,75,dname[:18],anchor='middle')
        c.save('MOTION_ASSOCIATION_HEATMAP',figures,'Spearman heatmap for numeric concepts and predefined descriptors')

        c=Canvas('Sobol total-order sensitivity within reconstructed FFN branch',1000,700)
        for oi,o in enumerate(OUTPUTS):
            col=oi%2;row=oi//2;lx=65+col*490;ty=80+row*300;vals=[next((r['total_order_index'] for r in importance if r['output']==o and r['concept']==f'C{k+1}'),0) or 0 for k in range(4)];mx=max([abs(float(x)) for x in vals]+[1e-9]);c.text(lx,ty,o);c.line(lx,ty+230,lx+420,ty+230)
            for k,v in enumerate(vals):x=lx+45+k*90;h=abs(float(v))/mx*180;c.rect(x,ty+230-h,x+45,ty+230,colors[k]);c.text(x+20,ty+238,f'C{k+1}',anchor='middle');c.text(x+20,ty+230-h-15,f'{float(v):.2f}',anchor='middle')
        c.save('SOBOL_IMPORTANCE',figures,'N=128 Jansen total-order indices for four scalar outputs')

        names=['full_reconstruction','zero_activation']+[f'C{k+1}_only' for k in range(4)];c=Canvas('Absolute intervention effects vs original model',1000,700)
        for oi,o in enumerate(OUTPUTS):
            col=oi%2;row=oi//2;lx=60+col*500;ty=80+row*300;vals=[next(r['absolute_change_from_original'] for r in absrows if r['intervention']==n and r['output']==o) for n in names];mx=max(max(vals),1e-12);c.text(lx,ty,o);c.line(lx,ty+230,lx+440,ty+230)
            for k,v in enumerate(vals):x=lx+12+k*70;h=float(v)/mx*175;c.rect(x,ty+230-h,x+42,ty+230,colors[k%len(colors)]);c.text(x+21,ty+238,names[k].replace('_',' ')[:10],anchor='middle',size=10)
        c.save('ABSOLUTE_INTERVENTION_EFFECTS',figures,'Absolute scalar effects from original model')

        c=Canvas('Insertion/deletion normalized curves',1000,700)
        for oi,o in enumerate(OUTPUTS):
            col=oi%2;row=oi//2;lx=70+col*480;ty=75+row*300;c.text(lx,ty,o);c.line(lx,ty+230,lx+410,ty+230);c.line(lx,ty+230,lx,ty+35)
            colors2=[('#c0392b','sobol insertion'),('#2471a3','sobol deletion'),('#d68910','reverse insertion'),('#566573','reverse deletion')]
            for label,mode,colr in [('sobol','insertion','#c0392b'),('sobol','deletion','#2471a3'),('reverse','insertion','#d68910'),('reverse','deletion','#566573')]:
                rr=[r for r in fidelity if r['output']==o and r['order_method']==label and r['mode']==mode and isinstance(r['step'],int)]
                if rr and den[o]>1e-10:
                    pts=[(lx+r['step']*100,ty+220-min(float(r['normalized_distance_from_full']),2)*85) for r in rr];c.poly(pts,colr,2);c.text(lx+10+len(colors2)*82,ty+20,f'{label[:3]} {mode[:3]}',colr,size=10);colors2.pop()
            if den[o]>1e-10:
                for mode,colr in [('insertion','#239b56'),('deletion','#8e44ad')]:
                    curves=[]
                    for seed in RANDOM_ORDERS:
                        rr=[r for r in fidelity if r['output']==o and r['order_method']==f'random_{seed}' and r['mode']==mode and isinstance(r['step'],int)];
                        if rr:curves.append([r['normalized_distance_from_full'] for r in rr])
                    if curves:c.poly([(lx+j*100,ty+220-min(float(np.mean(curves,axis=0)[j]),2)*85) for j in range(5)],colr,2)
        c.save('INSERTION_DELETION_FIDELITY',figures,'Sobol and reverse curves plus random-order mean trajectories')

        # Deterministic GP7 trajectory overlay: observed, original, full, zero, and each concept alone.
        ii=int(selected[0]);local=0;xone=xv[0:1];obs=val['observed_boxes'][ii].astype(float);obsc=(obs[:,:2]+obs[:,2:])/2
        def one_path(act=None):
            h=model.encoder_layers[2].feedforward[1].register_forward_hook(lambda _m,_a,o:torch.as_tensor(act,device=o.device,dtype=o.dtype)) if act is not None else None
            try:
                with torch.no_grad():p=model(xone).cpu().numpy()*std_np+mean_np
            finally:
                if h is not None:h.remove()
            return (p[0,:,:2]+p[0,:,2:])/2
        paths=[('original',one_path()),('full',one_path(contrib[local].sum(axis=1)[None,:,:])),('zero',one_path(np.zeros((1,15,128),np.float32)))]
        for k in range(4):paths.append((f'C{k+1} only',one_path(contrib[local,:,k,:][None,:,:])))
        c=Canvas(f'GP7 trajectory interventions: {vaids[ii]}',900,650);allpaths=[obsc]+[p for _,p in paths];xy=np.concatenate(allpaths);xmin,xmax=xy[:,0].min(),xy[:,0].max();ymin,ymax=xy[:,1].min(),xy[:,1].max();dx=max(xmax-xmin,1);dy=max(ymax-ymin,1)
        def points(p):return [(80+(x-xmin)/dx*680,80+(y-ymin)/dy*480) for x,y in p]
        c.poly(points(obsc),'#111111',3)
        for j,(label,p) in enumerate(paths):c.poly(points(p),colors[(len(label)+len(p))%len(colors)],2);c.text(770,85+j*22,label)
        c.save('CONCEPT_INTERVENTION_TRAJECTORY_OVERLAY',figures,'Observed and predicted centers for one deterministic GP7 window')
        atomic_csv(OUT/'FIGURE_MANIFEST.csv',figures)

        # Recursive report; no child semantics inferred from appearance.
        rlines=['# Recursive concept results','',f'For each parent, exactly {len(top[0])} of {988*15} training tokens (top decile by saved coefficient) were selected. Earlier activations were captured for {len(unique_windows)} unique windows at encoder layer 1 post-ReLU FFN. Recursive NMF rank was fixed at 2; seeds were {SEEDS}.','', '| Parent | Seed | Fit relative error | Coefficient zero fraction | Iterations | Warning |','|---|---:|---:|---:|---:|---|']
        for r in recrows:
            if r['seed']!='stability':rlines.append(f"| {r['parent_concept']} | {r['seed']} | {r['relative_fit_error']:.6f} | {r['zero_coefficient_fraction']:.4f} | {r['iterations']} | {r['convergence_warning']} |")
            else:rlines.append(f"| {r['parent_concept']} | stability | — | — | — | {r['warning_text']} |")
        rlines+=['',f'Numerical support flags (both fits converged without warning, error ≤0.50, matched cosine ≥0.90): `{child_ok}`. No semantic labels were assigned.']
        (OUT/'RECURSIVE_CONCEPT_RESULTS.md').write_text('\n'.join(rlines)+'\n',encoding='utf-8')

        # Determine project fidelity screens and write exact completion summary.
        sobol_ok={o:bool(next(r['converged'] for r in importance if r['output']==o)) for o in OUTPUTS}
        summary={'synthetic_sobol_max_abs_index_error':syn_err,'synthetic_expected_indices':syn_known.tolist(),
                 'sobol_converged_by_output':sobol_ok,'fidelity_ordering_screen':fidelity_gate,
                 'full_reconstruction_absolute_scalar_discrepancy':den,'identity_max_abs_error':float(np.max(np.abs(y_identity-y_original))),
                 'recursive_supported_by_parent':child_ok,'training_window_count':988,'gp7_sample_window_count':len(selected),
                 'runtime_seconds':time.perf_counter()-START,'cpu_threads_max':4,'official_test_accessed':False,
                 'model_training':False,'activation_extraction_repeated':False}
        atomic_json(OUT/'STAGE3_SUMMARY.json',summary)
        lines=['# CRAFT final experiment report — Stage 3','',f"**Status:** bounded exploratory development analysis completed in {summary['runtime_seconds']:.2f} seconds (four CPU threads maximum). The frozen AO-3 seed-20260928 model and canonical Stage 2B rank-4 bank were used. No model weights were changed, no final-layer activation extraction was repeated, and official-test data was not accessed.",'', 'This is a trajectory-regression adaptation inspired by the supervisor-provided CRAFT paper, not a faithful reproduction of its image-crop pipeline, implicit differentiation, attribution maps, or human utility experiments.','', '## Recursive decomposition','','See `RECURSIVE_CONCEPT_RESULTS.md` and `PARENT_TOP_TOKEN_MANIFEST.csv`. Each parent selected the top 10% of train token coefficients and used aligned preceding-layer ReLU FFN outputs. Child factors remained numeric; no labels were assigned from visualization.','', '## Motion association','','Six observation-derived descriptors were tested against four temporal-mean parent coefficients. Benjamini-Hochberg correction covered all 24 tests; association flags required |Spearman rho|≥0.30 and q<0.05. See `MOTION_ASSOCIATIONS.csv`. The 988 windows include repeated/overlapping samples from '+str(unique_tracks)+' video-track groups; window-level p-values are descriptive and do not account for all within-track dependence.']
        flagged=[r for r in assoc if r['association_flag']]
        lines.append('Associations meeting the fixed threshold: '+('; '.join(f"{r['concept']} vs {r['descriptor']} (rho {r['spearman_rho']:.3f}, q {r['bh_q_value']:.3g}, n={r['n_windows']})" for r in flagged) if flagged else 'none.'))
        lines+=['','## Sobol design and results','','The 32-window GP7 sample comprises 16 windows from each validation video, selected by sorted video and sample IDs without consulting predictions, errors, or concept importance. Four independent Uniform(0,1) amplitude multipliers scale the frozen parent contribution fields. Jansen total-order indices were estimated at nested base budgets 32, 64, and 128 using two independent scrambled Sobol sequences. This is relative sensitivity within the reconstructed FFN branch under that intervention distribution; it is not natural joint concept variation or complete-model global importance.','',f"The synthetic additive sensitivity check had maximum absolute index error {syn_err:.5f} (expected indices `[1,4,9,16]/30`). Sobol indices and output variances are in `SOBOL_IMPORTANCE.csv`; budget traces are in `SOBOL_CONVERGENCE.csv`.",'', '| Scalar output | Units | Output variance (N=128) | Converged by fixed rule? |','|---|---|---:|---|']
        for o in OUTPUTS:
            rr=next(r for r in conv if r['output']==o and r['base_samples']==128 and r['concept']=='C1');lines.append(f"| {o} | {UNITS[o]} | {rr['output_variance']:.6g} | {sobol_ok[o]} |")
        lines+=['','## Absolute effects and fidelity','','Original, identity, full reconstruction, zero activation, concept-only, and concept-deleted scalar outputs are in `ABSOLUTE_INTERVENTION_EFFECTS.csv`. Full-reconstruction deviations from original are recorded separately from concept interventions. Insertion/deletion curves compare Sobol order, reverse order, and five fixed random orders; see `FIDELITY_RESULTS.csv`. Normalized distances are undefined when the full-versus-original denominator is ≤1e-10. A fidelity-order screen passes only under the predeclared AUC margins against both controls; see `STAGE3_SUMMARY.json`.','', '| Output | Full-vs-original absolute scalar difference | Insertion screen | Deletion screen |','|---|---:|---|---|']
        for o in OUTPUTS:lines.append(f"| {o} | {den[o]:.6g} | {fidelity_gate.get(o+'_insertion')} | {fidelity_gate.get(o+'_deletion')} |")
        lines+=['','## Figures and limitations','','`FIGURE_MANIFEST.csv` lists PNG, editable SVG, and PDF exports generated from completed numeric results. Parent profiles are activation coefficients, not input attribution maps. Trajectory overlays are development examples only. No human evaluation, implicit differentiation, semantic validation, or additional rank search was performed. Sobol indices depend on the declared independent uniform scaling of fixed concept profiles and only 32 GP7 windows; high relative indices with small absolute output variance/effects are not evidence of strong model-level importance. Recursive factors, motion correlations, and fidelity controls should be interpreted within these limits.','', 'No follow-on phase was started.']
        (OUT/'CRAFT_FINAL_EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

        check_budget();mark('completed',**marks,input_hashes=input_hashes,runtime_seconds=time.perf_counter()-START,sobol_converged=sobol_ok,recursive_supported_by_parent=child_ok)
        hashrows=[]
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p.name not in ('ARTIFACT_HASHES.csv','run.lock'):hashrows.append({'artifact':f'results_paper/phase_6d_craft_stage3_v1/{p.name}','sha256':sha(p)})
        for rel,h in input_hashes.items():hashrows.append({'artifact':'input/'+rel,'sha256':h})
        atomic_csv(OUT/'ARTIFACT_HASHES.csv',hashrows)
        work=ROOT/'results_paper'/'WORKLOG.md'
        entry=f"\n## 2026-11-27 — CRAFT Stage 3 bounded development analysis\n\n- Completed the bounded frozen AO-3 Stage 3 experiment in {time.perf_counter()-START:.2f}s under the 30-minute budget and four-thread limit. Reused saved Stage 1 activations and the canonical Stage 2B rank-4 bank; no training, final-layer activation recapture, or official-test access.\n- Saved protocol, regression results, recursive factors, motion associations, Sobol estimates/convergence, intervention/fidelity tables, figures, report, and hashes in `results_paper/phase_6d_craft_stage3_v1/`.\n- See `CRAFT_FINAL_EXPERIMENT_REPORT.md` for actual findings and limitations. No manuscript edits, commit, or push.\n"
        with work.open('a',encoding='utf-8',newline='\n') as f:f.write(entry);f.flush();os.fsync(f.fileno())
        log(f'completed in {time.perf_counter()-START:.2f}s; Sobol={sobol_ok}; fidelity={fidelity_gate}; recursion={child_ok}')
    except Exception as ex:
        atomic_json(OUT/'status.json',{'state':'failed_or_incomplete','elapsed_seconds':time.perf_counter()-START,'budget_seconds':BUDGET,'error':repr(ex),'traceback':traceback.format_exc()});log('FAILED: '+repr(ex));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
