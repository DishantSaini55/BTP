"""One-shot, bounded Stage 2B diagnostic. Uses saved Stage 1 activations only."""
import csv, hashlib, json, os, time, warnings
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, nnls
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from MTN.run_ao_v1 import _read_split, _train_only_normalization
from MTN.ao_models_v1 import build_ao_model

ROOT=Path(__file__).resolve().parents[2]; BASE=ROOT/'results_paper/phase_6d_craft_stage1_v1'; CACHE=ROOT/'results_paper/splits/ao_annotation_only_v1'; OUT=ROOT/'results_paper/phase_6d_craft_stage2b_v1'; CK=ROOT/'results_paper/AO_confirmation_v1/AO-3/seed_20260928/best_model.pt'
RANKS=(4,8,16); SEEDS=(20260928,20260929); MAX_ITER=150; TOL=1e-3; QUALITY=.50; STABILITY=.90

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def relerr(x,r): return float(np.linalg.norm(x-r)/max(np.linalg.norm(x),1e-30))
def writecsv(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
def project(x,W):
 u=np.vstack([nnls(W.T,row)[0] for row in x]);return u

def main():
 lock=OUT/'run.lock'; fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'start':time.time()}).encode());os.close(fd); started=time.perf_counter()
 try:
  torch.set_num_threads(4);threadpool_limits(limits=4)
  inputs=[BASE/'AO3_FFN_activations.npz',BASE/'activation_audit.json',CACHE/'train.npz',CACHE/'validation.npz',CACHE/'split_manifest.json']
  input_hashes={str(p):sha(p) for p in inputs}
  for row in csv.DictReader((BASE/'artifact_hashes.csv').open()):
   if row['artifact'] in ('AO3_FFN_activations.npz','train.npz','validation.npz'):
    p=CACHE/row['artifact'] if row['artifact'] in ('train.npz','validation.npz') else BASE/row['artifact']
    assert sha(p).lower()==row['sha256'].lower(),f'hash mismatch: {p}'
  ar=np.load(BASE/'AO3_FFN_activations.npz'); train,val,manifest=_read_split(CACHE)
  assert manifest['official_test_sets_read']==[]
  Xtr=ar['train'].reshape(-1,128).astype(np.float64); Xva=ar['validation'].reshape(-1,128).astype(np.float64)
  assert Xtr.shape==(988*15,128) and Xva.shape==(432*15,128)
  assert np.isfinite(Xtr).all() and np.isfinite(Xva).all() and Xtr.min()>=0 and Xva.min()>=0
  vids=train['video_id'].astype(str); sorted_vids=sorted(set(vids)); hold_video=sorted_vids[-1]
  fit_w=np.flatnonzero(vids!=hold_video); hold_w=np.flatnonzero(vids==hold_video)
  fit_ix=(fit_w[:,None]*15+np.arange(15)).reshape(-1); hold_ix=(hold_w[:,None]*15+np.arange(15)).reshape(-1)
  Xfit=Xtr[fit_ix]; Xhold=Xtr[hold_ix]
  fit_rows=[]; stability_rows=[]; banks={}; hold_coefs={}
  def log(msg):
   print(msg,flush=True)
   with (OUT/'progress.log').open('a',encoding='utf-8') as f:f.write(msg+'\n');f.flush();os.fsync(f.fileno())
  log(f'begin; holdout_video={hold_video}; fit_windows={len(fit_w)}; holdout_windows={len(hold_w)}; ranks={RANKS}; seeds={SEEDS}; max_iter={MAX_ITER}')
  for rank in RANKS:
   components=[]; seed_hold={}
   for seed in SEEDS:
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always',ConvergenceWarning)
     model=NMF(n_components=rank,init='nndsvda',random_state=seed,max_iter=MAX_ITER,tol=TOL)
     Hfit=model.fit_transform(Xfit); W=model.components_
    train_err=relerr(Xfit,Hfit@W); Uhold=project(Xhold,W); hold_err=relerr(Xhold,Uhold@W)
    warned=any(issubclass(w.category,ConvergenceWarning) for w in caught)
    warning_text=' | '.join(str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning))
    fit_rows.append({'stage':'grouped_training_holdout','rank':rank,'seed':seed,'fit_relative_error':train_err,'holdout_relative_error':hold_err,'fit_zero_coef_fraction':float(np.mean(Hfit==0)),'holdout_zero_coef_fraction':float(np.mean(Uhold==0)),'iterations':int(model.n_iter_),'convergence_warning':warned,'warning_text':warning_text})
    components.append(W);seed_hold[seed]=Uhold
    log(f'candidate done rank={rank} seed={seed} train_fit={train_err:.6f} group_holdout={hold_err:.6f} iter={model.n_iter_} warning={warned}')
   normed=components[0]/np.maximum(np.linalg.norm(components[0],axis=1,keepdims=True),1e-30); normed2=components[1]/np.maximum(np.linalg.norm(components[1],axis=1,keepdims=True),1e-30)
   sim=normed@normed2.T; ii,jj=linear_sum_assignment(-sim); stab=float(sim[ii,jj].mean())
   stability_rows.append({'rank':rank,'matched_mean_cosine':stab,'matched_components':len(ii),'minimum_matched_cosine':float(sim[ii,jj].min())})
   banks[rank]=components
   hold_coefs[rank]=np.stack([seed_hold[s].reshape(len(hold_w),15,rank) for s in SEEDS])
  mean_hold={r:float(np.mean([q['holdout_relative_error'] for q in fit_rows if q['rank']==r])) for r in RANKS}
  stab_map={q['rank']:q['matched_mean_cosine'] for q in stability_rows}
  passing=[r for r in RANKS if mean_hold[r]<=QUALITY and stab_map[r]>=STABILITY]
  selected=min(passing) if passing else min(RANKS,key=lambda r:(mean_hold[r],r))
  gate_pass=bool(passing)
  # Refit the selected rank on all 988 training windows, retaining both fixed-seed banks.
  full_banks=[]; full_H=[]; full_val_u=[]; full_rows=[]
  for seed in SEEDS:
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',ConvergenceWarning)
    model=NMF(n_components=selected,init='nndsvda',random_state=seed,max_iter=MAX_ITER,tol=TOL)
    H=model.fit_transform(Xtr); W=model.components_
   Uva=project(Xva,W); e_train=relerr(Xtr,H@W);e_val=relerr(Xva,Uva@W)
   warned=any(issubclass(w.category,ConvergenceWarning) for w in caught);warning_text=' | '.join(str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning))
   full_rows.append({'rank':selected,'seed':seed,'train_relative_error':e_train,'validation_relative_error':e_val,'train_zero_coef_fraction':float(np.mean(H==0)),'validation_zero_coef_fraction':float(np.mean(Uva==0)),'iterations':int(model.n_iter_),'convergence_warning':warned,'warning_text':warning_text})
   full_banks.append(W);full_H.append(H);full_val_u.append(Uva)
   log(f'full refit done rank={selected} seed={seed} train={e_train:.6f} validation={e_val:.6f} iter={model.n_iter_} warning={warned}')
  n1=full_banks[0]/np.maximum(np.linalg.norm(full_banks[0],axis=1,keepdims=True),1e-30);n2=full_banks[1]/np.maximum(np.linalg.norm(full_banks[1],axis=1,keepdims=True),1e-30);sim=n1@n2.T;ii,jj=linear_sum_assignment(-sim);full_stab=float(sim[ii,jj].mean())
  writecsv(OUT/'GROUPED_HOLDOUT_RESULTS.csv',fit_rows);writecsv(OUT/'FACTOR_STABILITY_RESULTS.csv',stability_rows);writecsv(OUT/'FULL_TRAINING_REFIT_RESULTS.csv',full_rows)
  np.savez_compressed(OUT/'GROUPED_HOLDOUT_COEFFICIENTS.npz',holdout_window_indices=hold_w,rank4=hold_coefs[4],rank8=hold_coefs[8],rank16=hold_coefs[16])
  np.savez_compressed(OUT/'SELECTED_FULL_TRAINING_BANK.npz',rank=selected,seeds=np.array(SEEDS),components=np.stack(full_banks),train_coefficients=np.stack([h.reshape(988,15,selected) for h in full_H]),validation_coefficients=np.stack([u.reshape(432,15,selected) for u in full_val_u]))
  selection={'selected_rank':selected,'candidate_mean_grouped_holdout_error':mean_hold,'candidate_stability':stab_map,'quality_threshold':QUALITY,'stability_threshold':STABILITY,'screen_passed':gate_pass,'full_refit_stability':full_stab,'full_refit_stability_passed':full_stab>=STABILITY,'holdout_video_id':hold_video,'holdout_windows':len(hold_w),'fit_windows':len(fit_w),'validation_used_for_selection':False}
  # A model-output intervention is permitted only if group-holdout quality/stability and full-refit stability pass.
  run_intervention=gate_pass and full_stab>=STABILITY
  intervention={'performed':False,'reason':'quality/stability gate failed; no model-output intervention or Sobol analysis'}
  if run_intervention:
   from MTN.run_ao_v1 import _train_only_normalization
   stats=_train_only_normalization(train);mu=torch.tensor(stats['mean'],dtype=torch.float32);sd=torch.tensor(stats['std'],dtype=torch.float32)
   order=np.argsort(val['sample_id'].astype(str),kind='stable')[:32]
   x=(torch.tensor(val['observed_boxes'][order],dtype=torch.float32)-mu)/sd
   payload=torch.load(CK,map_location='cpu',weights_only=False);model=build_ao_model('AO-3').eval();model.load_state_dict(payload['model_state'],strict=True)
   W=full_banks[0]; U=project(ar['validation'][order].reshape(-1,128).astype(np.float64),W); rec=(U@W).reshape(32,15,128).astype(np.float32)
   mods={}
   with torch.no_grad(): mods['original']=model(x).cpu().numpy()
   def run_hook(mode):
    def hook(mod,args,out):
     if mode=='identity': return out
     if mode=='reconstructed': return torch.as_tensor(rec,device=out.device,dtype=out.dtype)
     return torch.zeros_like(out)
    h=model.encoder_layers[2].feedforward[1].register_forward_hook(hook)
    try:
     with torch.no_grad(): return model(x).cpu().numpy()
    finally:h.remove()
   mods['identity']=run_hook('identity');mods['reconstructed']=run_hook('reconstructed');mods['zero']=run_hook('zero')
   base=mods['original']; intervention={'performed':True,'windows':32,'sample_ids':val['sample_id'][order].astype(str).tolist(),'outputs':{k:{'relative_l2_deviation_from_original':relerr(base,v),'coordinate_mae_from_original':float(np.mean(np.abs(base-v)))} for k,v in mods.items()},'identity_max_abs_error':float(np.max(np.abs(base-mods['identity']))),'adequacy_threshold_relative_l2':0.05,'reconstruction_adequate':relerr(base,mods['reconstructed'])<=0.05}
  selection['conditional_intervention_performed']=run_intervention
  selection['conditional_intervention'] = intervention
  selection['full_refit_stability_passed']=full_stab>=STABILITY
  (OUT/'SELECTION_AND_GATE.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
  (OUT/'INTERVENTION_RESULTS.json').write_text(json.dumps(intervention,indent=2),encoding='utf-8')
  status={'status':'completed','runtime_seconds':time.perf_counter()-started,'max_cpu_threads':4,'official_test_accessed':False,'training_performed':False,'activation_extraction_repeated':False,'candidate_ranks':RANKS,'seeds':SEEDS,'max_iter':MAX_ITER,'selected_rank':selected,'screen_passed':gate_pass,'intervention_performed':run_intervention,'sobol_attempted':False,'input_hashes':input_hashes}
  (OUT/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
  log(f'complete elapsed={status["runtime_seconds"]:.3f}s selected_rank={selected} screen_pass={gate_pass} intervention={run_intervention}')
 finally: lock.unlink(missing_ok=True)
if __name__=='__main__':main()
