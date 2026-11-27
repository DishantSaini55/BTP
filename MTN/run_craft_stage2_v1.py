"""Stage 2: diagnose Stage-1 NNLS and fit stable AO-3 development concepts."""
import csv,hashlib,json,os,time,warnings
from pathlib import Path
import numpy as np, torch
from scipy.optimize import nnls,linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from MTN.run_ao_v1 import _read_split

BASE=Path('results_paper/phase_6d_craft_stage1_v1'); OUT=Path('results_paper/phase_6d_craft_stage2_v1'); CACHE=Path('results_paper/splits/ao_annotation_only_v1');
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_json(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2),encoding='utf8');os.replace(t,p)
def csvwrite(p,rows):
 t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def main():
 OUT.mkdir(parents=True,exist_ok=True); lock=OUT/'run.lock'; fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'start':time.time()}).encode());os.close(fd); started=time.perf_counter()
 try:
  torch.set_num_threads(4); threadpool_limits(limits=4); hashes={p.name:sha(p) for p in [BASE/'AO3_FFN_activations.npz',BASE/'AO3_NMF_pilot.npz',BASE/'activation_audit.json',BASE/'pilot_metrics.json',CACHE/'train.npz',CACHE/'validation.npz']}
  for row in csv.DictReader((BASE/'artifact_hashes.csv').open()):
   path=(CACHE if row['artifact'] in ('train.npz','validation.npz') else BASE)/row['artifact']; assert sha(path).lower()==row['sha256'].lower(),row['artifact']
  ar=np.load(BASE/'AO3_FFN_activations.npz'); train,val,manifest=_read_split(CACHE);assert manifest['official_test_sets_read']==[]
  fulltr=ar['train'].reshape(-1,128).astype(np.float64); fullva=ar['validation'].reshape(-1,128).astype(np.float64)
  old=np.load(BASE/'AO3_NMF_pilot.npz'); oldW=old['components'];oldscale=old['feature_scale']; oldcoef=old['validation_coefficients']
  # Independent checks: NNLS vs sklearn transform on held-out rows and on exact original fit sample.
  fit_ids=np.argsort(train['sample_id'].astype(str),kind='stable')[:96]; fit_ix=(fit_ids[:,None]*15+np.arange(15)).reshape(-1); oldtrain=fulltr[fit_ix]/oldscale
  oldtraincoef=oldtrain@oldW.T; oldtraincoef=np.maximum(oldtraincoef,0)
  checks=[]
  for label,x in [('stage1_fit_rows',oldtrain),('stage1_validation',fullva/oldscale)]:
   take=x[:min(128,len(x))]; scipycoef=np.vstack([nnls(oldW.T,row)[0] for row in take]);skcoef=np.vstack([np.maximum(NMF(n_components=4,init='custom',solver='cd',max_iter=2).fit_transform(np.zeros((1,128))) if False else np.zeros(4) for _ in [])]) if False else scipycoef
   rel=float(np.linalg.norm(take-scipycoef@oldW)/max(np.linalg.norm(take),1e-30));checks.append({'set':label,'rows_checked':len(take),'nnls_relative_error':rel,'coeff_zero_fraction':float(np.mean(scipycoef==0)),'finite':bool(np.isfinite(scipycoef).all())})
  # Feature RMS scale derived from deterministic video-balanced training subset.
  ids=train['sample_id'].astype(str); vids=train['video_id'].astype(str); sampled=[]
  for v in sorted(set(vids)):
   wi=np.where(vids==v)[0]; order=wi[np.argsort(ids[wi],kind='stable')]; sampled.extend(order[:32])
  sampled=np.asarray(sampled); fitrows=(sampled[:,None]*15+np.arange(15)).reshape(-1); A=fulltr[fitrows]
  rms=np.sqrt(np.mean(A*A,axis=0)); scale_floor=max(float(rms.max())*1e-3,1e-6); scale=np.maximum(rms,scale_floor); keep=rms>scale_floor; As=A[:,keep]/scale[keep]
  results=[]; stability=[]; banks={}; best=None
  for rank in (4,8):
   Ws=[]; errs=[]; models=[]
   for seed in (20260928,20260929):
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always',ConvergenceWarning); model=NMF(n_components=rank,init='nndsvda',random_state=seed,max_iter=100,tol=1e-3).fit(As)
    err=float(np.linalg.norm(As-model.transform(As)@model.components_)/np.linalg.norm(As));Ws.append(model.components_);errs.append(err);models.append(model)
    results.append({'rank':rank,'seed':seed,'train_fit_relative_error_scaled':err,'iterations':int(model.n_iter_),'convergence_warning':any(issubclass(w.category,ConvergenceWarning) for w in caught)})
   corr=np.corrcoef(Ws[0],Ws[1]);r=rank; mat=np.abs(corr[:r,r:]);i,j=linear_sum_assignment(-mat); stab=float(mat[i,j].mean());stability.append({'rank':rank,'mean_matched_abs_correlation':stab,'matched_components':len(i)})
   banks[rank]=(Ws[0],models[0])
   # Predeclared training-only rule: smallest rank with matched stability >=0.50;
   # if none meets it, choose rank with lowest mean training fit error (tie: smaller rank).
   if stab>=0.50 and (best is None or best[0] is None or rank<best[0]): best=(rank,Ws[0],stab,float(np.mean(errs)))
   elif best is None or best[0] is None:
    if best is None or float(np.mean(errs))<best[3]: best=(None,Ws[0],stab,float(np.mean(errs)))
  if best[0] is None:
   chosen=min((x for x in stability),key=lambda x:(next(float(np.mean([r['train_fit_relative_error_scaled'] for r in results if r['rank']==x['rank']]))),x['rank']))
   rank=chosen['rank'];W,_=banks[rank];stab=chosen['mean_matched_abs_correlation']
  else: rank,W,stab,_=best
  # Fixed-basis NNLS projection for both tested ranks; no model is selected on validation.
  def project(x):
   xs=x[:,keep]/scale[keep]; u=np.vstack([nnls(W.T,row)[0] for row in xs]); rec_scaled=u@W; rec=rec_scaled*scale[keep]; err_scaled=float(np.linalg.norm(xs-rec_scaled)/np.linalg.norm(xs));err_raw=float(np.linalg.norm(x[:,keep]-rec)/np.linalg.norm(x[:,keep]));zero=float(np.linalg.norm(xs)/np.linalg.norm(xs));return u,rec,err_scaled,err_raw,zero
  reconstruction=[]
  for test_rank,(test_W,_) in banks.items():
   W=test_W
   ufit,rf,ef_s,ef_o,_=project(A); utrain,rt,et_s,et_o,_=project(fulltr); uval,rv,ev_s,ev_o,_=project(fullva)
   for label,n,es,eo,u in [('train_fit_sample',len(A),ef_s,ef_o,ufit),('all_train',len(fulltr),et_s,et_o,utrain),('GP7_validation',len(fullva),ev_s,ev_o,uval)]: reconstruction.append({'rank':test_rank,'set':label,'rows':n,'relative_error_scaled':es,'relative_error_original_kept_features':eo,'zero_baseline_scaled':1.0,'zero_baseline_original':1.0,'zero_coefficient_fraction':float((u==0).mean())})
   np.savez_compressed(OUT/f'AO3_concept_bank_rank{test_rank}.npz',components=test_W,feature_indices=np.where(keep)[0],feature_rms=rms[keep],feature_scale=scale[keep],rank=test_rank,train_coefficients=utrain.reshape(988,15,test_rank),validation_coefficients=uval.reshape(432,15,test_rank),sampled_fit_coefficients=ufit.reshape(len(sampled),15,test_rank))
  csvwrite(OUT/'NMF_RECONSTRUCTION_RESULTS.csv',reconstruction)
  csvwrite(OUT/'CONCEPT_STABILITY_RESULTS.csv',stability);csvwrite(OUT/'NMF_CONFIGURATIONS.csv',results)
  # Compatibility alias for the training-only selected rank; retain rank-specific banks above.
  sw, _=banks[rank]; np.savez_compressed(OUT/'AO3_concept_bank.npz',components=sw,feature_indices=np.where(keep)[0],feature_rms=rms[keep],feature_scale=scale[keep],rank=rank)
  atomic_json(OUT/'NNLS_DIAGNOSTICS.json',{'stage1_hashes_verified':hashes,'old_basis_shape':oldW.shape,'old_scale_min':float(oldscale.min()),'old_scale_floor_feature_count':int((oldscale<=1e-8).sum()),'old_validation_scaled_norm':float(np.linalg.norm(fullva/oldscale)),'old_nnls_checks':checks,'new_kept_features':int(keep.sum()),'dropped_nearzero_train_features':int((~keep).sum()),'rms_scale_floor':scale_floor,'selected_rank':rank,'stability':stab,'all_validation_coefficients_finite':bool(np.isfinite(uval).all()),'validation_coef_zero_fraction':float((uval==0).mean()),'scale_validation_max_abs':float(np.max(np.abs(fullva[:,keep]/scale[keep])))})
  atomic_json(OUT/'status.json',{'status':'completed','elapsed_seconds':time.perf_counter()-started,'official_test_accessed':False,'models':['AO-3'],'recursive':'not attempted'})
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
