"""Validated AO-3 CRAFT activation and bounded NMF pilot (development roles only)."""
from __future__ import annotations
import os,json,time,hashlib,csv,warnings
from pathlib import Path
import numpy as np, torch
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import nnls
from MTN.ao_models_v1 import build_ao_model
from MTN.run_ao_v1 import _read_split,_train_only_normalization

OUT=Path('results_paper/phase_6d_craft_stage1_v1'); CACHE=Path('results_paper/splits/ao_annotation_only_v1'); CK=Path('results_paper/AO_confirmation_v1/AO-3/seed_20260928/best_model.pt')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_json(p,obj):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2),encoding='utf8');os.replace(t,p)
def atomic_csv(p,rows):
 t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def main():
 OUT.mkdir(parents=True,exist_ok=False); lock=OUT/'run.lock'; fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'started':time.time()}).encode());os.close(fd)
 started=time.perf_counter();torch.set_num_threads(4)
 try:
  atomic_json(OUT/'status.json',{'status':'incomplete','stage':'activation_capture'})
  tr,va,manifest=_read_split(CACHE); assert len(tr['sample_id'])==988 and len(va['sample_id'])==432
  assert manifest['official_test_sets_read']==[]
  stats=_train_only_normalization(tr);mu=torch.tensor(stats['mean']);sd=torch.tensor(stats['std'])
  xtr=(torch.tensor(tr['observed_boxes'])-mu)/sd;xva=(torch.tensor(va['observed_boxes'])-mu)/sd
  payload=torch.load(CK,map_location='cpu',weights_only=False);model=build_ao_model('AO-3').eval();model.load_state_dict(payload['model_state'],strict=True)
  cap={};handles=[]
  for i,l in enumerate(model.encoder_layers):handles.append(l.feedforward[1].register_forward_hook(lambda m,a,o,i=i:cap.__setitem__(i,o.detach().cpu().clone())))
  def run(x):
   pieces=[]
   with torch.no_grad():
    for ix in range(0,len(x),32):
     cap.clear();y=model(x[ix:ix+32]); pieces.append((y.cpu(),{k:v.clone() for k,v in cap.items()}))
   return pieces
  predtr=run(xtr);predva=run(xva)
  for h in handles:h.remove()
  acttr=np.concatenate([z[1][2].numpy() for z in predtr]);actva=np.concatenate([z[1][2].numpy() for z in predva]);yva=torch.cat([z[0] for z in predva])
  h=model.encoder_layers[2].feedforward[1].register_forward_hook(lambda m,a,o:actva_t.to(o.device,o.dtype))
  actva_t=torch.tensor(actva)
  with torch.no_grad():identity=model(xva)
  h.remove();err=float((identity-yva).abs().max())
  assert err<1e-6,err
  np.savez_compressed(OUT/'AO3_FFN_activations.npz',train=acttr,validation=actva,train_ids=tr['sample_id'],validation_ids=va['sample_id'])
  audit={'checkpoint_sha256':sha(CK),'train_cache_manifest_sha256':sha(CACHE/'split_manifest.json'),'train_shape':acttr.shape,'validation_shape':actva.shape,'min':float(acttr.min()),'max':float(acttr.max()),'negative_fraction':float((acttr<0).mean()),'zero_fraction':float((acttr==0).mean()),'identity_max_abs_error':err,'official_test_accessed':False}
  atomic_json(OUT/'activation_audit.json',audit);atomic_json(OUT/'status.json',{'status':'incomplete','stage':'nmf_pilot','activation_capture':'completed'})
  # Fixed sample rule: first 96 windows by sorted train sample ID, all 15 tokens.
  order=np.argsort(tr['sample_id'].astype(str),kind='stable')[:96];A=acttr[order].reshape(-1,128).astype(np.float64)
  # Feature-wise RMS scaling is invertible and improves numeric conditioning.
  scale=np.maximum(np.sqrt(np.mean(A*A,axis=0)),1e-8);As=A/scale
  records=[];chosen=None
  for rank in (4,8):
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',ConvergenceWarning);nmf=NMF(n_components=rank,init='nndsvda',random_state=20260928,max_iter=120,tol=1e-3).fit(As)
   err=float(np.linalg.norm(As-nmf.transform(As)@nmf.components_)/np.linalg.norm(As));warn=any(issubclass(w.category,ConvergenceWarning) for w in caught)
   records.append({'rank':rank,'train_relative_reconstruction_error':err,'iterations':int(nmf.n_iter_),'convergence_warning':warn})
   if chosen is None:chosen=(nmf,rank)
  nmf,rank=chosen;V=actva.reshape(-1,128).astype(np.float64)/scale
  coef=np.vstack([nnls(nmf.components_.T,row)[0] for row in V]);rec=coef@nmf.components_
  valerr=float(np.linalg.norm(V-rec)/np.linalg.norm(V))
  np.savez_compressed(OUT/'AO3_NMF_pilot.npz',components=nmf.components_,feature_scale=scale,validation_coefficients=coef,validation_reconstruction=rec*scale,rank=np.array(rank))
  atomic_csv(OUT/'nmf_pilot.csv',records);atomic_json(OUT/'pilot_metrics.json',{'selected_rank':'4 (small pilot rank; no semantic/rank claim)','validation_nnls_relative_error_scaled':valerr,'validation_rows':len(V),'warning_records':records,'runtime_seconds':time.perf_counter()-started})
  atomic_json(OUT/'status.json',{'status':'completed','stages':{'activation_capture':'completed','identity_intervention':'completed','nmf_nnls_pilot':'completed'},'runtime_seconds':time.perf_counter()-started,'official_test_accessed':False})
 finally: lock.unlink(missing_ok=True)
if __name__=='__main__':main()
