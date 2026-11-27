"""Bounded CRAFT-inspired AO-2/AO-3 development explainability experiment."""
from __future__ import annotations
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, nnls
from scipy.stats import qmc, spearmanr
from sklearn.decomposition import NMF
from PIL import Image, ImageDraw
from MTN.ao_models_v1 import build_ao_model, constant_velocity_forecast
from MTN.run_ao_v1 import _read_split, _train_only_normalization

ROOT=Path('results_paper/phase_6d_craft_execution_v1'); CACHE=Path('results_paper/splits/ao_annotation_only_v1'); CK=Path('results_paper/AO_confirmation_v1')
SEED=20260928; BUDGET=7200.; RANKS=(4,8,12); INITS=(20260928,20260929,20260930); SOBOL_N=256
def hashf(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''): h.update(x)
 return h.hexdigest()
def atomic(p,text):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(text,encoding='utf8');q.replace(p)
def csvout(p,rows):
 if not rows:return
 q=p.with_suffix(p.suffix+'.tmp')
 with q.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 q.replace(p)
def check(start):
 if time.perf_counter()-start>BUDGET: raise TimeoutError('7200-second budget exhausted')
def capture(model,x):
 acts={}; handles=[]
 for i,l in enumerate(model.encoder_layers):
  handles.append(l.feedforward[1].register_forward_hook(lambda m,a,o,i=i:acts.__setitem__(i,o.detach().cpu())))
 with torch.no_grad(): y=model(x)
 for h in handles:h.remove()
 return y,acts
def forward_override(model,x,activation):
 h=model.encoder_layers[-1].feedforward[1].register_forward_hook(lambda m,a,o: activation.to(o.device,o.dtype))
 with torch.no_grad(): y=model(x)
 h.remove();return y
def normdata(train,val):
 s=_train_only_normalization(train); mean=torch.from_numpy(s['mean']);std=torch.from_numpy(s['std'])
 return ((torch.from_numpy(train['observed_boxes'])-mean)/std,(torch.from_numpy(val['observed_boxes'])-mean)/std,mean,std)
def nmfgrid(a):
 rows=[];fits={}
 for r in RANKS:
  comps=[];errs=[]
  for seed in INITS:
   n=NMF(n_components=r,init='nndsvda',random_state=seed,max_iter=160,tol=1e-4).fit(a)
   comps.append(n.components_);errs.append(float(np.linalg.norm(a-n.transform(a)@n.components_)/np.linalg.norm(a)))
  sims=[]
  for i in range(len(comps)):
   for j in range(i):
    c=np.corrcoef(comps[i],comps[j])[:r,r:]; ii,jj=linear_sum_assignment(-np.abs(c));sims.append(float(np.mean(np.abs(c[ii,jj]))))
  rows.append({'rank':r,'train_relative_error':np.mean(errs),'component_stability_mean_abs_correlation':np.mean(sims),'n_initializations':len(INITS)})
  fits[r]=NMF(n_components=r,init='nndsvda',random_state=SEED,max_iter=300,tol=1e-4).fit(a)
 chosen=min(rows,key=lambda z:(z['train_relative_error'],-z['component_stability_mean_abs_correlation']))['rank']
 return fits[chosen],rows,chosen
def outputs(y,x,mean,std):
 raw=y*std+mean; last=x[:,-1]*std+mean; cx=(raw[:,-1,0]+raw[:,-1,2])/2;cy=(raw[:,-1,1]+raw[:,-1,3])/2
 d=((cx-(last[:,0]+last[:,2])/2)**2+(cy-(last[:,1]+last[:,3])/2)**2).sqrt()
 cv=constant_velocity_forecast(x)*std+mean; corr=((raw-cv)**2).mean((1,2))
 return np.array([cx.mean().item(),cy.mean().item(),d.mean().item(),corr.mean().item()])
def chart(path,values):
 w,h=900,480;im=Image.new('RGB',(w,h),'white');d=ImageDraw.Draw(im);mx=max(max(values),1e-12)
 d.line((70,30,70,410),fill='black');d.line((70,410,870,410),fill='black')
 for i,v in enumerate(values):
  x=95+i*max(1,760//len(values));bw=max(3,620//len(values));top=410-int(v/mx*340);d.rectangle((x,top,x+bw,410),fill='#2474a6');d.text((x,420),str(i+1),fill='black')
 d.text((70,8),'Mean total-order Sobol importance',fill='black');im.save(path.with_suffix('.png'));im.save(path.with_suffix('.pdf'))
 bars=''.join(f'<rect x="{95+i*max(1,760//len(values))}" y="{410-int(v/mx*340)}" width="{max(3,620//len(values))}" height="{int(v/mx*340)}" fill="#2474a6"/>' for i,v in enumerate(values));path.with_suffix('.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="480"><text x="70" y="20">Mean total-order Sobol importance</text><line x1="70" y1="30" x2="70" y2="410" stroke="black"/><line x1="70" y1="410" x2="870" y2="410" stroke="black"/>{bars}</svg>',encoding='utf8')
def main():
 start=time.perf_counter();ROOT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
 train,val,manifest=_read_split(CACHE);xt,xv,mean,std=normdata(train,val)
 # deterministic video-aware 24-per-video validation subset for intervention only
 ids=val['sample_id'].astype(str); vids=val['video_id'].astype(str); selected=[]
 for v in sorted(set(vids)): selected += sorted(np.where(vids==v)[0],key=lambda i:ids[i])[:24]
 selected=np.array(selected); csvout(ROOT/'CRAFT_SAMPLE_MANIFEST.csv',[{'split':'train','sample_id':x,'used_for_bank':True,'used_for_intervention':False} for x in train['sample_id'].astype(str)]+[{'split':'validation','sample_id':x,'used_for_bank':False,'used_for_intervention':bool(i in set(selected))} for i,x in enumerate(ids)])
 allrec=[];recon=[];stab=[];sob=[];fid=[];assoc=[];fig=[]
 for mid in ('AO-3','AO-2'):
  check(start);p=CK/mid/f'seed_{SEED}'/'best_model.pt'; payload=torch.load(p,map_location='cpu',weights_only=False);m=build_ao_model(mid).eval();m.load_state_dict(payload['model_state']);
  yt,at=capture(m,xt);yv,av=capture(m,xv); a=at[2].numpy().reshape(-1,128); b=av[2].numpy().reshape(-1,128); n,grid,r=nmfgrid(a);stab += [dict(model_id=mid,selected_rank=r,**z) for z in grid];u=n.transform(a);uv=np.vstack([nnls(n.components_.T,row)[0] for row in b]); ar=uv@n.components_; recon.append({'model_id':mid,'rank':r,'train_relative_error':float(np.linalg.norm(a-u@n.components_)/np.linalg.norm(a)),'validation_relative_error':float(np.linalg.norm(b-ar)/np.linalg.norm(b))})
  # activation audit/identity, intervention uses final ReLU and preserves other paths
  yi=forward_override(m,xv,av[2]); ident=float((yi-yv).abs().max()); allrec.append({'model_id':mid,'checkpoint_sha256':hashf(p),'layer':'encoder_layers.2.feedforward.1(ReLU)','train_shape':str(tuple(at[2].shape)),'validation_shape':str(tuple(av[2].shape)),'minimum':float(a.min()),'maximum':float(a.max()),'negative_fraction':float((a<0).mean()),'zero_fraction':float((a==0).mean()),'identity_max_abs_error':ident})
  # motion associations, BH omitted: exploratory p values retained
  boxes=train['observed_boxes'];c=(boxes[:,:,:2]+boxes[:,:,2:])/2; vel=np.diff(c,axis=1); speed=np.linalg.norm(vel,axis=2).mean(1); acc=np.linalg.norm(np.diff(vel,axis=1),axis=2).mean(1); disp=c[:,-1]-c[:,0];scale=(boxes[:,:,2]-boxes[:,:,0])*(boxes[:,:,3]-boxes[:,:,1]); sv=np.std(scale,axis=1);scores=u.reshape(len(train['sample_id']),15,r).mean(1)
  for j in range(r):
   for name,z in [('dx',disp[:,0]),('dy',disp[:,1]),('speed',speed),('acceleration',acc),('scale_variation',sv)]:
    rho,pv=spearmanr(scores[:,j],z);assoc.append({'model_id':mid,'concept':f'C{j+1}','descriptor':name,'n':len(z),'spearman_rho':rho,'p_value_unadjusted':pv,'semantic_label':'not assigned'})
  # Sobol on validation subset: global mean functional under masked reconstructed activation
  xv0=xv[selected]; base=ar.reshape(len(xv),15,128)[selected]; q=qmc.Sobol(d=2*r,scramble=True,seed=SEED).random_base2(int(np.log2(SOBOL_N))); 
  def ev(mask):
   vals=[]
   for z in mask:
    forced=torch.from_numpy((uv.reshape(len(xv),15,r)[selected]*z)@n.components_).float();vals.append(outputs(forward_override(m,xv0,forced),xv0,mean,std))
   return np.asarray(vals)
  A=q[:,:r];B=q[:,r:];YA,YB=ev(A),ev(B)
  for j in range(r):
   AB=A.copy();AB[:,j]=B[:,j];YAB=ev(AB)
   for k,name in enumerate(('final_center_x','final_center_y','final_displacement_norm','cv_correction_energy')):
    var=np.var(np.r_[YA[:,k],YB[:,k]]); total=np.mean((YA[:,k]-YAB[:,k])**2)/(2*max(var,1e-12));sob.append({'model_id':mid,'concept':f'C{j+1}','output':name,'sobol_n':SOBOL_N,'total_order':total,'stable':'pilot_only'})
  # fidelity rank using average total, reconstruction/original paths
  rank=np.argsort([-np.mean([x['total_order'] for x in sob if x['model_id']==mid and x['concept']==f'C{j+1}']) for j in range(r)])
  orig=outputs(yv,xv,mean,std);full=outputs(forward_override(m,xv,torch.from_numpy(ar.reshape(len(xv),15,128)).float()),xv,mean,std)
  for mode,order in [('sobol',rank),('reverse',rank[::-1]),('random',np.random.default_rng(SEED).permutation(r))]:
   for step in range(r+1):
    mask=np.zeros(r) if mode=='sobol' else np.ones(r); chosen=order[:step] if mode=='sobol' else order[:step];mask[chosen]=1-mask[chosen] if mode!='sobol' else 1
    forced=torch.from_numpy((uv.reshape(len(xv),15,r)*mask)@n.components_).float();o=outputs(forward_override(m,xv,forced),xv,mean,std);fid.append({'model_id':mid,'order':mode,'step':step,'mean_abs_change_from_full':float(np.abs(o-full).mean())})
  chart(ROOT/f'{mid}_sobol',[np.mean([x['total_order'] for x in sob if x['model_id']==mid and x['concept']==f'C{j+1}']) for j in range(r)])
  fig += [{'model_id':mid,'figure':f'{mid}_sobol.{e}','description':'actual pilot Sobol mean total-order importance'} for e in ('png','pdf','svg')]
 csvout(ROOT/'CRAFT_ACTIVATION_AUDIT.csv',allrec);csvout(ROOT/'CRAFT_RECONSTRUCTION_RESULTS.csv',recon);csvout(ROOT/'CRAFT_CONCEPT_STABILITY.csv',stab);csvout(ROOT/'CRAFT_MOTION_ASSOCIATIONS.csv',assoc);csvout(ROOT/'CRAFT_SOBOL_IMPORTANCE.csv',sob);csvout(ROOT/'CRAFT_FIDELITY_RESULTS.csv',fid);csvout(ROOT/'CRAFT_FIGURE_MANIFEST.csv',fig)
 atomic(ROOT/'status.json',json.dumps({'status':'completed','seconds':time.perf_counter()-start,'official_test_accessed':False,'models':['AO-3','AO-2'],'budget_seconds':BUDGET},indent=2))
if __name__=='__main__':main()
