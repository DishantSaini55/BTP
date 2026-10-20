"""Resumable unified B/C4/F/F+C4 CPU POC runner."""
import argparse,json,logging,random,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
import cpu_c4_larger_validation as L
import cpu_targeted_weight_sharing as C
from dynamic_fusion_TF import FusionContextIndividualTF
from c4_encoder_shared import make_c4_encoder_self_attention_shared,verify_c4_sharing
ROOT=Path(r'D:\IDD-PeD-Evaluation');OUT=ROOT/'outputs'/'final_validation';LOG=ROOT/'runs'/'final_four_way_validation.log';SEEDS=[20260928,20260929,20260930];MODELS=['B','C4','F','F_C4']
LOG.parent.mkdir(parents=True,exist_ok=True);logging.basicConfig(filename=LOG,level=logging.INFO,format='%(asctime)s %(message)s')
def fnew(c4=False):
 m=FusionContextIndividualTF(15,4,4,4,N=3,d_model=32,d_ff=128,h=4,dropout=.1);o=torch.load(C.OFFICIAL,map_location='cpu');s=m.state_dict();n=0
 for k in s:
  q=k.removeprefix('base.')
  if q in o and o[q].shape==s[k].shape:s[k]=o[q];n+=1
 m.load_state_dict(s)
 if c4:make_c4_encoder_self_attention_shared(m)
 return m,n
def setseed(x):random.seed(x);np.random.seed(x);torch.manual_seed(x)
def build(model,seed,td,vd):
 setseed(seed)
 if model in ('B','C4'):m,n=C.new('encoder_self_attn' if model=='C4' else 'none')
 else:m,n=fnew(model=='F_C4')
 opt=torch.optim.Adam(m.parameters(),lr=.001);loss=torch.nn.MSELoss();m.train();t=time.perf_counter()
 for b in DataLoader(td,batch_size=8,shuffle=False):opt.zero_grad();z=m.forward_batch(C.inputs(b));v=loss(z,b['pred_target']);v.backward();opt.step()
 m.eval();vals=[]
 with torch.no_grad():
  for b in DataLoader(vd,batch_size=8,shuffle=False):vals.append(loss(m.forward_batch(C.inputs(b)),b['pred_target']).item())
 return m,{'training_seconds':time.perf_counter()-t,'validation_mse':float(np.mean(vals)),'official_tensors_loaded':n}
def main(check=False):
 OUT.mkdir(parents=True,exist_ok=True);rpath=OUT/'final_four_way_results.json';results=json.loads(rpath.read_text()) if rpath.exists() else {'protocol':'Unified fixed 32/8/24 CPU Farneback POC; Adam 1e-3; batch 8; MSE; one epoch; three seeds.','models':{}}
 tr=L.load(C.FLOW/'train.pkl');va=L.load(C.FLOW/'validation.pkl');te=L.load(L.FLOW_LARGE/'test_full.pkl');mean,std,ms,ss=C.stats(tr);td=C.build(tr,'train',mean,std,ms,ss);vd=C.build(va,'validation',mean,std,ms,ss);ed=C.build(te,'test',mean,std,ms,ss);batches=list(DataLoader(ed,batch_size=1,shuffle=False))
 if check:
  shapes={}
  for name in MODELS:
   m,_=build(name,SEEDS[0],td,vd);shapes[name]=tuple(m.forward_batch(C.inputs(next(iter(batches)))).shape)
  print(json.dumps({'RUNNER_READY':'4 MODELS x 3 SEEDS = 12 RUNS','configuration':results['protocol'],'shapes':shapes,'output':str(OUT),'log':str(LOG)},indent=2));return
 for name in MODELS:
  for seed in SEEDS:
   if str(seed) in results['models'].get(name,{}):continue
   try:
    logging.info('START %s seed %s',name,seed);m,info=build(name,seed,td,vd);folder=OUT/name/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True);p=folder/'checkpoint.pth';torch.save(m.state_dict(),p);metrics,rows=C.evaluate(m,ed,mean,std);lat=L.latency(m,batches);item={'parameters':C.count(m),'trainable_parameters':C.count(m),'checkpoint_bytes':p.stat().st_size,'training':info,'metrics':metrics,'latency':lat,'sharing_verification':verify_c4_sharing(m) if name in ('C4','F_C4') else None};results['models'].setdefault(name,{})[str(seed)]=item;rpath.write_text(json.dumps(results,indent=2));logging.info('COMPLETE %s seed %s',name,seed)
   except Exception as e:logging.exception('FAILED %s seed %s',name,seed);rpath.write_text(json.dumps(results,indent=2));raise
if __name__=='__main__':main('--check' in __import__('sys').argv)
