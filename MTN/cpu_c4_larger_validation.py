"""Three-seed matched B-vs-C4 validation on all locally available GP6 holdout tracks."""
from __future__ import annotations
import csv, json, pickle, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import cpu_targeted_weight_sharing as common
from c4_encoder_shared import verify_c4_sharing

OUT=Path(r"D:\IDD-PeD-Evaluation\outputs\weight_sharing_targeted")
FLOW_LARGE=Path(r"D:\IDD-PeD-Evaluation\flow\cpu_farneback_c4_larger")
SEEDS=[20260928,20260929,20260930]; EPOCHS=1; BATCH=8; LR=1e-3
def set_seed(seed): random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def load(path):
 with path.open('rb') as f:return pickle.load(f)
def train(strategy,seed,train_ds,valid_ds):
 set_seed(seed); model,loaded=common.new(strategy);opt=torch.optim.Adam(model.parameters(),lr=LR);loss=torch.nn.MSELoss();best=float('inf');best_state=None;history=[];start=time.perf_counter()
 for epoch in range(EPOCHS):
  model.train()
  for batch in DataLoader(train_ds,batch_size=BATCH,shuffle=False):
   opt.zero_grad();pred=model.forward_batch(common.inputs(batch));value=loss(pred,batch['pred_target']);value.backward();opt.step()
  model.eval();values=[]
  with torch.no_grad():
   for batch in DataLoader(valid_ds,batch_size=BATCH,shuffle=False):values.append(loss(model.forward_batch(common.inputs(batch)),batch['pred_target']).item())
  score=float(np.mean(values));history.append(score)
  if score<best:best=score;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(best_state)
 return model,{'validation_mse':best,'validation_history':history,'training_seconds':time.perf_counter()-start,'official_tensors_loaded':loaded}
def latency(model,batches):
 model.eval()
 with torch.no_grad():
  for _ in range(2):
   for batch in batches:model.forward_batch(common.inputs(batch))
  values=[]
  for _ in range(5):
   for batch in batches:
    started=time.perf_counter();model.forward_batch(common.inputs(batch));values.append(time.perf_counter()-started)
 return {'mean_seconds':float(np.mean(values)),'std_seconds':float(np.std(values,ddof=1)),'runs':len(values),'warmup_passes':2}
def run(name,strategy,seed,train_ds,valid_ds,test_ds,mean,std,batches,manifest):
 model,info=train(strategy,seed,train_ds,valid_ds);path=OUT/f"C4_larger_{name}_seed_{seed}.pth";torch.save(model.state_dict(),path);metrics,per=common.evaluate(model,test_ds,mean,std)
 for i,row in enumerate(per): row.update({'test_index':i,'source_index':manifest[i]['source_index'],'pid':manifest[i]['pid'],'first_frame':manifest[i]['first_frame']})
 return {'parameters':common.count(model),'trainable_parameters':common.count(model),'checkpoint_bytes':path.stat().st_size,'training':info,'metrics':metrics,'per_sample':per,'latency':latency(model,batches),'sharing_verification':verify_c4_sharing(model) if strategy=='encoder_self_attn' else None}
def meanstd(values):return {'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1))}
def main():
 train_raw=load(common.FLOW/'train.pkl');valid_raw=load(common.FLOW/'validation.pkl');test_raw=load(FLOW_LARGE/'test_full.pkl');manifest=json.loads((FLOW_LARGE/'test_full_manifest.json').read_text(encoding='utf8'))
 mean,std,ms,ss=common.stats(train_raw);train_ds=common.build(train_raw,'train',mean,std,ms,ss);valid_ds=common.build(valid_raw,'validation',mean,std,ms,ss);test_ds=common.build(test_raw,'test',mean,std,ms,ss);batches=list(DataLoader(test_ds,batch_size=1,shuffle=False))
 result={'label':'Larger CPU-based C4 validation/ablation — Farneback flow, not an exact authors\' RAFT reproduction','configuration':{'seeds':SEEDS,'epochs':EPOCHS,'batch_size':BATCH,'optimizer':'Adam','learning_rate':LR,'loss':'MSE','cpu_threads':4,'train_trajectories':len(train_ds),'validation_trajectories':len(valid_ds),'test_trajectories':len(test_ds),'test_videos':['gp_set_0006_vid_0003'],'test_selection_rule':'all 24 eligible video-3 trajectories in source order; no outcome-based selection','flow_train_validation':str(common.FLOW),'flow_test':str(FLOW_LARGE),'latency_protocol':'batch=1; 2 warm-up passes then 5 complete traversals of all test trajectories'},'manifest':manifest,'seeds':{},'sharing_verification':None}
 for seed in SEEDS:
  b=run('B','none',seed,train_ds,valid_ds,test_ds,mean,std,batches,manifest);c=run('C4','encoder_self_attn',seed,train_ds,valid_ds,test_ds,mean,std,batches,manifest)
  result['seeds'][str(seed)]={'B':b,'C4':c};result['sharing_verification']=c['sharing_verification'];print(json.dumps({'seed':seed,'B':b['metrics'],'C4':c['metrics']},indent=2),flush=True)
 aggregate={}
 for name in ('B','C4'):
  aggregate[name]={'parameters':result['seeds'][str(SEEDS[0])][name]['parameters'],'checkpoint_bytes':meanstd([result['seeds'][str(s)][name]['checkpoint_bytes'] for s in SEEDS]),'training_seconds':meanstd([result['seeds'][str(s)][name]['training']['training_seconds'] for s in SEEDS]),'latency_seconds':meanstd([result['seeds'][str(s)][name]['latency']['mean_seconds'] for s in SEEDS])}
  for metric in ('mse','cmse','cfmse'):aggregate[name][metric]=meanstd([result['seeds'][str(s)][name]['metrics'][metric] for s in SEEDS])
 changes={}
 for metric in ('mse','cmse','cfmse'):
  b=aggregate['B'][metric]['mean'];c=aggregate['C4'][metric]['mean'];changes[metric]={'absolute':c-b,'percent':100*(c-b)/b}
 b=aggregate['B']['latency_seconds']['mean'];c=aggregate['C4']['latency_seconds']['mean'];changes['latency_seconds']={'absolute':c-b,'percent':100*(c-b)/b};changes['parameters']={'absolute':aggregate['C4']['parameters']-aggregate['B']['parameters'],'percent':100*(aggregate['C4']['parameters']-aggregate['B']['parameters'])/aggregate['B']['parameters']}
 better=[]
 for seed in SEEDS:
  data=result['seeds'][str(seed)];pairs=[]
  for b,c in zip(data['B']['per_sample'],data['C4']['per_sample']):pairs.append({'seed':seed,'test_index':b['test_index'],'source_index':b['source_index'],'pid':b['pid'],'B_mse':b['mse'],'C4_mse':c['mse'],'better_model':'C4' if c['mse']<b['mse'] else ('B' if b['mse']<c['mse'] else 'tie')})
  better.extend(pairs)
 result['aggregate']=aggregate;result['changes_C4_minus_B']=changes;result['sample_comparison']=better;result['sample_counts']={'C4_better':sum(x['better_model']=='C4' for x in better),'B_better':sum(x['better_model']=='B' for x in better),'ties':sum(x['better_model']=='tie' for x in better),'total':len(better)}
 (OUT/'C4_larger_validation_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 with (OUT/'C4_larger_validation_summary.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=['seed','test_index','source_index','pid','B_mse','C4_mse','better_model']);w.writeheader();w.writerows(better)
 (OUT/'C4_larger_validation_config_manifest.json').write_text(json.dumps({'configuration':result['configuration'],'manifest':manifest},indent=2),encoding='utf8')
 print(json.dumps({'aggregate':aggregate,'changes':changes,'sample_counts':result['sample_counts']},indent=2))
if __name__=='__main__':main()
