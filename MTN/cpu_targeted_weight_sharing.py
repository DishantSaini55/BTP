"""Matched targeted-attention-sharing ablation for the existing CPU POC."""
from __future__ import annotations
import json, pickle, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from baselineUtils import OnboardTfDataset, get_data
from context_TF import ContextIndividualTF
from targeted_weight_sharing import apply_targeted_sharing, STRATEGY_LABELS
from c4_encoder_shared import make_c4_encoder_self_attention_shared

ROOT=Path(r"D:\IDD-PeD-Evaluation");FLOW=ROOT/"flow"/"cpu_farneback_poc";OUT=ROOT/"outputs"/"weight_sharing_targeted";OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
OFFICIAL=ROOT/"checkpoints"/"mtn"/"mtn"/"epoch_latest.pth"; B_CKPT=ROOT/"outputs"/"version_b_cpu_poc.pth"; SEED=20260928; KW=dict(N=3,d_model=32,d_ff=128,h=4,dropout=.1); OPTS=dict(normalize_bbox=True,track_overlap=.5,observe_length=15,predict_length=45,enc_input_type=["bbox"],dec_input_type=["obd_speed","ego_op_flow","ped_op_flow"],prediction_type=["bbox"])
STRATEGIES=["decoder_src_attn","decoder_src_qkv","decoder_src_kv","encoder_self_attn"]
def seed():random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
def load(n):
 with (FLOW/f"{n}.pkl").open("rb") as f:return pickle.load(f)
def stats(x):
 b=np.concatenate([np.asarray(a) for a in x["bbox"]]);s=np.concatenate([np.asarray(a) for a in x["obd_speed"]]);return b.mean(0),np.maximum(b.std(0),1e-6),s.mean(0),np.maximum(s.std(0),1e-6)
def build(raw,flag,mean,std,ms,ss):
 d=get_data(raw,flag,mean,std,ms,ss,include_context=True,**OPTS);d["image_name"]=d["obs_image"];return OnboardTfDataset(d,flag,mean,std)
def inputs(b):return {"boxes":b["enc_input"],"speed":b["obd_speed"],"ego_flow":b["ego_op_flow"],"ped_flow":b["ped_op_flow"],"scene_context":b["scene_context"],"observed_occlusion":b["observed_occlusion"]}
def new(strategy="none"):
 m=ContextIndividualTF(15,4,4,4,**KW);official=torch.load(OFFICIAL,map_location="cpu");state=m.state_dict();loaded=[]
 for k in state:
  source=k.removeprefix("base.")
  if source in official and official[source].shape==state[k].shape:state[k]=official[source];loaded.append(k)
 m.load_state_dict(state)
 if strategy == "encoder_self_attn":
  make_c4_encoder_self_attention_shared(m)
 else:
  apply_targeted_sharing(m,strategy)
 return m,len(loaded)
def count(m):return sum(p.numel() for p in m.parameters() if p.requires_grad)
def attention_count(m):return sum(p.numel() for n,p in m.named_parameters() if any(s in n for s in ("self_attn","src_attn","cross_attn","ped_attn","spd_attn")))
def train(strategy,train,val):
 seed();m,loaded=new(strategy);opt=torch.optim.Adam(m.parameters(),lr=1e-3);loss=torch.nn.MSELoss();best=float("inf");best_state=None;hist=[];grad=None
 for ep in range(1):
  m.train()
  for b in DataLoader(train,batch_size=8,shuffle=False):
   opt.zero_grad();pred=m.forward_batch(inputs(b));l=loss(pred,b["pred_target"]);l.backward()
   if grad is None:grad=float(sum((p.grad.abs().sum() for n,p in m.named_parameters() if not n.startswith("base.") and p.grad is not None),torch.tensor(0.)).item())
   opt.step()
  m.eval();scores=[]
  with torch.no_grad():
   for b in DataLoader(val,batch_size=8,shuffle=False):scores.append(loss(m.forward_batch(inputs(b)),b["pred_target"]).item())
  sc=float(np.mean(scores));hist.append(sc);print(f"{strategy} epoch={ep+1} val_mse={sc:.8f}",flush=True)
  if sc<best:best=sc;best_state={k:v.detach().clone() for k,v in m.state_dict().items()}
 m.load_state_dict(best_state);return m,{"validation_mse":best,"context_gradient_l1_first_batch":grad,"validation_history":hist,"official_tensors_loaded":loaded}
def evaluate(m,ds,mean,std):
 m.eval();rows=[]
 with torch.no_grad():
  for b in DataLoader(ds,batch_size=1,shuffle=False):
   st=time.perf_counter();p=m.forward_batch(inputs(b));sec=time.perf_counter()-st;gt=b["pred_target"].numpy()[0]*std+mean;pr=p.numpy()[0]*std+mean;cg=np.stack(((gt[:,0]+gt[:,2])/2,(gt[:,1]+gt[:,3])/2),-1);cp=np.stack(((pr[:,0]+pr[:,2])/2,(pr[:,1]+pr[:,3])/2),-1)
   rows.append({"mse":float(np.mean((gt-pr)**2)),"cmse":float(np.mean((cg-cp)**2)),"cfmse":float(np.mean((cg[-1]-cp[-1])**2)),"seconds":sec})
 return {k:float(np.mean([r[k] for r in rows])) for k in ("mse","cmse","cfmse","seconds")},rows
def warm_latency(m,batches):
 m.eval()
 with torch.no_grad():
  for _ in range(5):
   for b in batches:m.forward_batch(inputs(b))
  times=[]
  for _ in range(20):
   for b in batches:st=time.perf_counter();m.forward_batch(inputs(b));times.append(time.perf_counter()-st)
 return {"mean_seconds":float(np.mean(times)),"std_seconds":float(np.std(times,ddof=1)),"runs":len(times),"warmup_passes":5}
def main():
 seed();tr,va,te=load("train"),load("validation"),load("test");mean,std,ms,ss=stats(tr);train_ds,valid_ds,test_ds=build(tr,"train",mean,std,ms,ss),build(va,"validation",mean,std,ms,ss),build(te,"test",mean,std,ms,ss);batches=list(DataLoader(test_ds,batch_size=1,shuffle=False))
 b,baseline_train=train("none",train_ds,valid_ds);baseline_path=OUT/"B_no_sharing.pth";torch.save(b.state_dict(),baseline_path);b_metrics,b_rows=evaluate(b,test_ds,mean,std)
 results={"protocol":"same cached Farneback flows; same 32/8/10 split; seed 20260928; Adam lr 1e-3; batch 8; MSE; 12 epochs","B":{"strategy":"No sharing","parameters":count(b),"attention_parameters":attention_count(b),"metrics":b_metrics,"per_sample":b_rows,"latency":warm_latency(b,batches),"checkpoint_bytes":B_CKPT.stat().st_size},"variants":{},"not_run":{"C3_decoder_self_attention":"Not applicable: current DecoderLayer has no self_attn module. It uses src_attn, spd_attn, and ped_attn cross-attention paths.","C5_head_sharing":"Not implemented: each MultiHeadAttention uses four full d_model×d_model projections, then reshapes into heads; there are no independent per-head Q/K/V parameter tensors to alias safely without redesigning the projection parameterization."},"architecture_facts":{"encoder_layers":3,"decoder_layers":3,"attention_modules":18,"attention_calls_per_forward":18,"d_model":32,"heads":4,"d_ff":128}}
 results["protocol"]="same cached Farneback flows; same 32/8/10 split; seed 20260928; Adam lr 1e-3; batch 8; MSE; one epoch; CPU thread cap 4. Baseline and every targeted variant were independently initialized from the same official checkpoint and trained identically."
 results["B"]["checkpoint_bytes"]=baseline_path.stat().st_size;results["B"]["training"]=baseline_train;results["architecture_facts"]["cpu_threads"]=4
 for strategy in STRATEGIES:
  m,train_info=train(strategy,train_ds,valid_ds);path=OUT/f"{strategy}.pth";torch.save(m.state_dict(),path);metrics,rows=evaluate(m,test_ds,mean,std);results["variants"][strategy]={"shared_at":STRATEGY_LABELS[strategy],"parameters":count(m),"attention_parameters":attention_count(m),"metrics":metrics,"per_sample":rows,"latency":warm_latency(m,batches),"checkpoint_bytes":path.stat().st_size,"training":train_info};print(json.dumps({strategy:{"parameters":count(m),"metrics":metrics}},indent=2),flush=True)
 (OUT/"targeted_weight_sharing_results.json").write_text(json.dumps(results,indent=2),encoding="utf8")
 print("completed",OUT/"targeted_weight_sharing_results.json")
if __name__=="__main__":main()
