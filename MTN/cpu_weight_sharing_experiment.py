"""Matched CPU POC: current context MTN (B) vs tied-layer context MTN (C)."""
from __future__ import annotations
import json, pickle, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from baselineUtils import OnboardTfDataset, get_data
from context_TF import ContextIndividualTF

ROOT=Path(r"D:\IDD-PeD-Evaluation"); FLOW=ROOT/"flow"/"cpu_farneback_poc"; OUT=ROOT/"outputs"
OFFICIAL=ROOT/"checkpoints"/"mtn"/"mtn"/"epoch_latest.pth"; B_CKPT=OUT/"version_b_cpu_poc.pth"; C_CKPT=OUT/"version_c_weight_shared_cpu_poc.pth"
SEED=20260928; MODEL=dict(N=3,d_model=32,d_ff=128,h=4,dropout=.1); OPTS=dict(normalize_bbox=True,track_overlap=.5,observe_length=15,predict_length=45,enc_input_type=["bbox"],dec_input_type=["obd_speed","ego_op_flow","ped_op_flow"],prediction_type=["bbox"])
def seed(): random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
def load(n):
 with (FLOW/f"{n}.pkl").open("rb") as f:return pickle.load(f)
def stats(x):
 b=np.concatenate([np.asarray(a) for a in x["bbox"]]);s=np.concatenate([np.asarray(a) for a in x["obd_speed"]]);return b.mean(0),np.maximum(b.std(0),1e-6),s.mean(0),np.maximum(s.std(0),1e-6)
def build(raw,flag,mean,std,ms,ss):
 d=get_data(raw,flag,mean,std,ms,ss,include_context=True,**OPTS);d["image_name"]=d["obs_image"];return OnboardTfDataset(d,flag,mean,std)
def inputs(b): return {"boxes":b["enc_input"],"speed":b["obd_speed"],"ego_flow":b["ego_op_flow"],"ped_flow":b["ped_op_flow"],"scene_context":b["scene_context"],"observed_occlusion":b["observed_occlusion"]}
def new(shared=False): return ContextIndividualTF(15,4,4,4,share_layers=shared,**MODEL)
def init_c():
 """Load every compatible official MTN tensor; tie layers to official layer 0."""
 official=torch.load(OFFICIAL,map_location="cpu");m=new(True);state=m.state_dict();loaded=[]
 for key in state:
  source=key.removeprefix("base.").replace("model.encoder.layer.","model.encoder.layers.0.").replace("model.decoder.layer.","model.decoder.layers.0.")
  if source in official and official[source].shape==state[key].shape: state[key]=official[source];loaded.append(key)
 m.load_state_dict(state);return m,loaded
def count(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
def breakdown(m):
 out={"encoder":0,"decoder":0,"self_attention":0,"cross_attention":0,"ffn":0,"context_branch":0,"other":0}
 for n,p in m.named_parameters():
  k="other"
  if n.startswith("context_encoder") or n.startswith("gate"): k="context_branch"
  elif ".feed_forward." in n:k="ffn"
  elif any(x in n for x in (".src_attn.",".cross_attn.",".ped_attn.",".spd_attn.")):k="cross_attention"
  elif ".self_attn." in n:k="self_attention"
  elif ".encoder." in n:k="encoder"
  elif ".decoder." in n:k="decoder"
  out[k]+=p.numel()
 return out
def train_c(train,val):
 seed();m,loaded=init_c();opt=torch.optim.Adam(m.parameters(),lr=1e-3);loss=torch.nn.MSELoss();best=float("inf");best_state=None;grad=None;hist=[]
 for ep in range(12):
  m.train()
  for b in DataLoader(train,batch_size=8,shuffle=False):
   opt.zero_grad();pred=m.forward_batch(inputs(b));l=loss(pred,b["pred_target"]);l.backward()
   if grad is None:grad=float(sum((p.grad.abs().sum() for n,p in m.named_parameters() if not n.startswith("base.") and p.grad is not None),torch.tensor(0.)).item())
   opt.step()
  m.eval();scores=[]
  with torch.no_grad():
   for b in DataLoader(val,batch_size=8,shuffle=False):scores.append(loss(m.forward_batch(inputs(b)),b["pred_target"]).item())
  score=float(np.mean(scores));hist.append(score);print(f"C epoch={ep+1} validation_mse={score:.8f}",flush=True)
  if score<best:best=score;best_state={k:v.detach().clone() for k,v in m.state_dict().items()}
 m.load_state_dict(best_state);return m,best,grad,hist,len(loaded)
def evaluate(m,ds,mean,std):
 m.eval();rows=[]
 with torch.no_grad():
  for b in DataLoader(ds,batch_size=1,shuffle=False):
   start=time.perf_counter();p=m.forward_batch(inputs(b));sec=time.perf_counter()-start;gt=b["pred_target"].numpy()[0]*std+mean;pr=p.numpy()[0]*std+mean
   cg=np.stack(((gt[:,0]+gt[:,2])/2,(gt[:,1]+gt[:,3])/2),-1);cp=np.stack(((pr[:,0]+pr[:,2])/2,(pr[:,1]+pr[:,3])/2),-1)
   rows.append({"mse":float(np.mean((gt-pr)**2)),"cmse":float(np.mean((cg-cp)**2)),"cfmse":float(np.mean((cg[-1]-cp[-1])**2)),"seconds":sec})
 return {k:float(np.mean([r[k] for r in rows])) for k in ("mse","cmse","cfmse","seconds")},rows
def main():
 seed();tr,va,te=load("train"),load("validation"),load("test");mean,std,ms,ss=stats(tr);train,valid,test=build(tr,"train",mean,std,ms,ss),build(va,"validation",mean,std,ms,ss),build(te,"test",mean,std,ms,ss)
 # B is the saved, already-trained current architecture; evaluate it on exactly this test dataset.
 b=new(False);b.load_state_dict(torch.load(B_CKPT,map_location="cpu"));b_metrics,b_rows=evaluate(b,test,mean,std)
 c,best,grad,hist,loaded= train_c(train,valid);torch.save(c.state_dict(),C_CKPT);c_metrics,c_rows=evaluate(c,test,mean,std)
 variants={"B_current":new(False),"C_encoder_shared":ContextIndividualTF(15,4,4,4,share_encoder=True,**MODEL),"C_decoder_shared":ContextIndividualTF(15,4,4,4,share_decoder=True,**MODEL),"C_full_layer_shared":new(True)}
 result={"label":"CPU weight-sharing POC; same cached Farneback flows and 32/8/10 split","seed":SEED,"epochs":12,"shared_strategy":"one encoder layer and one decoder layer executed three times each","B":{"parameters":count(variants["B_current"]),"breakdown":breakdown(variants["B_current"]),"metrics":b_metrics,"checkpoint_bytes":B_CKPT.stat().st_size},"C":{"parameters":count(c),"breakdown":breakdown(c),"metrics":c_metrics,"checkpoint_bytes":C_CKPT.stat().st_size,"validation_mse":best,"context_gradient_l1_first_batch":grad,"official_tensors_loaded":loaded,"validation_history":hist},"candidate_parameter_counts":{n:count(x) for n,x in variants.items()},"per_sample":{"B":b_rows,"C":c_rows},"flow_config":json.loads((FLOW/"farneback_config.json").read_text())}
 (OUT/"weight_sharing_results.json").write_text(json.dumps(result,indent=2),encoding="utf8");print(json.dumps({"B":b_metrics,"C":c_metrics,"params":result["candidate_parameter_counts"]},indent=2))
if __name__=="__main__":main()
