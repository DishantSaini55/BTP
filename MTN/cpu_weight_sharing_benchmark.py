"""Warmed repeated forward-pass timing for the saved B and C POC checkpoints."""
import json, pickle, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from baselineUtils import OnboardTfDataset, get_data
from context_TF import ContextIndividualTF

ROOT=Path(r"D:\IDD-PeD-Evaluation");FLOW=ROOT/"flow"/"cpu_farneback_poc";OUT=ROOT/"outputs";KW=dict(N=3,d_model=32,d_ff=128,h=4,dropout=.1);OPTS=dict(normalize_bbox=True,track_overlap=.5,observe_length=15,predict_length=45,enc_input_type=["bbox"],dec_input_type=["obd_speed","ego_op_flow","ped_op_flow"],prediction_type=["bbox"])
def main():
 with (FLOW/"train.pkl").open("rb") as f:tr=pickle.load(f)
 with (FLOW/"test.pkl").open("rb") as f:te=pickle.load(f)
 bb=np.concatenate([np.asarray(x) for x in tr["bbox"]]);sp=np.concatenate([np.asarray(x) for x in tr["obd_speed"]]);mean,std,ms,ss=bb.mean(0),np.maximum(bb.std(0),1e-6),sp.mean(0),np.maximum(sp.std(0),1e-6)
 d=get_data(te,"test",mean,std,ms,ss,include_context=True,**OPTS);d["image_name"]=d["obs_image"];batches=list(DataLoader(OnboardTfDataset(d,"test",mean,std),batch_size=1,shuffle=False))
 def inp(x):return {"boxes":x["enc_input"],"speed":x["obd_speed"],"ego_flow":x["ego_op_flow"],"ped_flow":x["ped_op_flow"],"scene_context":x["scene_context"],"observed_occlusion":x["observed_occlusion"]}
 def bench(shared,path):
  m=ContextIndividualTF(15,4,4,4,share_layers=shared,**KW);m.load_state_dict(torch.load(path,map_location="cpu"));m.eval()
  with torch.no_grad():
   for _ in range(5):
    for x in batches:m.forward_batch(inp(x))
   t=[]
   for _ in range(20):
    for x in batches:
     s=time.perf_counter();m.forward_batch(inp(x));t.append(time.perf_counter()-s)
  return {"mean_seconds":float(np.mean(t)),"std_seconds":float(np.std(t,ddof=1)),"runs":len(t),"warmup_passes":5}
 r=json.loads((OUT/"weight_sharing_results.json").read_text());r["latency_benchmark"]={"B":bench(False,OUT/"version_b_cpu_poc.pth"),"C":bench(True,OUT/"version_c_weight_shared_cpu_poc.pth")};(OUT/"weight_sharing_results.json").write_text(json.dumps(r,indent=2));print(json.dumps(r["latency_benchmark"],indent=2))
if __name__=="__main__":main()
