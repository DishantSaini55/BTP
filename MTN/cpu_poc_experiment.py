"""Train/evaluate the small, CPU-only MTN context proof-of-concept."""
from __future__ import annotations

import json, pickle, random, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from torch.utils.data import DataLoader

from baselineUtils import OnboardTfDataset, get_data
from context_TF import ContextIndividualTF
from individual_TF import IndividualTF

ROOT=Path(r"D:\IDD-PeD-Evaluation"); FLOW=ROOT/"flow"/"cpu_farneback_poc"; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
CKPT=ROOT/"checkpoints"/"mtn"/"mtn"/"epoch_latest.pth"; SEED=20260928
OPTIONS=dict(normalize_bbox=True,track_overlap=.5,observe_length=15,predict_length=45,enc_input_type=["bbox"],dec_input_type=["obd_speed","ego_op_flow","ped_op_flow"],prediction_type=["bbox"])
MODEL=dict(N=3,d_model=32,d_ff=128,h=4,dropout=.1)

def seed(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
def load(name):
    with (FLOW/f"{name}.pkl").open("rb") as f:return pickle.load(f)
def stats(raw):
    b=np.concatenate([np.asarray(x) for x in raw["bbox"]]); s=np.concatenate([np.asarray(x) for x in raw["obd_speed"]])
    return b.mean(0),np.maximum(b.std(0),1e-6),s.mean(0),np.maximum(s.std(0),1e-6)
def adapt(b):
    return {"boxes":b["enc_input"],"speed":b["obd_speed"],"ego_flow":b["ego_op_flow"],"ped_flow":b["ped_op_flow"],"scene_context":b.get("scene_context"),"observed_occlusion":b.get("observed_occlusion")}
def build(raw,flag,mean,std,ms,ss,context):
    data = get_data(raw,flag,mean,std,ms,ss,include_context=context,**OPTIONS)
    # The repository's dataset wrapper expects the display key used by its
    # higher-level factory.  This direct POC path calls get_data itself.
    data["image_name"] = data["obs_image"]
    return OnboardTfDataset(data,flag,mean,std)
def new_model(context):
    ck=torch.load(CKPT,map_location="cpu"); cls=ContextIndividualTF if context else IndividualTF
    m=cls(15,4,4,4,**MODEL)
    if context: missing=m.load_state_dict({"base."+k:v for k,v in ck.items()},strict=False).missing_keys; assert len(missing)==16
    else:m.load_state_dict(ck)
    return m
def train(context,train,val):
    seed(); m=new_model(context); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=torch.nn.MSELoss()
    tr=DataLoader(train,batch_size=8,shuffle=False); va=DataLoader(val,batch_size=8,shuffle=False); best=float("inf"); best_state=None; grad=None
    print(f"starting {'B' if context else 'A'}: train_windows={len(train)} validation_windows={len(val)}", flush=True)
    for epoch in range(12):
        m.train()
        for b in tr:
            x=adapt(b); opt.zero_grad(); pred=m.forward_batch(x) if context else m.forward_batch(x); loss=lossfn(pred,b["pred_target"]); loss.backward()
            if context and grad is None: grad=float(sum((p.grad.abs().sum() for n,p in m.named_parameters() if not n.startswith("base.") and p.grad is not None),torch.tensor(0.)).item())
            opt.step()
        m.eval(); losses=[]
        with torch.no_grad():
            for b in va: losses.append(lossfn(m.forward_batch(adapt(b)),b["pred_target"]).item())
        score=float(np.mean(losses))
        print(f"{'B' if context else 'A'} epoch={epoch + 1} validation_mse={score:.8f}", flush=True)
        if score<best: best=score; best_state={k:v.detach().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best_state); return m,best,grad
def evaluate(m,ds,mean,std,context):
    dl=DataLoader(ds,batch_size=1,shuffle=False); rec=[]; m.eval()
    with torch.no_grad():
      for b in dl:
        x=adapt(b); start=time.perf_counter(); pred=m.forward_batch(x); sec=time.perf_counter()-start
        target=b["pred_target"].numpy()[0]; p=pred.numpy()[0]; gt=target*std+mean; pp=p*std+mean
        center_gt=np.stack(((gt[:,0]+gt[:,2])/2,(gt[:,1]+gt[:,3])/2),-1); center_p=np.stack(((pp[:,0]+pp[:,2])/2,(pp[:,1]+pp[:,3])/2),-1)
        rec.append(dict(id=str(b["image_name"][0]),mse=float(np.mean((gt-pp)**2)),cmse=float(np.mean((center_gt-center_p)**2)),cfmse=float(np.mean((center_gt[-1]-center_p[-1])**2)),seconds=sec,gt=gt.tolist(),prediction=pp.tolist(),context=[int(v) for v in b.get("scene_context",torch.zeros(1,5))[0].tolist()]))
    return rec
def summarize(rows): return {k:float(np.mean([x[k] for x in rows])) for k in ("mse","cmse","cfmse","seconds")}
def report(a,b,meta,best_a,best_b,grad):
    pdf=OUT/"CPU_Farneback_MTN_Context_POC.pdf"; diff=[]
    for x,y in zip(a,b): diff.append({"sample":x["id"],"context":x["context"],"A_MSE":x["mse"],"B_MSE":y["mse"],"delta":y["mse"]-x["mse"],"A_CMSE":x["cmse"],"B_CMSE":y["cmse"],"A_CFMSE":x["cfmse"],"B_CFMSE":y["cfmse"]})
    result={"label":"CPU Farneback Optical-Flow Proof-of-Concept — Not an Exact Reproduction of the Authors' RAFT Pipeline","seed":SEED,"epochs":12,"validation_mse":{"A":best_a,"B":best_b},"context_gradient_l1_first_batch":grad,"aggregate":{"A":summarize(a),"B":summarize(b)},"samples":diff,"flow_config":meta}
    (OUT/"results.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    with PdfPages(pdf) as p:
      fig=plt.figure(figsize=(8.27,11.69)); plt.axis("off"); plt.text(.05,.94,"CPU-based small-scale proof-of-concept",fontsize=16,weight="bold",wrap=True); plt.text(.05,.88,result["label"],fontsize=12,weight="bold",wrap=True); plt.text(.05,.76,"Version A: official MTN backbone, fine-tuned on gp_set_0006_vid_0002.\nVersion B: identical backbone plus five native scene labels and observed occlusion; context parameters trained.\n\nStrict video-level partitions: train = vid_0002 (32 tracks); validation = vid_0001, frames ending before 4000 (8 tracks); held-out test = vid_0003 (10 tracks). No training trajectory is used for testing. The vid_0001 validation restriction excludes its confirmed decode-corrupted region.\n\nBoth versions use the identical cached CPU Farneback flow tensors. This is a small CPU proof-of-concept only; Farneback is not the authors' RAFT pipeline and these results are not a reproduction of the published benchmark.",fontsize=10,va="top",wrap=True); p.savefig(fig); plt.close(fig)
      fig,ax=plt.subplots(figsize=(8.27,5)); ax.axis("off"); ax.table(cellText=[[d["sample"],f'{d["A_MSE"]:.3f}',f'{d["B_MSE"]:.3f}',f'{d["delta"]:.3f}',f'{d["A_CMSE"]:.3f}',f'{d["B_CMSE"]:.3f}',f'{d["A_CFMSE"]:.3f}',f'{d["B_CFMSE"]:.3f}'] for d in diff],colLabels=["sample","A MSE","B MSE","Δ","A C-MSE","B C-MSE","A CF-MSE","B CF-MSE"],loc="center"); p.savefig(fig,bbox_inches="tight");plt.close(fig)
      for x,y in zip(a,b):
        gt=np.array(x["gt"]); pa=np.array(x["prediction"]); pb=np.array(y["prediction"]); fig,ax=plt.subplots(figsize=(7,5));
        for z,label,color in ((gt,"Ground truth","black"),(pa,"A: without context","tab:blue"),(pb,"B: with context","tab:orange")):
          c=np.stack(((z[:,0]+z[:,2])/2,(z[:,1]+z[:,3])/2),-1); ax.plot(c[:,0],c[:,1],label=label,color=color)
        ax.invert_yaxis();ax.legend();ax.set_title(f'{x["id"]}\ncontext={x["context"]}; A MSE={x["mse"]:.3f}, B MSE={y["mse"]:.3f}');ax.set_xlabel("x");ax.set_ylabel("y");p.savefig(fig);plt.close(fig)
    print(json.dumps(result["aggregate"],indent=2)); print(pdf)
def main():
    seed(); tr_raw,va_raw,te_raw=load("train"),load("validation"),load("test"); mean,std,ms,ss=stats(tr_raw)
    tr_a=build(tr_raw,"train",mean,std,ms,ss,False); va_a=build(va_raw,"validation",mean,std,ms,ss,False); te_a=build(te_raw,"test",mean,std,ms,ss,False)
    tr_b=build(tr_raw,"train",mean,std,ms,ss,True); va_b=build(va_raw,"validation",mean,std,ms,ss,True); te_b=build(te_raw,"test",mean,std,ms,ss,True)
    print(f"prepared A windows train/validation/test={len(tr_a)}/{len(va_a)}/{len(te_a)}; B windows={len(tr_b)}/{len(va_b)}/{len(te_b)}", flush=True)
    a,best_a,_=train(False,tr_a,va_a); b,best_b,grad=train(True,tr_b,va_b)
    torch.save(a.state_dict(),OUT/"version_a_cpu_poc.pth");torch.save(b.state_dict(),OUT/"version_b_cpu_poc.pth")
    meta=json.loads((FLOW/"farneback_config.json").read_text()); report(evaluate(a,te_a,mean,std,False),evaluate(b,te_b,mean,std,True),meta,best_a,best_b,grad)
if __name__=="__main__":main()
