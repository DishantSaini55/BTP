"""Build a concise evidence-based report for the dedicated C4 reproduction."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"DejaVu Sans"})
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(r"D:\IDD-PeD-Evaluation\outputs\weight_sharing_targeted")
R=json.loads((ROOT/"C4_reproduction_results.json").read_text(encoding="utf8")); B=R["B"]; C=R["C4"]
OUT=ROOT/"C4_Encoder_SelfAttention_Sharing_Report.pdf"
NAVY="#102A43";BLUE="#2F80ED";TEAL="#00A6A6";PALE="#F7FAFC";RED="#D64545"
def fig(title,sub):
 f=plt.figure(figsize=(11.69,8.27),facecolor="white");f.text(.065,.93,title,fontsize=22,fontweight="bold",color=NAVY);f.text(.065,.892,sub,fontsize=9.5,color="#52616B");f.add_artist(plt.Line2D([.065,.935],[.875,.875],color=TEAL,lw=2));return f
def text(f,x,y,s,size=10,weight="normal",color="#263238") : f.text(x,y,s,fontsize=size,fontweight=weight,color=color,va="top")
def tab(ax,cols,rows,widths):
 ax.axis("off");t=ax.table(cellText=rows,colLabels=cols,loc="center",cellLoc="left",colLoc="left",colWidths=widths);t.auto_set_font_size(False);t.set_fontsize(8.5);t.scale(1,1.55)
 for (r,c),cell in t.get_celld().items():
  cell.set_edgecolor("#D9E2EC")
  if r==0: cell.set_facecolor(NAVY);cell.get_text().set_color("white");cell.get_text().set_weight("bold")
  elif r%2==0:cell.set_facecolor(PALE)
def pct(v,b):return 100*(v-b)/b
with PdfPages(OUT) as pdf:
 f=fig("C4 Encoder Self-Attention Sharing","Dedicated reproduction on the verified CPU Farneback POC — measured evidence, not a full benchmark")
 text(f,.065,.82,"Motivation",13,"bold",NAVY);text(f,.065,.785,"The targeted ablation found a promising result when only encoder self-attention was tied across depth. C4 tests that\nselective change while preserving Version B's native context branch and every decoder path.",10.6)
 text(f,.065,.68,"C4 change",13,"bold",NAVY);text(f,.065,.642,"Encoder Layer 1 self-attention is the shared reference. Encoder Layers 2 and 3 use the exact same Q, K, V and O\nparameter objects. Encoder depth executions remain three; only storage/learnable degrees of freedom are reduced.",10.6)
 text(f,.065,.53,"Unchanged",13,"bold",NAVY);text(f,.065,.493,"Context inputs, six 16-D embeddings, 96→32→32 projection, dynamic gate, context token fusion, encoder non-self\nattention and FFN, decoder src/speed/ped attention, decoder FFN, output head, data, flow, and evaluation code.",10.4)
 text(f,.065,.37,"Reproduced outcome",13,"bold",NAVY);text(f,.065,.33,f"B MSE {B['metrics']['mse']:.3f}  →  C4 MSE {C['metrics']['mse']:.3f}  ({pct(C['metrics']['mse'],B['metrics']['mse']):+.1f}%)",18,"bold",TEAL)
 text(f,.065,.21,"Interpretation is deliberately limited: this is a 10-sample, one-epoch CPU proof-of-concept using Farneback flow,\nnot an exact RAFT reproduction or a basis to call C4 the final architecture.",9.5,"normal",RED);pdf.savefig(f);plt.close(f)

 f=fig("Implementation and parameter audit","Programmatic identity checks were run on the actual C4 model after official-checkpoint initialization.")
 ev=C["implementation_verification"]
 rows=[]
 for p,v in ev["projections"].items(): rows.append([p,"Yes" if v["same_module_object"] else "No","Yes" if v["same_weight_parameter_object"] else "No","Yes" if v["same_weight_data_ptr"] else "No","Yes" if v["same_bias_parameter_object"] else "No"])
 ax=f.add_axes([.09,.55,.82,.21]);tab(ax,["Projection","Same module","Same weight object","Same weight storage","Same bias object"],rows,[.16,.18,.22,.23,.2])
 rows2=[["Version B total/trainable",f"{B['parameters']:,}"],["C4 total/trainable",f"{C['parameters']:,}"],["Absolute reduction",f"{R['comparison']['parameter_reduction']:,}"],["Percentage reduction",f"{R['comparison']['parameter_reduction_percent']:.2f}%"],["Named parameter entries / unique objects",f"{ev['optimizer_registration']['named_parameter_entries']} / {ev['optimizer_registration']['unique_parameter_objects']}"],["Duplicate optimizer parameters",str(ev['optimizer_registration']['has_duplicate_parameter_objects'])]]
 ax=f.add_axes([.20,.21,.60,.22]);tab(ax,["Audit item","Actual result"],rows2,[.48,.48])
 text(f,.065,.82,"Verification result: all Q/K/V/O modules, weights, biases, and underlying storage are shared by identity across\nencoder layers 0, 1, and 2; optimizer registration contains no duplicate independent parameter objects.",10.4,"bold",TEAL);pdf.savefig(f);plt.close(f)

 f=fig("Fair experimental setup and aggregate results","Both models were independently initialized from the same official MTN checkpoint and trained/evaluated under the same recorded protocol.")
 text(f,.065,.82,R["protocol"],9.6)
 rows=[["Version B — no sharing",f"{B['parameters']:,}",f"{B['metrics']['mse']:.3f}",f"{B['metrics']['cmse']:.3f}",f"{B['metrics']['cfmse']:.3f}",f"{B['latency']['mean_seconds']*1000:.2f}",f"{B['checkpoint_bytes']/1024/1024:.2f}",f"{B['training_seconds']:.2f}"],["C4 — encoder self-attn Q/K/V/O",f"{C['parameters']:,}",f"{C['metrics']['mse']:.3f}",f"{C['metrics']['cmse']:.3f}",f"{C['metrics']['cfmse']:.3f}",f"{C['latency']['mean_seconds']*1000:.2f}",f"{C['checkpoint_bytes']/1024/1024:.2f}",f"{C['training_seconds']:.2f}"]]
 ax=f.add_axes([.045,.48,.91,.22]);tab(ax,["Model","Parameters","MSE","C-MSE","CF-MSE","Mean latency ms","Checkpoint MiB","Train s"],rows,[.26,.1,.09,.1,.1,.12,.12,.09])
 text(f,.065,.37,f"C4 Δ MSE: {R['comparison']['mse_delta']:.3f} ({R['comparison']['mse_delta_percent']:+.1f}%).  C4 improved {R['comparison']['samples_c4_better_mse']} / 10 held-out samples and was worse on {R['comparison']['samples_c4_worse_mse']} / 10.",11,"bold",TEAL)
 text(f,.065,.26,"CPU memory was not collected because the locked environment has no psutil. Checkpoint size is a durable storage proxy;\nthe C4 checkpoint is smaller because only one encoder self-attention module is serialized as shared parameters.",9.5,"normal","#52616B");pdf.savefig(f);plt.close(f)

 f=fig("Held-out sample comparison","Per-sample MSE on exactly the same 10 trajectories; lower is better.")
 ax=f.add_axes([.09,.42,.84,.34]);x=np.arange(1,11);bm=np.array([r['mse'] for r in B['per_sample']]);cm=np.array([r['mse'] for r in C['per_sample']]);ax.plot(x,bm,"o-",lw=2,color=BLUE,label="Version B");ax.plot(x,cm,"o-",lw=2,color=TEAL,label="C4");ax.set_xticks(x);ax.set_xlabel("Held-out sample index");ax.set_ylabel("MSE");ax.grid(alpha=.25);ax.legend()
 rows=[[str(i),f"{bb:.2f}",f"{cc:.2f}","C4" if cc<bb else "B"] for i,(bb,cc) in enumerate(zip(bm,cm),1)]
 ax=f.add_axes([.16,.08,.68,.23]);tab(ax,["Sample","B MSE","C4 MSE","Lower MSE"],rows,[.14,.22,.22,.22]);pdf.savefig(f);plt.close(f)

 f=fig("Compute analysis, hypotheses, and next step","Parameter sharing is not computation sharing in this architecture.")
 text(f,.065,.82,"FLOPs and latency",13,"bold",NAVY);text(f,.065,.785,"Theoretical attention FLOPs remain approximately unchanged: all three encoder self-attention operations still execute with\nthe same dimensions and sequence length. C4 removes duplicated parameter storage, not the three matrix operations.",10.3)
 text(f,.065,.695,f"Actual mean CPU latency was {B['latency']['mean_seconds']*1000:.2f} ms for B and {C['latency']['mean_seconds']*1000:.2f} ms for C4 ({pct(C['latency']['mean_seconds'],B['latency']['mean_seconds']):+.1f}%). This is an empirical runtime measurement, not evidence of reduced FLOPs.",10.3,"bold",TEAL)
 text(f,.065,.59,"Why C4 may help — hypotheses, not established causes",13,"bold",NAVY);text(f,.065,.553,"• Tying may regularize a small training problem by limiting redundant depth-specific attention parameters.\n• Similar encoder transformations may be sufficient across three depths on this tiny split.\n• The apparent gain may partly reflect split-specific variance or reduced overfitting after one epoch.",10.2)
 text(f,.065,.42,"Recommendation",13,"bold",NAVY);text(f,.065,.382,"Keep C4 as the next candidate for a larger controlled experiment. Do not select it as the final architecture yet.\nNext: repeat over more held-out video-level samples and seeds, retain the same RAFT/Farneback decision explicitly, and compare confidence intervals.",10.5,"bold",TEAL)
 text(f,.065,.26,"Not tested here: full encoder+decoder sharing and decoder cross-attention sharing are intentionally excluded from C4.\nNo claim is made about published MTN/IDD-PeD benchmark performance.",9.6,"normal",RED);pdf.savefig(f);plt.close(f)
print(OUT)
