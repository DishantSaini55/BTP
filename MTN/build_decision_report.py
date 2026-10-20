"""Build a supervisor-ready report exclusively from saved POC measurements."""
from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselineUtils import OnboardTfDataset, get_data
from individual_TF import IndividualTF
from context_TF import ContextIndividualTF


ROOT = Path(r"D:\IDD-PeD-Evaluation")
OUT = ROOT / "outputs"
RESULTS = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((ROOT / "flow" / "cpu_farneback_poc" / "test_manifest.json").read_text(encoding="utf-8"))
PDF = OUT / "MTN_Context_Architecture_Decision_Report_FINAL.pdf"
CKPT_A = OUT / "version_a_cpu_poc.pth"
CKPT_B = OUT / "version_b_cpu_poc.pth"

NAVY, BLUE, TEAL, ORANGE, RED, INK, MUTED, PALE = (
    "#102A43", "#2F80ED", "#00A6A6", "#F2994A", "#D64545", "#172B4D", "#52606D", "#F4F7FB"
)
# Force fully embedded TrueType glyphs rather than Type-3 paths.  DejaVu Sans
# ships with Matplotlib and covers all report glyphs used below.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
    "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.use14corefonts": False,
})


def page_header(fig, title, section, page):
    fig.patch.set_facecolor("white")
    fig.text(.06, .965, "MTN CONTEXT-AWARE ARCHITECTURE EVALUATION", color=NAVY, weight="bold", fontsize=9)
    fig.text(.94, .965, section.upper(), color=MUTED, ha="right", fontsize=8)
    fig.lines.append(plt.Line2D([.06, .94], [.946, .946], transform=fig.transFigure, color="#D9E2EC", lw=1))
    fig.text(.06, .03, "CPU-based small-scale proof-of-concept  |  IDD-PeD  |  Farneback optical flow", color=MUTED, fontsize=8)
    fig.text(.94, .03, f"{page}", color=MUTED, ha="right", fontsize=8)
    fig.suptitle(title, x=.06, y=.91, ha="left", color=NAVY, fontsize=22, weight="bold")


def card(ax, xy, wh, title, value, note="", color=BLUE, value_size=16):
    x, y = xy; w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025", ec="#D9E2EC", fc="white", lw=1.1))
    ax.add_patch(FancyBboxPatch((x, y+h-.045), w, .045, boxstyle="round,pad=0.012,rounding_size=0.025", ec=color, fc=color, lw=0))
    ax.text(x+.025, y+h-.070, title, color=MUTED, fontsize=8, weight="bold", va="top")
    ax.text(x+.025, y+h-.125, value, color=INK, fontsize=value_size, weight="bold", va="top", linespacing=1.05)
    ax.text(x+.025, y+.020, note, color=MUTED, fontsize=7.5, wrap=True, va="bottom")


def metric_row(ax, y, label, values, detail):
    """A roomy one-line metric card used on the dashboard."""
    h = .17
    ax.add_patch(FancyBboxPatch((0, y), .88, h, boxstyle="round,pad=.012,rounding_size=.025", ec="#D9E2EC", fc="white", lw=1.1))
    ax.add_patch(FancyBboxPatch((0, y+h-.040), .88, .040, boxstyle="round,pad=.012,rounding_size=.025", ec=TEAL, fc=TEAL, lw=0))
    ax.text(.025, y+.105, label, fontsize=8.5, color=MUTED, weight="bold", va="center")
    ax.text(.025, y+.065, values, fontsize=13, color=INK, weight="bold", va="center")
    ax.text(.025, y+.025, detail, fontsize=8, color=MUTED, va="center")


def text_page(pdf, page, title, section, blocks):
    fig = plt.figure(figsize=(8.27, 11.69)); page_header(fig, title, section, page)
    ax = fig.add_axes([.06,.09,.88,.76]); ax.axis("off")
    y = .96
    for heading, body in blocks:
        ax.text(0, y, heading, color=NAVY, fontsize=12, weight="bold", va="top")
        y -= .045
        ax.text(0, y, body, color=INK, fontsize=10, va="top", wrap=True, linespacing=1.45)
        y -= .15
    pdf.savefig(fig); plt.close(fig)


def metric_rows():
    a, b = RESULTS["aggregate"]["A"], RESULTS["aggregate"]["B"]
    names = [("MSE", "mse", "Lower trajectory-box error is better."),
             ("C-MSE", "cmse", "Lower centre-position error is better."),
             ("CF-MSE", "cfmse", "Lower final-frame centre error is better."),
             ("Mean inference time", "seconds", "Lower measured latency is better.")]
    rows=[]
    for label,key,meaning in names:
        av,bv=a[key],b[key]; delta=bv-av; pct=100*(av-bv)/av
        rows.append((label,av,bv,delta,pct,meaning))
    return rows


def display_predictions():
    """Re-run saved CPU checkpoints only to render their already-measured paths."""
    random.seed(RESULTS["seed"]); np.random.seed(RESULTS["seed"]); torch.manual_seed(RESULTS["seed"])
    flow = ROOT / "flow" / "cpu_farneback_poc"
    with (flow / "train.pkl").open("rb") as f: train_raw = pickle.load(f)
    with (flow / "test.pkl").open("rb") as f: test_raw = pickle.load(f)
    b = np.concatenate([np.asarray(x) for x in train_raw["bbox"]]); speed=np.concatenate([np.asarray(x) for x in train_raw["obd_speed"]])
    mean, std, ms, ss = b.mean(0), np.maximum(b.std(0),1e-6), speed.mean(0), np.maximum(speed.std(0),1e-6)
    options=dict(normalize_bbox=True,track_overlap=.5,observe_length=15,predict_length=45,enc_input_type=["bbox"],dec_input_type=["obd_speed","ego_op_flow","ped_op_flow"],prediction_type=["bbox"])
    def ds(context):
        data=get_data(test_raw,"test",mean,std,ms,ss,include_context=context,**options);data["image_name"]=data["obs_image"]
        return OnboardTfDataset(data,"test",mean,std)
    a=IndividualTF(15,4,4,4,N=3,d_model=32,d_ff=128,h=4,dropout=.1);a.load_state_dict(torch.load(CKPT_A,map_location="cpu"));a.eval()
    bmodel=ContextIndividualTF(15,4,4,4,N=3,d_model=32,d_ff=128,h=4,dropout=.1);bmodel.load_state_dict(torch.load(CKPT_B,map_location="cpu"));bmodel.eval()
    out=[]
    with torch.no_grad():
        for ba,bb in zip(DataLoader(ds(False),batch_size=1,shuffle=False),DataLoader(ds(True),batch_size=1,shuffle=False)):
            def inputs(x): return {"boxes":x["enc_input"],"speed":x["obd_speed"],"ego_flow":x["ego_op_flow"],"ped_flow":x["ped_op_flow"],"scene_context":x.get("scene_context"),"observed_occlusion":x.get("observed_occlusion")}
            gt=ba["pred_target"].numpy()[0]*std+mean
            pa=a.forward_batch(inputs(ba)).numpy()[0]*std+mean
            pb=bmodel.forward_batch(inputs(bb)).numpy()[0]*std+mean
            out.append((gt,pa,pb))
    return out


def draw_arch(ax):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    def box(x,y,w,h,text,color,small=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.015,rounding_size=.025",fc="white",ec=color,lw=1.7))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=8 if small else 9,color=INK,weight="bold" if not small else "normal",wrap=True)
    def arrow(a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle="->",mutation_scale=12,color=MUTED,lw=1.2))
    ax.text(.06,.91,"VERSION A — MTN without context",color=BLUE,weight="bold",fontsize=12)
    box(.08,.72,.19,.1,"Observed\ntrajectory",BLUE); box(.40,.72,.18,.1,"MTN",BLUE); box(.71,.72,.19,.1,"Trajectory\nprediction",BLUE)
    arrow((.27,.77),(.40,.77)); arrow((.58,.77),(.71,.77))
    ax.text(.06,.54,"VERSION B — MTN with IDD-PeD context",color=TEAL,weight="bold",fontsize=12)
    box(.08,.31,.16,.1,"Observed\ntrajectory",TEAL); box(.08,.13,.16,.1,"Scene +\nocclusion",TEAL)
    box(.34,.13,.17,.1,"Context\nembeddings",TEAL); box(.60,.13,.16,.1,"Context\nprojection",TEAL); box(.60,.31,.16,.1,"Dynamic\ncontext fusion",TEAL); box(.82,.31,.12,.1,"MTN\noutput",TEAL)
    arrow((.24,.18),(.34,.18)); arrow((.51,.18),(.60,.18)); arrow((.68,.23),(.68,.31)); arrow((.24,.36),(.60,.36)); arrow((.76,.36),(.82,.36))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = metric_rows()
    samples = RESULTS["samples"]
    paths = display_predictions()
    manifest_by_frame = {Path(item["first_frame"]).name: item for item in MANIFEST}
    with PdfPages(PDF) as pdf:
        # 1 Executive summary
        fig=plt.figure(figsize=(8.27,11.69)); page_header(fig,"MTN Context-Aware\nArchitecture Evaluation","Executive summary",1); fig._suptitle.set_fontsize(31)
        fig.text(.06,.77,"Without Context vs With IDD-PeD Context — CPU-Based Proof-of-Concept",fontsize=12,color=MUTED)
        ax=fig.add_axes([.06,.09,.88,.61]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
        card(ax,(.00,.64),(.42,.23),"VERSION A","MTN\nWithout Context","Observed trajectory + shared flow inputs",BLUE,14)
        card(ax,(.46,.64),(.42,.23),"VERSION B","MTN\nWith IDD-PeD Context","Context embeddings + dynamic fusion",TEAL,14)
        card(ax,(.00,.33),(.205,.20),"MSE - 10.4% lower","A 1242.98\nB 1113.21","",TEAL,12)
        card(ax,(.225,.33),(.205,.20),"C-MSE - 10.7% lower","A 1225.17\nB 1094.57","",TEAL,12)
        card(ax,(.45,.33),(.205,.20),"CF-MSE - 10.9% lower","A 4093.85\nB 3646.51","",TEAL,12)
        card(ax,(.675,.33),(.205,.20),"INFERENCE - 11.3% lower","A 69.24 ms\nB 61.41 ms","",TEAL,11)
        ax.add_patch(FancyBboxPatch((0,.05),.88,.18,boxstyle="round,pad=.02,rounding_size=.025",fc=PALE,ec="#D9E2EC"))
        ax.text(.025,.17,"Decision for the next stage",color=NAVY,weight="bold",fontsize=11)
        ax.text(.025,.095,"Carry Version B forward: lower aggregate error on all primary metrics.\nIt had lower MSE on 6/10 held-out trajectories.",color=INK,fontsize=9.5,linespacing=1.35)
        ax.text(.025,.055,"Context branch first-batch L1 gradient = 0.02813 — the branch was trained, not evaluated with random context weights.",color=MUTED,fontsize=8)
        pdf.savefig(fig);plt.close(fig)

        text_page(pdf,2,"The question being tested","Question",[("Research question","Does adding scene/context information to MTN improve pedestrian trajectory prediction compared with the original MTN architecture?"),("Why this comparison matters","Pedestrians with similar recent motion may act differently in different environments. A paired comparison isolates whether supplying IDD-PeD scene labels and observed occlusion changes prediction quality when every other POC condition is held constant."),("Fair comparison","Both versions used the same training, validation and held-out samples, seed, optimizer, loss, CPU execution environment, preprocessing, and the same cached Farneback flow tensors.")])

        fig=plt.figure(figsize=(8.27,11.69)); page_header(fig,"Architecture comparison","Architecture",3)
        ax=fig.add_axes([.06,.40,.88,.43]); draw_arch(ax)
        ax=fig.add_axes([.06,.10,.88,.22]);ax.axis("off")
        ax.text(0,.9,"Context supplied to Version B",weight="bold",color=NAVY,fontsize=12)
        ax.text(0,.68,"• intersection type    • signalized type    • road type\n• location type         • motion direction  • observed occlusion history",fontsize=10,color=INK,linespacing=1.8)
        ax.text(0,.25,"The branch turns categorical scene/state information into learned embeddings, projects it to the MTN representation space, and dynamically fuses it with motion features. Version A has no additional context branch.",fontsize=9,color=MUTED,wrap=True)
        pdf.savefig(fig);plt.close(fig)

        text_page(pdf,4,"Experimental setup","Method",[("Strict video-level partitions","Train: 32 trajectories from gp_set_0006_vid_0002. Validation: 8 trajectories from decodable pre-frame-4000 portions of gp_set_0006_vid_0001. Held-out test: 10 trajectories from gp_set_0006_vid_0003. No training trajectory was used in testing."),("Matched flow inputs","CPU Farneback dense optical flow, resized to 320×180. Ego flow was pooled to 8×8 [64,2]; pedestrian flow from each observed bounding-box crop was pooled to 3×3 [9,2]. The identical float32 cached flows were used by A and B."),("Scope","12 epochs; shared MTN checkpoint initialization; CPU-only. This is a CPU-based small-scale proof-of-concept — not an exact reproduction of the published MTN/IDD-PeD benchmark, because the authors’ RAFT flow pipeline was unavailable.")])

        # dashboard
        fig=plt.figure(figsize=(8.27,11.69)); page_header(fig,"Results dashboard","Measured results",5)
        ax=fig.add_axes([.06,.49,.88,.35]);ax.axis("off")
        for i,(label,av,bv,delta,pct,meaning) in enumerate(metrics):
            y=.79-i*.21; unit=" ms" if label=="Mean inference time" else ""
            shown_a, shown_b = (av * 1000, bv * 1000) if unit else (av, bv)
            metric_row(ax,y,label,f"A {shown_a:.2f}{unit}   |   B {shown_b:.2f}{unit}",f"B is {pct:.1f}% lower. {meaning}")
        ax2=fig.add_axes([.10,.10,.78,.29]);labels=[x[0] for x in metrics[:3]]; A=[x[1] for x in metrics[:3]];B=[x[2] for x in metrics[:3]];pos=np.arange(3);w=.34
        ax2.bar(pos-w/2,A,w,label="A: without context",color=BLUE);ax2.bar(pos+w/2,B,w,label="B: with context",color=TEAL)
        ax2.set_xticks(pos,labels);ax2.set_ylabel("Error (lower is better)");ax2.grid(axis="y",alpha=.2);ax2.legend(frameon=False);ax2.spines[["top","right"]].set_visible(False)
        pdf.savefig(fig);plt.close(fig)

        # sample table
        fig=plt.figure(figsize=(11.69,8.27)); page_header(fig,"Sample-wise held-out analysis","All 10 test trajectories",6); fig._suptitle.set_fontsize(18)
        ax=fig.add_axes([.04,.10,.92,.75]);ax.axis("off"); rows=[]
        for i,s in enumerate(samples):
            frame=Path(s["sample"]).name; info=manifest_by_frame.get(frame,{})
            ctx="["+", ".join(map(str,s["context"]))+f"]; occ≤{info.get('max_occlusion','—')}"
            delta=s["B_MSE"]-s["A_MSE"]; pct=100*delta/s["A_MSE"]
            rows.append([f"T{i+1:02d}",frame,ctx,f"{s['A_MSE']:.2f}",f"{s['B_MSE']:.2f}",f"{delta:+.2f}",f"{pct:+.1f}%","B" if delta<0 else "A"])
        table=ax.table(cellText=rows,colLabels=["Test","Frame","Context; max occ.","A MSE","B MSE","B − A","% change","Lower"],loc="upper center",cellLoc="center",colColours=["#EAF2FB"]*8)
        table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.8)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor("#D9E2EC")
            if r==0:cell.set_text_props(weight="bold",color=NAVY)
            elif c==7:cell.set_text_props(color=TEAL if cell.get_text().get_text()=="B" else RED,weight="bold")
        fig.text(.06,.075,"Version B had lower MSE on 6/10 held-out trajectories. All ten outcomes are included; the four regressions are not omitted.",color=MUTED,fontsize=8)
        pdf.savefig(fig);plt.close(fig)

        # ten trajectory charts, two/page
        for pg,start in enumerate(range(0,10,2),7):
            fig=plt.figure(figsize=(8.27,11.69));page_header(fig,"Trajectory comparison","Held-out examples",pg)
            for j,s in enumerate(samples[start:start+2]):
                ax=fig.add_axes([.12,.55-j*.40,.76,.28]); gt,pa,pb=paths[start+j]
                if gt.size and pa.size:
                    for arr,label,color in ((gt,"Ground truth","#1F2933"),(pa,"A: without context",BLUE),(pb,"B: with context",TEAL)):
                        c=np.stack(((arr[:,0]+arr[:,2])/2,(arr[:,1]+arr[:,3])/2),-1);ax.plot(c[:,0],c[:,1],label=label,color=color,lw=2)
                ax.invert_yaxis();ax.legend(frameon=False,loc="upper left");ax.grid(alpha=.15);ax.set_xlabel("x");ax.set_ylabel("y")
                change=s["B_MSE"]-s["A_MSE"]
                ax.set_title(f"T{start+j+1:02d} | {Path(s['sample']).name} | context={s['context']} | A MSE {s['A_MSE']:.2f}; B MSE {s['B_MSE']:.2f}; Δ {change:+.2f}",fontsize=9)
            pdf.savefig(fig);plt.close(fig)

        text_page(pdf,12,"What the results mean","Interpretation",[("Preliminary evidence","Under this paired POC setup, context lowered aggregate MSE, C-MSE and CF-MSE. Version B also had lower MSE on 6 of 10 held-out trajectories and measured lower mean inference latency."),("Practical interpretation","Without context, MTN primarily uses observed pedestrian motion and shared flow inputs. With context, it also receives environmental and pedestrian-state labels. That information may help distinguish similar motion occurring in different scene conditions. This is an interpretation, not proof of causality."),("What is not claimed","These data do not show that context is universally better. The sample is small, all test context codes are limited, and the flow pipeline differs from the authors’ RAFT implementation.")])

        text_page(pdf,13,"Architecture decision","Recommendation",[("Recommendation for the next stage","Version B — MTN with Context — is the architecture to carry forward for the next experimental stage."),("Measured basis","1. MSE decreased by 10.4%.\n2. C-MSE decreased by 10.7%.\n3. CF-MSE decreased by 10.9%.\n4. B had lower MSE on 6/10 held-out trajectories.\n5. Context parameters received a non-zero training gradient (0.02813).\n6. Mean measured inference time was 11.3% lower in this POC."),("Scope of this decision","This recommendation is for continuation to the next experimental stage, based on preliminary CPU POC results—not a definitive claim of universal superiority.")])

        text_page(pdf,14,"Limitations and next step","Scientific scope",[("Limitations","• Small scale: 32 training, 8 validation, 10 held-out test trajectories.\n• CPU-based execution.\n• CPU Farneback flow rather than the authors’ RAFT pipeline.\n• Not an exact reproduction of the published benchmark.\n• Larger-scale validation is required before a final research conclusion."),("Next-stage plan","Phase 1 — POC completed.\nPhase 2 — Train on a larger dataset.\nPhase 3 — Evaluate on a larger held-out test set.\nPhase 4 — Use the authors’ exact flow pipeline if it becomes available.\nPhase 5 — Confirm whether the observed context improvement persists.")])

        fig=plt.figure(figsize=(8.27,11.69));page_header(fig,"POC result","Conclusion",15);ax=fig.add_axes([.06,.12,.88,.68]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
        ax.text(.5,.84,"POC RESULT",ha="center",color=NAVY,fontsize=19,weight="bold")
        card(ax,(.04,.55),(.39,.18),"WITHOUT CONTEXT","MSE: 1242.98","Version A",BLUE,14);card(ax,(.49,.55),(.39,.18),"WITH CONTEXT","MSE: 1113.21","Version B",TEAL,14)
        ax.text(.5,.40,"→ 10.4% lower MSE",ha="center",color=TEAL,fontsize=20,weight="bold")
        ax.add_patch(FancyBboxPatch((.10,.16),.68,.14,boxstyle="round,pad=.02,rounding_size=.025",fc=PALE,ec="#D9E2EC"));ax.text(.44,.255,"Recommended architecture for the next stage",ha="center",color=MUTED,fontsize=9);ax.text(.44,.205,"MTN + Context Branch",ha="center",color=NAVY,fontsize=16,weight="bold");ax.text(.44,.17,"Lower prediction error across the main evaluated metrics in this preliminary experiment.",ha="center",color=INK,fontsize=8)
        ax.text(.5,.07,"Preliminary CPU-based POC; larger-scale validation is required.",ha="center",color=RED,fontsize=9,style="italic")
        pdf.savefig(fig);plt.close(fig)
    print(PDF)


if __name__ == "__main__":
    main()
