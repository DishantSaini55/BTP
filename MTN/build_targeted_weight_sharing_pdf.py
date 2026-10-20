"""Create the supervisor-ready PDF from measured targeted-ablation results."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(r"D:\IDD-PeD-Evaluation\outputs\weight_sharing_targeted")
DATA = json.loads((ROOT / "targeted_weight_sharing_results.json").read_text(encoding="utf8"))
OUT = ROOT / "Weight_Sharing_Targeted_Ablation_Report.pdf"
NAVY, BLUE, TEAL, ORANGE, RED, PALE = "#102A43", "#2F80ED", "#00A6A6", "#F2994A", "#D64545", "#EAF2F8"

def page(title, subtitle=""):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.text(.065, .93, title, fontsize=22, fontweight="bold", color=NAVY)
    fig.text(.065, .892, subtitle, fontsize=9.5, color="#52616B")
    fig.add_artist(plt.Line2D([.065,.935],[.875,.875], color=TEAL, lw=2))
    return fig
def txt(fig, x, y, s, size=10, weight="normal", color="#263238", **kw):
    fig.text(x,y,s,fontsize=size,fontweight=weight,color=color,va="top",**kw)
def table(ax, cols, rows, widths=None, fs=8.2):
    ax.axis("off")
    t=ax.table(cellText=rows,colLabels=cols,loc="center",cellLoc="left",colLoc="left",colWidths=widths)
    t.auto_set_font_size(False);t.set_fontsize(fs);t.scale(1,1.5)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor("#D9E2EC")
        if r==0: cell.set_facecolor(NAVY);cell.get_text().set_color("white");cell.get_text().set_weight("bold")
        elif r%2==0: cell.set_facecolor("#F7FAFC")
    return t
def pct(a,b): return 100*(a-b)/b

b=DATA["B"]
variants=[("C1", "decoder_src_attn"),("C2a", "decoder_src_qkv"),("C2b", "decoder_src_kv"),("C4", "encoder_self_attn")]
allrows=[("B",b["strategy"],b)]+[(tag,DATA["variants"][key]["shared_at"],DATA["variants"][key]) for tag,key in variants]

with PdfPages(OUT) as pdf:
    fig=page("Targeted Weight-Sharing Ablation", "Improved MTN Version B | CPU-based small-scale proof-of-concept | measured results only")
    txt(fig,.065,.82,"Decision question",13,"bold",NAVY); txt(fig,.065,.783,"Can attention parameters be shared across depth without tying the whole encoder and decoder?",11)
    txt(fig,.065,.705,"Scope",13,"bold",NAVY)
    txt(fig,.065,.668,"This report replaces no prior result. It evaluates targeted attention-only tying; the earlier full encoder+decoder\nsharing audit remains a reference-only result and is not proposed here.",10.5)
    txt(fig,.065,.57,"Measured protocol",13,"bold",NAVY)
    txt(fig,.065,.535,"• Version B context architecture unchanged\n• Fixed 32 train / 8 validation / 10 held-out test trajectories; cached identical CPU Farneback flow\n• All five models independently initialized from the official MTN checkpoint, with the same seed, Adam (1e-3), batch 8, MSE, and one epoch\n• CPU latency: 5 warm-up passes then 200 measured single-sample forwards",10.3)
    txt(fig,.065,.365,"Headline result",13,"bold",NAVY)
    best=DATA["variants"]["encoder_self_attn"]
    txt(fig,.065,.327,f"C4 (encoder self-attention sharing) achieved {best['metrics']['mse']:.2f} MSE vs. {b['metrics']['mse']:.2f} for B\n({pct(best['metrics']['mse'],b['metrics']['mse']):+.1f}%) with {pct(best['parameters'],b['parameters']):+.1f}% parameters.",16,"bold",TEAL)
    txt(fig,.065,.19,"Interpretation: promising on this POC only; it is not a published-benchmark reproduction (CPU Farneback flow, tiny split, one epoch).",9.5,"normal","#52616B")
    pdf.savefig(fig);plt.close(fig)

    fig=page("Architecture audit: what is actually shared", "Source of truth: MTN/transformer/multihead_attention.py, encoder.py, decoder.py, individual_TF.py")
    txt(fig,.065,.825,"Current Version B",13,"bold",NAVY)
    txt(fig,.065,.79,"3 encoder layers + 3 decoder layers; d_model = 32; 4 heads; d_ff = 128; 18 attention modules/calls per forward.\nContext branch is unchanged in every targeted variant.",10.2)
    rows=[
      ["C1", "Decoder src_attn", "Q + K + V + O", "2nd/3rd decoder depth alias 1st", "Yes"],
      ["C2a", "Decoder src_attn", "Q + K + V", "O remains depth-specific", "Yes"],
      ["C2b", "Decoder src_attn", "K + V", "Q and O remain depth-specific", "Yes"],
      ["C4", "Encoder self_attn", "Q + K + V + O", "2nd/3rd encoder depth alias 1st", "Yes"],
      ["C3", "Decoder self_attn", "—", "No such module exists", "N/A"],
      ["C5", "Individual heads", "—", "Requires projection redesign", "Not run"],]
    ax=fig.add_axes([.065,.37,.87,.33]);table(ax,["ID","Shared at","Shared projections","What remains independent","Run?"],rows,[.07,.20,.18,.37,.1],8.6)
    txt(fig,.065,.29,"Why decoder src_attn?",13,"bold",NAVY)
    txt(fig,.065,.253,"It is the decoder query → encoder-memory cross-attention path. Its memory has the same embedding width (32) at each depth;\nadding the context token changes sequence length, not d_model, so the selected sharing is shape-valid.",10)
    txt(fig,.065,.15,"Per attention module: Q, K, V, O are each Linear(32,32) + bias = 1,056 parameters; full module = 4,224 parameters.",9.5,"normal","#52616B")
    pdf.savefig(fig);plt.close(fig)

    fig=page("Measured aggregate results", "Lower MSE / C-MSE / CF-MSE and latency are better. Percentages are versus matched B.")
    rows=[]
    for tag,label,v in allrows:
        m=v["metrics"];lat=v["latency"]["mean_seconds"]*1000
        rows.append([tag,label.replace("Decoder source cross-attention: ","Dec. src: ").replace("Encoder self-attention: ","Enc. self: "),f"{v['parameters']:,}",f"{pct(v['parameters'],b['parameters']):+.1f}%",f"{m['mse']:.2f}",f"{pct(m['mse'],b['metrics']['mse']):+.1f}%",f"{m['cmse']:.2f}",f"{m['cfmse']:.2f}",f"{lat:.2f}"])
    ax=fig.add_axes([.045,.49,.91,.27]);table(ax,["ID","Shared at","Params","Δ params","MSE","Δ MSE","C-MSE","CF-MSE","Latency ms"],rows,[.05,.28,.09,.08,.09,.08,.09,.09,.1],7.8)
    labels=[x[0] for x in allrows];mses=[x[2]["metrics"]["mse"] for x in allrows];params=[x[2]["parameters"] for x in allrows]
    ax1=fig.add_axes([.09,.15,.36,.22]);ax1.bar(labels,mses,color=[BLUE,ORANGE,ORANGE,ORANGE,TEAL]);ax1.set_title("Test MSE",fontsize=10,color=NAVY);ax1.set_ylabel("MSE");ax1.grid(axis="y",alpha=.25)
    ax2=fig.add_axes([.56,.15,.36,.22]);ax2.bar(labels,params,color=[BLUE,ORANGE,ORANGE,ORANGE,TEAL]);ax2.set_title("Trainable parameters",fontsize=10,color=NAVY);ax2.set_ylabel("count");ax2.grid(axis="y",alpha=.25)
    pdf.savefig(fig);plt.close(fig)

    fig=page("Sample-wise differences", "Each point is one of the same 10 held-out trajectories. Lower is better.")
    ax=fig.add_axes([.09,.46,.84,.28]);x=np.arange(1,11);ax.plot(x,[r['mse'] for r in b['per_sample']],marker="o",lw=2,label="B — no sharing",color=BLUE)
    for tag,key,color in [("C1","decoder_src_attn",ORANGE),("C2a","decoder_src_qkv","#9B51E0"),("C2b","decoder_src_kv",RED),("C4","encoder_self_attn",TEAL)]: ax.plot(x,[r['mse'] for r in DATA['variants'][key]['per_sample']],marker="o",lw=1.5,label=tag,color=color)
    ax.set_xlabel("Held-out sample index");ax.set_ylabel("MSE");ax.grid(alpha=.25);ax.legend(ncol=3,fontsize=8);ax.set_xticks(x)
    counts=[]
    for tag,key in variants:
       arr=np.array([r['mse'] for r in DATA['variants'][key]['per_sample']]);base=np.array([r['mse'] for r in b['per_sample']]);counts.append([tag,int((arr<base).sum()),int((arr>base).sum())])
    ax2=fig.add_axes([.16,.17,.68,.16]);table(ax2,["Variant","Better than B samples","Worse than B samples"],counts,[.22,.3,.3],9)
    txt(fig,.065,.84,"No sample selection changed between B and the four targeted variants. Sample identities are retained in the JSON result artifact.",9.5,"normal","#52616B")
    pdf.savefig(fig);plt.close(fig)

    fig=page("Feasibility, compute, and recommendation", "Conclusions are restricted to the measured CPU proof-of-concept.")
    txt(fig,.065,.825,"C3 — decoder self-attention",12,"bold",NAVY);txt(fig,.065,.792,"Not applicable. The current DecoderLayer exposes src_attn, spd_attn, and ped_attn; it has no self_attn field. No result was fabricated.",9.8)
    txt(fig,.065,.715,"C5 — attention-head sharing",12,"bold",NAVY);txt(fig,.065,.682,"Not a clean aliasing operation in this implementation. The four 32×32 Q/K/V/O projections are learned as whole matrices, then reshaped into four heads.\nThere are no separate per-head Q/K/V tensors to tie. Head sharing would require a new block/grouped projection parameterization and a separate experiment.",9.7)
    txt(fig,.065,.57,"FLOPs",12,"bold",NAVY);txt(fig,.065,.537,"Approximately unchanged: all variants retain the same 18 attention calls and all depth executions. Tying removes stored parameters, not the repeated matrix operations.\nLatency differences are therefore empirical CPU implementation effects, not an expected compute reduction.",9.7)
    txt(fig,.065,.43,"Recommendation",12,"bold",NAVY)
    txt(fig,.065,.392,"Advance C4 (encoder self-attention Q/K/V/O sharing) to the next controlled experiment. It was the only tested targeted option to reduce both\nparameters (−6.1%) and measured MSE (−15.7%) versus the matched B baseline. Do not advance C1/C2a/C2b from this POC: all regressed MSE.",10.5,"bold",TEAL)
    txt(fig,.065,.29,"Do not use full encoder+decoder sharing as the proposed solution. Prior full-sharing figures (53,894 parameters; −61.3%; MSE +20.3%)\nremain reference-only and were generated under the earlier audit protocol, not this one-epoch matched ablation.",9.5,"normal","#52616B")
    txt(fig,.065,.18,"Limitations: CPU Farneback flow rather than authors' RAFT, 10 held-out trajectories, one epoch, and no full-dataset benchmark.\nThis establishes a direction for follow-up, not a publishable performance claim.",9.5,"normal",RED)
    pdf.savefig(fig);plt.close(fig)
print(OUT)
