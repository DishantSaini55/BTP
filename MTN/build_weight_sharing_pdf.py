"""Create a concise PDF from the measured weight-sharing audit results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT=Path(r"D:\IDD-PeD-Evaluation");OUT=ROOT/"outputs";R=json.loads((OUT/"weight_sharing_results.json").read_text())
PDF=OUT/"Weight_Sharing_Architecture_Audit_Report.pdf"
NAVY="#102A43";BLUE="#2F80ED";TEAL="#00A6A6";RED="#D64545";MUTED="#52606D";PALE="#F4F7FB"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"ps.fonttype":42,"pdf.use14corefonts":False})
def header(fig,title,page):
 fig.text(.06,.965,"MTN CONTEXT-AWARE ARCHITECTURE EVALUATION",fontsize=9,color=NAVY,weight="bold");fig.text(.94,.965,"WEIGHT-SHARING AUDIT",fontsize=8,color=MUTED,ha="right");fig.lines.append(plt.Line2D([.06,.94],[.946,.946],transform=fig.transFigure,color="#D9E2EC"));fig.suptitle(title,x=.06,y=.91,ha="left",fontsize=22,color=NAVY,weight="bold");fig.text(.06,.03,"CPU-based small-scale proof-of-concept  |  same cached Farneback flows and 32/8/10 split",fontsize=8,color=MUTED);fig.text(.94,.03,str(page),ha="right",fontsize=8,color=MUTED)
def card(ax,x,y,w,h,title,value,note,color,value_size=17):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012,rounding_size=.025",fc="white",ec="#D9E2EC"));ax.add_patch(FancyBboxPatch((x,y+h-.04),w,.04,boxstyle="round,pad=.012,rounding_size=.025",fc=color,ec=color));ax.text(x+.025,y+h-.07,title,fontsize=8,weight="bold",color=MUTED,va="top");ax.text(x+.025,y+.058,value,fontsize=value_size,weight="bold",color=NAVY,va="bottom");ax.text(x+.025,y+.018,note,fontsize=8,color=MUTED,va="bottom")
def para(ax,y,head,body):
 ax.text(0,y,head,fontsize=12,color=NAVY,weight="bold",va="top");ax.text(0,y-.05,body,fontsize=10,color="#243B53",va="top",wrap=True,linespacing=1.4)
def main():
 b,c=R["B"],R["C"];lat=R["latency_benchmark"]
 with PdfPages(PDF) as pdf:
  fig=plt.figure(figsize=(8.27,11.69));header(fig,"Weight-Sharing Architecture Audit",1);fig.text(.06,.77,"Current context-aware MTN (Version B) vs full encoder/decoder layer sharing (Version C)",fontsize=11,color=MUTED)
  ax=fig.add_axes([.06,.12,.88,.57]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
  card(ax,.00,.66,.42,.20,"VERSION B","139,142 parameters","Current context-aware MTN",BLUE);card(ax,.46,.66,.42,.20,"VERSION C","53,894 parameters","Encoder + decoder layer tying",TEAL)
  card(ax,.00,.37,.27,.18,"PARAMETER REDUCTION","61.3%","85,248 fewer unique parameters",TEAL,14);card(ax,.305,.37,.27,.18,"HELD-OUT MSE","+20.3%","1113.21 -> 1339.31",RED,14);card(ax,.61,.37,.27,.18,"WARMED CPU LATENCY","+6.2%","47.20 ms -> 50.13 ms",RED,14)
  ax.add_patch(FancyBboxPatch((0,.05),.88,.20,boxstyle="round,pad=.02,rounding_size=.025",fc=PALE,ec="#D9E2EC"));ax.text(.025,.19,"Decision",fontsize=12,color=NAVY,weight="bold");ax.text(.025,.115,"Do not replace Version B with fully shared Version C for the next accuracy-focused stage.",fontsize=10,color=NAVY,weight="bold");ax.text(.025,.065,"Full sharing reduced parameters but increased held-out error and did not reduce measured inference latency.",fontsize=9,color=MUTED)
  pdf.savefig(fig);plt.close(fig)
  fig=plt.figure(figsize=(8.27,11.69));header(fig,"Current architecture and sharing options",2);ax=fig.add_axes([.06,.10,.88,.76]);ax.axis("off")
  para(ax,.96,"Current Version B","139,142 trainable parameters; three encoder layers and three decoder layers; 18 attention modules; d_model = 32, 4 heads, d_ff = 128. The context encoder plus gate contributes 5,442 parameters and remains unchanged in Version C.")
  rows=[["Current B","139,142","0","0.0%"],["Encoder layers tied","96,454","42,688","30.7%"],["Decoder layers tied","96,582","42,560","30.6%"],["Attention-only,\nsame-role tied","88,454","50,688","36.4%"],["All attention\nprojections tied","67,334","71,808","51.6%"],["C: full layer tying","53,894","85,248","61.3%"]]
  table=ax.table(cellText=rows,colLabels=["Candidate","Parameters","Reduction","Reduction %"],colColours=["#EAF2FB"]*4,cellLoc="center",loc="center",bbox=[0,.22,1,.47]);table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.6)
  for (r,col),cell in table.get_celld().items():
   cell.set_edgecolor("#D9E2EC")
   if r==0:cell.set_text_props(weight="bold",color=NAVY)
   if r==6:cell.set_text_props(weight="bold",color=TEAL)
  para(ax,.15,"Selected safe candidate","Version C reuses one encoder layer three times and one decoder layer three times. It retains the context labels, context token, gate, flow tensors, prediction target, and output interface. Attention-only or global encoder/decoder projection tying were not selected because they are more semantically aggressive.")
  pdf.savefig(fig);plt.close(fig)
  fig=plt.figure(figsize=(8.27,11.69));header(fig,"Measured B-vs-C comparison",3);ax=fig.add_axes([.06,.12,.88,.70]);ax.axis("off")
  rows=[["Parameters","139,142","53,894","-61.3%"],["Checkpoint size","1,941,515 bytes","1,541,251 bytes","-20.6%"],["MSE","1113.21","1339.31","+20.3%"],["C-MSE","1094.57","1326.93","+21.2%"],["CF-MSE","3646.51","4530.43","+24.2%"],["Warmed latency","47.20 ms","50.13 ms","+6.2%"],["Latency variability","4.98 ms SD","8.04 ms SD","200 forwards/model"]]
  table=ax.table(cellText=rows,colLabels=["Metric","B: current context MTN","C: layer-tied MTN","Change"],colColours=["#EAF2FB"]*4,cellLoc="center",loc="upper center",bbox=[0,.43,1,.42]);table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.8)
  for (r,col),cell in table.get_celld().items():
   cell.set_edgecolor("#D9E2EC")
   if r==0:cell.set_text_props(weight="bold",color=NAVY)
   elif r in (3,4,5,6):cell.set_text_props(color=RED if col==3 else "#243B53")
  para(ax,.32,"Latency protocol","Five complete warm-up passes preceded timing. Each saved model then ran 200 forward passes over the identical held-out inputs. The +6.2% value is the observed CPU POC result, not a claim that sharing is generally slower.")
  para(ax,.14,"Accuracy result","C had lower MSE on 5/10 held-out trajectories, but its aggregate held-out MSE, C-MSE, and CF-MSE were all higher. The C context branch was trained: first-batch L1 gradient = 1.45948.")
  pdf.savefig(fig);plt.close(fig)
  fig=plt.figure(figsize=(8.27,11.69));header(fig,"Computation impact and recommendation",4);ax=fig.add_axes([.06,.12,.88,.70]);ax.axis("off")
  para(ax,.95,"Parameter reduction is not FLOP reduction","Weight sharing reduces unique parameters and optimizer state. Version C still performs three encoder and three decoder executions, 18 attention operations, and 9 FFN calls per forward—the same execution count as Version B. Inference FLOPs therefore remain approximately unchanged.")
  para(ax,.68,"Why the context branch was kept independent","The context encoder forms a scene token using label embeddings and context projections. It does not add an attention module. Its projections are not semantically interchangeable with MTN Q/K/V/O projections, so tying them would unnecessarily change the context mechanism.")
  para(ax,.42,"Recommendation","Keep Version B as the accuracy reference. Full layer tying is appropriate only if the 61.3% unique-weight reduction is worth the measured accuracy regression. The next safer footprint experiments are encoder-only or decoder-only tying (about 30.6% reduction each).")
  ax.add_patch(FancyBboxPatch((0,.07),.88,.19,boxstyle="round,pad=.02,rounding_size=.025",fc=PALE,ec="#D9E2EC"));ax.text(.03,.20,"Bottom line",fontsize=12,color=NAVY,weight="bold");ax.text(.03,.145,"Full sharing reduces parameters by 61.3%, while inference\ncomputation remains approximately unchanged.",fontsize=9.5,color=NAVY,weight="bold",linespacing=1.35);ax.text(.03,.085,"In this matched POC, latency changed by +6.2% and held-out prediction error regressed.",fontsize=9,color=MUTED)
  pdf.savefig(fig);plt.close(fig)
 print(PDF)
if __name__=="__main__":main()
