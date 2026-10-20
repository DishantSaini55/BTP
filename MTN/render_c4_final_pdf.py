import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans'})
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
O=Path(r'D:\IDD-PeD-Evaluation\outputs\final_architecture\C4_Final_Architecture.pdf');N='#102A43';B='#2F80ED';T='#00A6A6';A='#F2994A'
def box(ax,x,y,w,h,s,c=B):
 ax.add_patch(plt.Rectangle((x,y),w,h,fc='#EEF5FF' if c==B else '#E8F8F5',ec=c,lw=1.6));ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=8.2,wrap=True)
def ar(ax,a,b,c=B,ls='-'):ax.annotate('',b,a,arrowprops=dict(arrowstyle='->',lw=1.5,color=c,linestyle=ls))
with PdfPages(O) as p:
 f,ax=plt.subplots(figsize=(16,9));ax.set_xlim(0,16);ax.set_ylim(0,9);ax.axis('off');f.text(.5,.95,'C4 Context-Aware MTN — Proposed POC Architecture',ha='center',fontsize=19,fontweight='bold',color=N)
 box(ax,.3,6.8,1.8,.55,'Observed pedestrian trajectory\n[B,15,4]');box(ax,2.4,6.8,2.1,.55,'Source embedding + positional encoding\n[B,15,32]');ar(ax,(2.1,7.08),(2.4,7.08));
 ax.text(5.95,8.05,'MTN Encoder ×3',ha='center',fontsize=10,fontweight='bold',color=N)
 for i,y in enumerate([7.1,6.1,5.1]):box(ax,4.8,y,2.3,.58,f'Encoder Layer {i+1}\nself_attn + src_attn + cross_attn + FFN');
 ar(ax,(4.5,7.08),(4.8,7.38));ar(ax,(5.95,7.1),(5.95,6.68));ar(ax,(5.95,6.1),(5.95,5.68));box(ax,7.7,6.0,2.2,.95,'C4 Weight Sharing\nSAME self-attn Q/K/V/O\nEncoder layers 0–2',T)
 for y in [7.38,6.38,5.38]:ar(ax,(7.7,6.48),(7.1,y),T,'--')
 box(ax,4.9,4.05,2.1,.55,'Trajectory memory\n[B,15,32]');ar(ax,(5.95,5.1),(5.95,4.6));
 ax.text(13.05,7.75,'Context Branch',ha='center',fontsize=10,fontweight='bold',color=T)
 box(ax,10.8,6.75,4.5,.65,'Six native IDD-PeD context inputs: intersection type, signalized type, road type, location type, motion direction, observed occlusion',T);box(ax,11.7,5.75,2.7,.55,'Six 16-D embeddings',T);box(ax,11.5,4.75,3.1,.6,'Concatenation [B,96]\nContext projection [B,32]',T);box(ax,11.35,3.65,3.4,.65,'Context admission gate\nGated context token [B,1,32]',A);ar(ax,(13.05,6.75),(13.05,6.3),T);ar(ax,(13.05,5.75),(13.05,5.35),T);ar(ax,(13.05,4.75),(13.05,4.3),A)
 box(ax,5.0,2.8,3.0,.6,'Append → augmented decoder memory [B,16,32]',T);ar(ax,(5.95,4.05),(6.5,3.4));ar(ax,(11.35,3.98),(8,3.1),T);box(ax,5.2,1.65,2.6,.65,'MTN Decoder ×3\nunchanged attention + FFN');box(ax,5.2,.55,2.6,.55,'Output head + residual\nFuture trajectory [B,45,4]');ar(ax,(6.5,2.8),(6.5,2.3));ar(ax,(6.5,1.65),(6.5,1.1))
 box(ax,.5,3.0,3.4,1.3,'C4 Weight Sharing\nOnly encoder self-attention Q/K/V/O is shared. FFNs, normalization, src_attn, cross_attn, decoder, context branch and output head remain independent.',T);box(ax,9.0,.55,6.2,1.45,'Current POC Evidence\nParameters: −6.07% vs B  |  MSE: −5.01% vs B\nC-MSE: −4.84% vs B  |  CF-MSE: −2.82% vs B\nCPU latency: +3.93% vs B  |  FLOPs: not reduced',A)
 f.text(.5,.015,'Solid arrows: data flow. Dashed teal arrows: shared parameters only. Results are CPU Farneback POC: one held-out video, 24 trajectories, one epoch, three seeds; not an official RAFT-based IDD-PeD benchmark.',ha='center',fontsize=7.5);p.savefig(f,bbox_inches='tight');plt.close(f)
 f=plt.figure(figsize=(16,9));f.text(.06,.93,'Module-wise Architecture Description',fontsize=20,fontweight='bold',color=N);txt='''MTN trajectory encoder — Observed boxes [B,15,4] are embedded with positional encoding to [B,15,32]. Three EncoderLayer blocks produce trajectory memory [B,15,32].

C4 weight sharing — Only self_attn Q/K/V/O projections are aliases across Encoder.layers[0..2]. The three layers still execute; src_attn, cross_attn, FFNs, norms, decoder and output head remain independent.

Context branch — intersection type, signalized type, road type, location type, motion direction and observed occlusion become six 16-D embeddings. They concatenate to [B,96] and project to [B,32].

Context gate — The original B admission gate determines the gated context token [B,1,32], which is appended after the encoder to form augmented decoder memory [B,16,32].

Decoder and output — The unchanged MTN Decoder ×3 consumes augmented memory with the original flow paths. Linear(32,4) plus the final observed-box residual predicts [B,45,4].

Interpretation — C4 reduces parameter/checkpoint storage, not FLOPs. Measured CPU latency was higher than B in the unified POC.''';f.text(.08,.82,txt,fontsize=12,va='top',wrap=True);p.savefig(f,bbox_inches='tight');plt.close(f)
