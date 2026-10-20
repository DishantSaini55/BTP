from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
OUT=Path(r'D:\IDD-PeD-Evaluation\outputs\weight_sharing_targeted')
def draw():
 f,ax=plt.subplots(figsize=(18,11));ax.set_xlim(0,18);ax.set_ylim(0,11);ax.axis('off');f.patch.set_facecolor('white')
 ax.text(9,10.6,'Improved MTN Architecture — Version B vs C4',ha='center',fontsize=22,weight='bold',color='#102A43')
 def b(x,y,w,h,t,c='#2F80ED'):
  ax.add_patch(plt.Rectangle((x,y),w,h,fc='#EEF5FF' if c=='#2F80ED' else '#E8F8F5',ec=c,lw=1.8));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8.5,wrap=True)
 def ar(x1,y1,x2,y2,c='#2F80ED',ls='-'):ax.annotate('',(x2,y2),(x1,y1),arrowprops=dict(arrowstyle='->',color=c,lw=1.5,linestyle=ls))
 for x,title,c in [(0.4,'VERSION B — No Weight Sharing','#2F80ED'),(9.3,'VERSION C4 — Encoder Self-Attention Weight Sharing','#00A6A6')]: ax.add_patch(plt.Rectangle((x,.8),8.3,8.9,fc='#F8FBFF',ec=c,lw=2));ax.text(x+4.15,9.35,title,ha='center',weight='bold',fontsize=14,color=c)
 for off,c4 in [(0,False),(8.9,True)]:
  b(.8+off,8.2,1.2,.55,'Observed boxes\n15 × 4-D');b(2.4+off,8.2,1.8,.55,'MTN Encoder ×3\nsrc/cross/FFN unchanged');ar(2+off,8.48,2.4+off,8.48)
  for i,y in enumerate([7.8,6.8,5.8]): b(5+off,y,1.7,.62,f'Encoder Layer {i+1}\n'+('self-attn executes' if c4 else f'Self-Attention {chr(65+i)}\nown Q/K/V/O'));ar(4.2+off,8.48,5+off,8.1); 
  ar(5.85+off,7.8,5.85+off,7.42);ar(5.85+off,6.8,5.85+off,6.42)
  b(.8+off,4.2,3,1,'Context branch unchanged\n5 labels + occlusion\n6×16-D → 96-D → projection\n32-D token → dynamic gate','#00A6A6');b(5+off,4.35,1.8,.65,'Memory + gated\ncontext token');ar(3.8+off,4.7,5+off,4.7,'#00A6A6');ar(5.85+off,5.8,5.85+off,5,'#2F80ED');b(5+off,3,2,0.75,'MTN Decoder ×3\ncross-attn + flow + FFN');ar(5.9+off,4.35,6+off,3.75);b(5.1+off,1.6,1.8,.6,'Final output\n45 × 4-D');ar(6+off,3,6,2.2)
  if c4:
   b(7.2+off,6.5,1.55,1.3,'C4 shared Q/K/V/O\nEncoder.layers[0..2]','#00A6A6')
   for y in [8.1,7.1,6.1]:ar(7.2+off,y,6.7+off,y,'#00A6A6','--')
 ax.text(13.45,5.0,'Same Q/K/V/O parameters are reused\nacross all 3 Encoder layers.\nAll 3 attention operations still execute.\nParameter sharing ≠ computation sharing.',ha='center',fontsize=9,color='#006B6B',weight='bold')
 ax.text(9,.25,'C4 validation: Parameters −6.07% | MSE −5.01% | C-MSE −4.84% | CF-MSE −2.82% | CPU latency +10.14%\n3 seeds × 24 held-out trajectories • Candidate for further evaluation; not yet final architecture.',ha='center',fontsize=10,weight='bold',color='#8A4B08')
 return f
f=draw();f.savefig(OUT/'C4_MTN_Architecture.png',dpi=180,bbox_inches='tight');
with PdfPages(OUT/'C4_MTN_Architecture.pdf') as p:p.savefig(f,bbox_inches='tight')
plt.close(f)
