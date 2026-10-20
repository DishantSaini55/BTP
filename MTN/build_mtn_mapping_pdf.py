"""Supervisor-ready code-grounded MTN architecture mapping PDF."""
from pathlib import Path
import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans'})
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT=Path(r'D:\IDD-PeD-Evaluation\outputs\weight_sharing_targeted\MTN_Architecture_Mapping_and_Weight_Sharing_Explanation.pdf')
NAVY='#102A43';BLUE='#2F80ED';TEAL='#00A6A6';PALE='#F7FAFC';ORANGE='#F2994A';RED='#D64545'
def page(title,sub):
 f=plt.figure(figsize=(11.69,8.27),facecolor='white');f.text(.055,.93,title,fontsize=20,fontweight='bold',color=NAVY);f.text(.055,.892,sub,fontsize=9,color='#52616B');f.add_artist(plt.Line2D([.055,.945],[.875,.875],color=TEAL,lw=2));return f
def tx(f,x,y,s,sz=10,w='normal',c='#263238'):f.text(x,y,s,fontsize=sz,fontweight=w,color=c,va='top')
def box(ax,x,y,w,h,label,color=BLUE):
 ax.add_patch(plt.Rectangle((x,y),w,h,facecolor='#EEF5FF' if color==BLUE else '#E8F8F5',edgecolor=color,lw=1.8));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8.5,color=NAVY,wrap=True)
def arrow(ax,x1,y1,x2,y2,c=BLUE):ax.annotate('',(x2,y2),(x1,y1),arrowprops=dict(arrowstyle='->',color=c,lw=1.5))
def table(ax,cols,rows,widths,fs=7.6):
 ax.axis('off');t=ax.table(cellText=rows,colLabels=cols,loc='center',cellLoc='left',colLoc='left',colWidths=widths);t.auto_set_font_size(False);t.set_fontsize(fs);t.scale(1,1.38)
 for (r,c),cell in t.get_celld().items():
  cell.set_edgecolor('#D9E2EC')
  if r==0:cell.set_facecolor(NAVY);cell.get_text().set_color('white');cell.get_text().set_weight('bold')
  elif r%2==0:cell.set_facecolor(PALE)

with PdfPages(OUT) as pdf:
 f=page('MTN Architecture Mapping','Actual source-code audit of Version B and C4 — no new training or architecture modification')
 tx(f,.055,.82,'Direct answer: Where is the encoder?',14,'bold',NAVY)
 tx(f,.055,.78,'MTN has an explicit Encoder class: MTN/transformer/encoder.py::Encoder. It is constructed by\nIndividualTF (individual_TF.py) and stored at ContextIndividualTF.base.model.encoder. In this POC it owns\nthree EncoderLayer objects: encoder.layers[0], encoder.layers[1], encoder.layers[2].',11)
 tx(f,.055,.65,'Direct answer: At what layer is sharing done?',14,'bold',NAVY)
 tx(f,.055,.61,'C4 aliases only EncoderLayer.self_attn across those three actual encoder layers. Specifically, Q/K/V/O are\nMultiHeadAttention.linears[0]/[1]/[2]/[3] in encoder.layers[0].self_attn and the SAME module is referenced\nby encoder.layers[1].self_attn and encoder.layers[2].self_attn.',11,'bold',TEAL)
 tx(f,.055,.46,'What C4 does not share',13,'bold',NAVY)
 tx(f,.055,.42,'Encoder src_attn, encoder cross_attn, encoder FFNs/norms, every decoder module, all context modules, and\nthe prediction head remain independent and unchanged. This is not full encoder sharing or full architecture sharing.',10.5)
 tx(f,.055,.30,'Verified experimental terminology',13,'bold',NAVY)
 tx(f,.055,.265,'“C4: cross-depth sharing of EncoderLayer.self_attn Q/K/V/O in MTN Encoder.layers[0..2]” is technically\naccurate. The shorter prior label “encoder self-attention sharing” is correct but less specific.',10.5)
 tx(f,.055,.14,'Scope: shapes and paths are from the current 32/8/10 CPU POC (d_model=32, N=3); results are retained but\nnot reinterpreted as a full benchmark.',9.2,'normal',RED);pdf.savefig(f);plt.close(f)

 f=page('Actual Version B forward architecture','Exact class and module flow; B=batch size. Blue = original MTN, teal = Version B context addition.')
 ax=f.add_axes([.05,.12,.90,.70]);ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,7)
 box(ax,.2,5.8,1.55,.65,'boxes\n[B,15,4]');box(ax,2.15,5.8,1.75,.65,'src_embed\nLinear(4,32)+PE');box(ax,4.35,5.8,1.75,.65,'Encoder\nEncoder.layers ×3');box(ax,6.55,5.8,1.55,.65,'encode\n[B,15,32]');
 arrow(ax,1.75,6.13,2.15,6.13);arrow(ax,3.9,6.13,4.35,6.13);arrow(ax,6.1,6.13,6.55,6.13)
 box(ax,.2,4.45,1.55,.65,'speed slice\n[B,15,1]');box(ax,2.15,4.45,1.75,.65,'speed embed\n[B,1,32]');arrow(ax,1.75,4.78,2.15,4.78);arrow(ax,3.9,4.78,4.35,5.8)
 box(ax,.2,2.9,1.7,.8,'5 scene labels +\nocclusion [B,15]',TEAL);box(ax,2.2,2.9,1.85,.8,'SceneContextEncoder\n5×16 + pooled 16\nconcat [B,96]',TEAL);box(ax,4.4,2.9,1.8,.8,'projection + gate\n96→32→32; 36→16→2',TEAL);box(ax,6.55,3.0,1.55,.6,'append gated token\n[B,16,32]',TEAL)
 arrow(ax,1.9,3.3,2.2,3.3,TEAL);arrow(ax,4.05,3.3,4.4,3.3,TEAL);arrow(ax,6.2,3.3,6.55,3.3,TEAL);arrow(ax,7.3,3.6,7.3,5.8,TEAL)
 box(ax,6.55,1.55,1.6,.65,'speed-add memory\n[B,16,32]');box(ax,8.35,1.55,1.35,.65,'Decoder ×3\n[B,45,32]');box(ax,8.35,.35,1.35,.65,'Generator\nLinear(32,4)+residual');arrow(ax,7.3,3.0,7.3,2.2,TEAL);arrow(ax,8.15,1.88,8.35,1.88);arrow(ax,9.02,1.55,9.02,1.0)
 box(ax,.2,1.3,1.65,.65,'zero target\n[B,45,4]');box(ax,2.15,1.3,1.75,.65,'tgt_embed\n[B,45,32]');arrow(ax,1.85,1.63,2.15,1.63);arrow(ax,3.9,1.63,8.35,1.88)
 box(ax,4.25,.35,1.55,.65,'ego flow embed\n[B,64,32]');box(ax,4.25,1.25,1.55,.65,'ped flow embed\n[B,9,32]');arrow(ax,5.8,.68,8.35,1.75);arrow(ax,5.8,1.58,8.35,1.95)
 tx(f,.055,.09,'Encoder output becomes “memory” only after Version B appends the optional gated token; the decoder then adds the speed representation by broadcast.',8.8,'normal','#52616B');pdf.savefig(f);plt.close(f)

 f=page('Explicit Encoder and Decoder: module-level mapping','The code does use conventional Encoder/Decoder class names, but their attention pattern is MTN-specific.')
 rows=[['Input embedding','IndividualTF.src_embed','individual_TF.py','once','[B,15,4] → [B,15,32]'],['Encoder','Encoder / EncoderLayer','encoder.py / encoder_layer.py','0..2','x [B,15,32], m [B,1,32]'],['Encoder self-attn','EncoderLayer.self_attn','encoder_layer.py','0..2','Q=K=V=x'],['Encoder src-attn','EncoderLayer.src_attn','encoder_layer.py','0..2','Q=x; K/V=speed m'],['Encoder cross-attn','EncoderLayer.cross_attn','encoder_layer.py','0..2','Q=m; K/V=pre_x'],['Encoder FFN','EncoderLayer.feed_forward','encoder_layer.py','0..2','32→128→32 for x and m'],['Context branch','SceneContextEncoder + gate','context_TF.py','once','[B,5],[B,15] → gated [B,32]'],['Memory fusion','EncoderDecoder.forward/decode','encoder_decoder.py','once','15 tokens → 16 tokens'],['Decoder','Decoder / DecoderLayer','decoder.py / decoder_layer.py','0..2','target state [B,45,32]'],['Decoder src_attn','DecoderLayer.src_attn','decoder_layer.py','0..2','Q=decoder; K/V=memory'],['Decoder spd/ped','spd_attn / ped_attn','decoder_layer.py','0..2','Q=decoder; K/V=ego/ped flow'],['Output head','Generator.proj + residual','individual_TF.py','once','[B,45,32] → [B,45,4]']]
 ax=f.add_axes([.035,.13,.93,.70]);table(ax,['Diagram block','Actual class/module','Source file','Depth','Function'],rows,[.16,.21,.18,.08,.30],7.1)
 tx(f,.055,.09,'Important: DecoderLayer contains no active decoder self-attention. Its historical self-attention lines are commented out.',9,'bold',RED);pdf.savefig(f);plt.close(f)

 f=page('C4: exact sharing location and parameter trace','C4 is a targeted alias of a real submodule in the explicit Encoder stack.')
 tx(f,.055,.82,'Actual implementation: c4_encoder_shared.py::make_c4_encoder_self_attention_shared',11,'bold',NAVY)
 tx(f,.055,.785,'shared = model.base.model.encoder.layers[0].self_attn\nlayers[1].self_attn = shared\nlayers[2].self_attn = shared',12,'bold',TEAL)
 rows=[['Q','linears[0]','layers.0 self_attn','layers.1/.2 reference layer.0','weight [32,32], bias [32]'],['K','linears[1]','layers.0 self_attn','layers.1/.2 reference layer.0','weight [32,32], bias [32]'],['V','linears[2]','layers.0 self_attn','layers.1/.2 reference layer.0','weight [32,32], bias [32]'],['O','linears[3]','layers.0 self_attn','layers.1/.2 reference layer.0','weight [32,32], bias [32]']]
 ax=f.add_axes([.08,.45,.84,.23]);table(ax,['Projection','Actual field','Original C4 owner','C4 shared owner','Shape'],rows,[.12,.18,.22,.27,.20],8.5)
 tx(f,.055,.36,'Verification passed: Q/K/V/O module identity, Parameter identity, and data_ptr equality are true for layers 0/1/2.\nNamed parameters: 240 entries = 240 unique parameter objects; no duplicate independent optimizer parameters.',10.5,'bold',TEAL)
 tx(f,.055,.25,'Parameter audit: Version B 139,142; C4 130,694; reduction 8,448 (6.07%). Each Q/K/V/O Linear(32,32)\nhas 1,056 parameters; tying three complete self-attention modules retains one 4,224-parameter module instead of three.',10)
 tx(f,.055,.13,'Correct label: C4 — cross-depth sharing of EncoderLayer.self_attn Q/K/V/O in Encoder.layers[0..2].',10.5,'bold',NAVY);pdf.savefig(f);plt.close(f)

 f=page('Decoder src_attn experiments: C1, C2a, C2b','These are separate from C4 and occur in the explicit Decoder, not in the Encoder.')
 tx(f,.055,.82,'What is decoder src_attn?',13,'bold',NAVY);tx(f,.055,.785,'DecoderLayer.src_attn is MultiHeadAttention in decoder_layer.py. There are three copies: decoder.layers[0..2].src_attn.\nIt is cross-attention by data flow: Q is decoder state [B,45,32]; K/V are encoder-derived memory [B,16,32] in Version B.',10.3)
 rows=[['C1','entire src_attn module','Q/K/V/O','decoder layers 1,2 alias layer 0'],['C2a','src_attn.linears[0..2]','Q/K/V','O remains independent at each depth'],['C2b','src_attn.linears[1..2]','K/V','Q and O remain independent at each depth']]
 ax=f.add_axes([.10,.47,.80,.19]);table(ax,['Variant','Shared code object','Shared projections','What stays independent'],rows,[.11,.32,.18,.32],8.5)
 tx(f,.055,.37,'Why “cross-attention” is correct: MultiHeadAttention itself has one generic class; it becomes cross-attention because\nquery and key/value are from different streams. Its `cross_attention` argument is not used by this code.',10)
 tx(f,.055,.26,'Do not call these decoder self-attention experiments. The decoder self-attention call is commented out. spd_attn and\nped_attn are other decoder-to-flow cross-source attentions and were not tied in C1/C2.',10,'bold',RED);pdf.savefig(f);plt.close(f)

 f=page('Terminology corrections and supervisor-ready statement','Use these terms in future diagrams and discussions.')
 rows=[['“Encoder sharing”','C4 cross-depth EncoderLayer.self_attn sharing','Only self_attn, not all encoder parameters, is tied.'],['“Decoder self-attention”','No active decoder self-attention','The relevant code is commented out.'],['“src_attn”','decoder-to-memory cross-attention','Decoder state provides Q; memory provides K/V.'],['“speed attention”','encoder src_attn vs decoder spd_attn','Different modules and different sources.'],['“context token in encoder”','token appended after encode, before decode','It bypasses the EncoderLayer stack.'],['“memory”','encoder output + appended context; then speed-added','Two code operations create the decoder memory.']]
 ax=f.add_axes([.06,.44,.88,.30]);table(ax,['Current wording','Clear wording','Why'],rows,[.25,.35,.33],8)
 tx(f,.055,.34,'Suggested response to the supervisor',13,'bold',NAVY)
 tx(f,.055,.295,'“MTN has an explicit Encoder class in transformer/encoder.py. In our N=3 configuration it contains three EncoderLayer\nblocks. C4 shares only the Q/K/V/O projections of EncoderLayer.self_attn across those three blocks: layer 0 owns the\nattention module and layers 1 and 2 reference that same object. The remaining encoder attentions/FFNs, the entire\ndecoder, and the Version B context branch are unchanged.”',10.5,'bold',TEAL)
 tx(f,.055,.14,'Result retained without alteration: B MSE 972.513; C4 MSE 819.989. This supports follow-up only, not final model selection:\ntiny one-epoch CPU/Farneback POC, 10 held-out trajectories, no full-dataset confidence interval.',9.4,'normal',RED);pdf.savefig(f);plt.close(f)
print(OUT)
