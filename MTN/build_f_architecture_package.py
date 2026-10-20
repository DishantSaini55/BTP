"""Render the verified experimental F (DynMM-style) MTN architecture package."""
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans'})
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(r'D:\IDD-PeD-Evaluation\outputs\multimodal_architecture')
OUT.mkdir(parents=True, exist_ok=True)
NAVY, BLUE, TEAL, ORANGE, RED, INK = '#102A43', '#2F80ED', '#00A6A6', '#F2994A', '#C0392B', '#243B53'

def box(ax, x, y, w, h, text, color=BLUE, fs=7.4):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor='#EEF5FF' if color == BLUE else '#E8F8F5' if color == TEAL else '#FFF3E8', edgecolor=color, linewidth=1.45))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, color=INK, wrap=True)

def arrow(ax, a, b, color=BLUE, dashed=False):
    ax.annotate('', b, a, arrowprops=dict(arrowstyle='->', lw=1.35, color=color, linestyle='--' if dashed else '-'))

def diagram(fig=None):
    fig = fig or plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')
    fig.text(.5, .953, 'Dynamic Multimodal Fusion MTN — Experimental Architecture', ha='center', fontsize=18, fontweight='bold', color=NAVY)
    fig.text(.5, .923, 'Trajectory–Context Fusion using a DynMM-style Dynamic Fusion Cell', ha='center', fontsize=10.5, color='#52616B')
    ax.text(2.2, 8.35, 'Trajectory modality', ha='center', fontsize=10, weight='bold', color=BLUE)
    box(ax, .35, 7.4, 1.55, .58, 'Observed pedestrian trajectory\n[B,15,4]')
    box(ax, 2.2, 7.4, 1.75, .58, 'Source embedding +\npositional encoding\n[B,15,32]')
    box(ax, 4.25, 7.4, 1.65, .58, 'MTN Encoder ×3\nunchanged')
    box(ax, 6.2, 7.4, 1.45, .58, 'Trajectory memory\n[B,15,32]')
    arrow(ax, (1.9, 7.69), (2.2, 7.69)); arrow(ax, (3.95, 7.69), (4.25, 7.69)); arrow(ax, (5.9, 7.69), (6.2, 7.69))
    ax.text(13.2, 8.35, 'Context modality', ha='center', fontsize=10, weight='bold', color=TEAL)
    box(ax, 11.15, 7.18, 4.45, .8, 'Six native IDD-PeD context inputs\nintersection_type • signalized_type • road_type\nlocation_type • motion_direction • observed_occlusion', TEAL, 7.2)
    box(ax, 12.0, 6.08, 2.75, .56, '6 × 16-D embeddings', TEAL)
    box(ax, 10.45, 5.02, 2.85, .62, 'Token projection\nSix context tokens [B,6,32]', TEAL)
    box(ax, 13.58, 5.02, 1.95, .62, 'Concatenate\n[B,96]', TEAL)
    box(ax, 13.58, 4.50, 1.95, .50, 'Context projection\nGlobal context [B,32]', TEAL)
    arrow(ax, (13.38, 7.18), (13.38, 6.64), TEAL); arrow(ax, (13.2, 6.08), (11.9, 5.64), TEAL); arrow(ax, (13.55, 6.08), (14.55, 5.64), TEAL); arrow(ax, (14.55, 5.02), (14.55, 5.00), TEAL)
    ax.add_patch(plt.Rectangle((.35, 1.52), 15.2, 2.9, facecolor='#F7FBFF', edgecolor=ORANGE, linewidth=1.8))
    ax.text(7.95, 4.18, 'DYNAMIC MULTIMODAL FUSION', ha='center', fontsize=12, weight='bold', color=ORANGE)
    ax.text(7.95, 3.91, 'Inputs: trajectory memory [B,15,32] + context tokens [B,6,32] + global context [B,32]', ha='center', fontsize=7.6, color=INK)
    box(ax, .85, 2.48, 2.8, .72, 'Identity Fusion\nTrajectory memory unchanged\n[B,15,32]', ORANGE)
    box(ax, 4.1, 2.48, 3.0, .72, 'Additive Fusion\nGlobal context broadcast across 15 tokens\n+ trajectory memory → [B,15,32]', ORANGE)
    box(ax, 7.55, 2.28, 3.4, 1.1, 'Cross-Attention Fusion\nQ: trajectory [B,15,32]\nK,V: context tokens [B,6,32]\nMHA → residual → norm → FFN → [B,15,32]\nTrajectory queries Context', ORANGE, 6.85)
    box(ax, 11.45, 2.42, 1.62, .86, 'Dynamic gate\npooled trajectory [B,32]\n+ global context [B,32]\nconcat [B,64]\n64 → 32 → 3', ORANGE, 6.8)
    box(ax, 13.45, 2.42, 1.65, .86, 'Operation logits\n[Identity, Additive,\nCross-Attention]\nHard Gumbel-Softmax\n(train); Argmax / one-hot\n(eval)', ORANGE, 6.45)
    box(ax, 6.5, 1.72, 3.3, .4, 'Fused trajectory memory [B,15,32]', ORANGE, 7.8)
    arrow(ax, (6.9, 7.4), (2.25, 3.2), BLUE); arrow(ax, (6.9, 7.4), (5.6, 3.2), BLUE); arrow(ax, (7.1, 7.4), (9.0, 3.38), BLUE)
    arrow(ax, (11.9, 5.02), (9.25, 3.38), TEAL); arrow(ax, (14.55, 4.50), (5.85, 3.2), TEAL); arrow(ax, (14.55, 4.50), (12.25, 3.28), TEAL)
    arrow(ax, (2.25, 2.48), (7.2, 2.12), ORANGE); arrow(ax, (5.6, 2.48), (7.65, 2.12), ORANGE); arrow(ax, (9.25, 2.28), (8.1, 2.12), ORANGE)
    arrow(ax, (13.07, 2.85), (13.45, 2.85), ORANGE); arrow(ax, (14.25, 2.42), (9.55, 2.12), ORANGE, True)
    box(ax, 5.9, .72, 2.0, .54, 'MTN Decoder ×3\nexisting flow paths unchanged')
    box(ax, 8.35, .72, 2.0, .54, 'Output head\n+ final observed-box residual')
    box(ax, 10.8, .72, 1.8, .54, 'Future trajectory\n[B,45,4]')
    arrow(ax, (8.15, 1.72), (6.9, 1.26), BLUE); arrow(ax, (7.9, .99), (8.35, .99), BLUE); arrow(ax, (10.35, .99), (10.8, .99), BLUE)
    box(ax, .5, .28, 4.65, .82, 'Computational note\nAll three candidate fusion branches are computed before selection. F provides dynamic fusion but does NOT skip unused branches; it is not a computational-saving architecture. No resource-aware loss is used.', RED, 6.75)
    box(ax, 12.95, .17, 2.55, 1.05, 'F Experimental Evidence\nMSE 1154.78 ± 9.95\nC-MSE 1124.20 ± 10.08\nCF-MSE 3098.16 ± 137.74\nParameters 154,007\nCPU latency 50.68 ± 2.48 ms', RED, 6.45)
    fig.text(.5, .012, 'Solid arrows = data flow. Dashed orange arrows = control/gating signals. CPU Farneback POC: one held-out video, 24 trajectories, one epoch, three seeds; not an official RAFT-based IDD-PeD benchmark.', ha='center', fontsize=7.1, color='#52616B')
    return fig

def drawio():
    cells=[]; cid=2
    def add(value, x, y, w, h, style='rounded=1;whiteSpace=wrap;html=1;fillColor=#EEF5FF;strokeColor=#2F80ED;', parent='1'):
        nonlocal cid; i=str(cid); cid += 1; cells.append(f'<mxCell id="{i}" value="{escape(value)}" style="{style}" vertex="1" parent="{parent}"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'); return i
    def edge(src, dst, dashed=False):
        nonlocal cid; i=str(cid); cid += 1; style='endArrow=block;html=1;rounded=0;' + ('dashed=1;dashPattern=6 6;strokeColor=#F2994A;' if dashed else 'strokeColor=#2F80ED;'); cells.append(f'<mxCell id="{i}" style="{style}" edge="1" parent="1" source="{src}" target="{dst}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    title=add('Dynamic Multimodal Fusion MTN — Experimental Architecture',420,20,660,40,'text;html=1;fontSize=20;fontStyle=1;fontColor=#102A43;')
    add('Trajectory modality',75,80,200,28,'text;html=1;fontSize=14;fontStyle=1;fontColor=#2F80ED;')
    obs=add('Observed pedestrian trajectory&lt;br&gt;[B,15,4]',40,125,160,62); emb=add('Source embedding + positional encoding&lt;br&gt;[B,15,32]',230,125,180,62); enc=add('MTN Encoder ×3&lt;br&gt;unchanged',440,125,160,62); mem=add('Trajectory memory&lt;br&gt;[B,15,32]',630,125,160,62)
    add('Context modality',1120,80,200,28,'text;html=1;fontSize=14;fontStyle=1;fontColor=#00A6A6;'); inp=add('Six native IDD-PeD context inputs&lt;br&gt;intersection_type • signalized_type • road_type&lt;br&gt;location_type • motion_direction • observed_occlusion',1050,120,400,76,'rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F8F5;strokeColor=#00A6A6;'); e16=add('6 × 16-D embeddings',1140,225,200,55,'rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F8F5;strokeColor=#00A6A6;'); tok=add('Token projection&lt;br&gt;Six context tokens [B,6,32]',985,315,235,65,'rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F8F5;strokeColor=#00A6A6;'); cat=add('Concatenate [B,96]',1260,315,180,55,'rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F8F5;strokeColor=#00A6A6;'); glob=add('Context projection&lt;br&gt;Global context [B,32]',1260,390,180,55,'rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F8F5;strokeColor=#00A6A6;')
    fusion=add('DYNAMIC MULTIMODAL FUSION\nInputs: trajectory memory + context tokens + global context',85,455,1370,45,'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF8F0;strokeColor=#F2994A;fontSize=14;fontStyle=1;'); identity=add('Identity Fusion&lt;br&gt;Trajectory memory unchanged&lt;br&gt;[B,15,32]',120,540,230,75,'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E8;strokeColor=#F2994A;'); additive=add('Additive Fusion&lt;br&gt;Global context broadcast across 15 tokens + trajectory&lt;br&gt;[B,15,32]',400,540,280,75,'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E8;strokeColor=#F2994A;'); cross=add('Cross-Attention Fusion&lt;br&gt;Q: trajectory [B,15,32]&lt;br&gt;K,V: context tokens [B,6,32]&lt;br&gt;MHA → residual → norm → FFN&lt;br&gt;Trajectory queries Context',730,525,300,105,'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E8;strokeColor=#F2994A;'); gate=add('Dynamic gate&lt;br&gt;pooled trajectory + global context&lt;br&gt;[B,64] → 64 → 32 → 3',1080,535,190,85,'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E8;strokeColor=#F2994A;'); select=add('Hard Gumbel-Softmax (train)&lt;br&gt;Argmax / one-hot (eval)&lt;br&gt;selects one operation per sample',1300,535,165,85,'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E8;strokeColor=#F2994A;'); fused=add('Fused trajectory memory [B,15,32]',660,680,270,48,'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E8;strokeColor=#F2994A;fontStyle=1;'); dec=add('MTN Decoder ×3&lt;br&gt;existing flow paths unchanged',700,770,210,65); out=add('Output head + final observed-box residual',960,770,230,65); pred=add('Future trajectory&lt;br&gt;[B,45,4]',1240,770,180,65)
    note=add('All three candidate fusion branches are computed before selection. Dynamic fusion only; no branch skipping and no resource-aware loss.',70,865,700,58,'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF7F7;strokeColor=#C0392B;'); evidence=add('F Experimental Evidence&lt;br&gt;MSE 1154.78 ± 9.95; C-MSE 1124.20 ± 10.08; CF-MSE 3098.16 ± 137.74&lt;br&gt;Parameters 154,007; CPU latency 50.68 ± 2.48 ms',820,850,620,76,'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF7F7;strokeColor=#C0392B;')
    for a,b in [(obs,emb),(emb,enc),(enc,mem),(inp,e16),(e16,tok),(e16,cat),(cat,glob),(mem,identity),(mem,additive),(mem,cross),(tok,cross),(glob,additive),(glob,gate),(gate,select),(identity,fused),(additive,fused),(cross,fused),(fused,dec),(dec,out),(out,pred)]: edge(a,b)
    edge(select, fused, True)
    return '<mxfile host="app.diagrams.net" type="device"><diagram name="F Architecture">' + '<mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + ''.join(cells) + '</root></mxGraphModel></diagram></mxfile>'

def add_text_page(pdf, title, body):
    f=plt.figure(figsize=(16,9)); f.text(.07,.91,title,fontsize=21,fontweight='bold',color=NAVY); f.text(.08,.81,body,fontsize=12,va='top',color=INK,linespacing=1.55,wrap=True); pdf.savefig(f,bbox_inches='tight'); plt.close(f)

def third_page(pdf):
    f=plt.figure(figsize=(16,9)); ax=f.add_axes([.06,.25,.88,.48]); ax.axis('off'); f.text(.07,.9,'F vs B vs C4 — Experimental Interpretation',fontsize=21,fontweight='bold',color=NAVY)
    rows=[['B','Context-aware MTN','1137.11 ± 20.16','139,142','45.58 ± 3.23 ms'],['C4','Context + encoder self-attention sharing','1080.16 ± 93.33','130,694','47.38 ± 2.48 ms'],['F','Dynamic multimodal fusion','1154.78 ± 9.95','154,007','50.68 ± 2.48 ms']]
    tab=ax.table(cellText=rows,colLabels=['Model','Definition','MSE','Parameters','CPU latency'],loc='center',cellLoc='center',colWidths=[.1,.35,.2,.16,.19]); tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1,2.3)
    for (r,c),cell in tab.get_celld().items(): cell.set_edgecolor('#D9E2EC'); cell.set_facecolor('#DCEEFF' if r==0 else '#FFFFFF'); cell.set_text_props(weight='bold' if r==0 else 'normal')
    f.text(.08,.16,'C4 is the current proposed POC architecture. F is the evaluated Dynamic Multimodal Fusion candidate.',fontsize=12,color=TEAL,weight='bold')
    f.text(.08,.09,'F versus B: MSE +1.55%, C-MSE +1.52%, CF-MSE +4.27%, parameters +10.68%, latency +11.18%. F did not improve the measured prediction metrics over B and increased parameter count and CPU latency.',fontsize=10.5,color=RED,wrap=True)
    pdf.savefig(f,bbox_inches='tight'); plt.close(f)

def main():
    (OUT/'Dynamic_Multimodal_Fusion_MTN.drawio').write_text(drawio(),encoding='utf-8')
    f=diagram(); f.savefig(OUT/'Dynamic_Multimodal_Fusion_MTN.svg',format='svg',bbox_inches='tight'); f.savefig(OUT/'Dynamic_Multimodal_Fusion_MTN.png',dpi=300,bbox_inches='tight'); plt.close(f)
    with PdfPages(OUT/'Dynamic_Multimodal_Fusion_MTN.pdf') as pdf:
        f=diagram(); pdf.savefig(f,bbox_inches='tight'); plt.close(f)
        add_text_page(pdf,'Module-wise Description','''1. MTN trajectory encoder — observed boxes [B,15,4] become [B,15,32]; the unchanged three-layer encoder produces trajectory memory [B,15,32].

2. Six-token context representation — the five static IDD-PeD labels plus mean-pooled observed occlusion each form a 16-D embedding. A shared linear projection creates six tokens [B,6,32].

3. Global context representation — the six 16-D embeddings concatenate to [B,96] and pass through LayerNorm → Linear(96,32) → GELU → Dropout → Linear(32,32), yielding global context [B,32].

4. Identity fusion — retains trajectory memory unchanged.

5. Additive fusion — broadcasts global context across 15 trajectory tokens, adds it to trajectory memory, then applies LayerNorm.

6. Cross-attention fusion — trajectory features act as queries, while the six context tokens provide keys and values. Multi-head cross-attention is followed by a residual connection, normalization, FFN and a second residual/normalization.

7. Dynamic gate — pooled trajectory [B,32] and global context [B,32] form [B,64]. The MLP 64 → 32 → 3 selects one of the three candidate operations per sample: hard Gumbel-Softmax during training and argmax/one-hot during evaluation.

8. Fused trajectory representation — the selected result remains [B,15,32]. All three candidates are currently calculated before selection.

9. MTN decoder — the unchanged three-layer decoder consumes fused memory with the existing ego-flow and pedestrian-flow paths.

10. Prediction head — Linear(32,4) plus the final observed-box residual produces [B,45,4].''')
        third_page(pdf)
    (OUT/'Dynamic_Multimodal_Fusion_MTN_Description.md').write_text('''# Dynamic Multimodal Fusion MTN — Experimental Architecture\n\n## Status\n\nF is an evaluated experimental candidate, not the selected C4 POC architecture. It does not include C4 sharing.\n\n## Architecture\n\nObserved pedestrian boxes `[B,15,4]` are embedded to `[B,15,32]` and processed by the unchanged MTN encoder ×3, producing trajectory memory `[B,15,32]`. Six native context sources are each embedded to 16-D. Their per-source projections create six context tokens `[B,6,32]`, while their concatenation `[B,96]` is separately projected to global context `[B,32]`.\n\nThe Dynamic Multimodal Fusion cell computes identity, additive and cross-attention candidates. Cross-attention uses trajectory features as Q and context tokens as K/V. A per-sample gate uses pooled trajectory plus global context, `[B,64] → 32 → 3`, to select one candidate: hard Gumbel-Softmax in training and argmax/one-hot in evaluation. The fused output remains `[B,15,32]`, enters the unchanged MTN decoder ×3 and output head/residual, and predicts `[B,45,4]`.\n\n## Computation\n\nAll three fusion candidates are executed before gate selection. F therefore implements dynamic fusion, but does not skip unused branches and has no resource-aware loss; it is not a computational-saving architecture.\n\n## Unified POC evidence\n\nF: MSE `1154.78 ± 9.95`, C-MSE `1124.20 ± 10.08`, CF-MSE `3098.16 ± 137.74`, parameters `154,007`, latency `50.68 ± 2.48 ms`. Versus B: MSE `+1.55%`, C-MSE `+1.52%`, CF-MSE `+4.27%`, parameters `+10.68%`, latency `+11.18%`. F did not improve measured prediction metrics over B.\n\nResults are CPU Farneback POC measurements: one held-out video, 24 trajectories, one training epoch and three seeds, not an official RAFT-based IDD-PeD benchmark.\n''',encoding='utf-8')

if __name__ == '__main__': main()
