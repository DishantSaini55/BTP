from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).parent/'diagrams'; OUT.mkdir(exist_ok=True)
def font(size,bold=False):
 for p in (['C:/Windows/Fonts/arialbd.ttf'] if bold else ['C:/Windows/Fonts/arial.ttf']):
  try:return ImageFont.truetype(p,size)
  except:pass
 return ImageFont.load_default()
# Each diagram uses nodes with [id,label,x,y,w,h,color], edges [src,dst,label].
diagrams={
'overall_comparative_framework':(1500,820,[
('data','IDD-PeD annotations\n15 observed boxes\n45 target boxes',40,300,210,140,'#dbeafe'),
('split','Frozen splits + train-only\nnormalization\nGP3/5/8/9 held-out',320,300,235,140,'#e0e7ff'),
('cv','AO-0\nconstant velocity',650,65,205,100,'#dbeafe'),
('ridge','AO-RIDGE\nlinear residual regression',650,210,205,110,'#dcfce7'),
('ao1','AO-1\nconstant-position anchor',650,370,205,100,'#fee2e2'),
('ao2','AO-2\nCV-anchored transformer',650,520,205,100,'#fef3c7'),
('ao3','AO-3\nshared encoder attention',650,670,205,100,'#ede9fe'),
('eval','Frozen official test\nMSE + C-MSE + CF-MSE\n13 videos / 861 tracks',1030,300,235,150,'#f1f5f9'),
('craft','Development-only\nAO-3 temporal analysis\nno test explanations',1280,300,205,150,'#fce7f3')],
[('data','split',''),('split','cv',''),('split','ridge',''),('split','ao1',''),('split','ao2',''),('split','ao3',''),('cv','eval',''),('ridge','eval',''),('ao1','eval',''),('ao2','eval',''),('ao3','eval',''),('ao3','craft','')]),
'cv_anchored_transformer':(1500,720,[
('obs','Observed boxes\n[B,15,4] (xyxy)',40,280,200,105,'#dbeafe'),('ip','Input projection 4→32\n× √32 + sinusoidal PE',290,280,225,105,'#e0e7ff'),('enc','Encoder ×3\n4 heads, d=32\nFFN 32→128→32 (ReLU)',570,250,260,165,'#ede9fe'),('mem','Encoder memory\n[B,15,32]',885,280,205,105,'#e0e7ff'),('qry','Zero target boxes\n[B,45,4] → projection\n+ target PE',290,500,225,115,'#f1f5f9'),('dec','Decoder ×3\ncross-attention to memory\nFFN + residual/LN\n(no target self-attention)',590,475,270,165,'#dcfce7'),('corr','Generator 32→4\ncorrection [B,45,4]',920,500,220,115,'#fef3c7'),('anchor','CV anchor [B,45,4]\nlast box + k·velocity\nvelocity=(x[t]-x[t-2])/2',40,500,210,115,'#dbeafe'),('out','Forecast = anchor + correction\n[B,45,4] raw after inverse norm',1180,330,270,125,'#fee2e2')],
[('obs','ip',''),('ip','enc',''),('enc','mem',''),('mem','dec','K,V'),('qry','dec','Q'),('dec','corr',''),('obs','anchor',''),('anchor','out','+'),('corr','out','+')]),
'satie_parameter_sharing':(1450,640,[
('in','Input stream\n[B,15,32]',40,260,190,100,'#dbeafe'),('e1','Encoder layer 1\nSelf-attention module A\nFFN + norms (per layer)',330,80,250,130,'#ede9fe'),('e2','Encoder layer 2\nSame module A\nFFN + norms (per layer)',650,240,250,130,'#ede9fe'),('e3','Encoder layer 3\nSame module A\nFFN + norms (per layer)',330,430,250,130,'#ede9fe'),('a','Shared trainable parameters\nAttention A executed 3 times\nQ/K/V/output projections shared',1010,230,370,150,'#fef3c7'),('out','Encoded memory\n[B,15,32]',650,465,220,100,'#dbeafe')],
[('in','e1',''),('e1','e2',''),('e2','e3',''),('e3','out',''),('e1','a',''),('e2','a',''),('e3','a','')]),
'ridge_residual_forecasting':(1750,600,[
('obs','Observed normalized boxes\n15 x 4 = 60 features',40,220,230,110,'#dbeafe'),('last','Last normalized box\n4 features',40,390,230,90,'#dbeafe'),
('feat','Concatenate = 64\ntrain-fitted standardization',350,280,250,120,'#e0e7ff'),('ridge','Ridge regression\nalpha=100 (GP7 validation selected)\n11,700 fitted coefficients',700,260,285,145,'#dcfce7'),
('res','Predicted normalized\n45 x 4 CV residual',1040,280,250,110,'#fef3c7'),('cv','CV forecast in\nnormalized coordinates',700,465,285,90,'#dbeafe'),
('add','Add residual in same\nnormalized coordinate system',1040,455,250,100,'#ede9fe'),('inv','Frozen train-only inverse\nnormalization to pixel forecast',1360,330,340,140,'#fee2e2')],
[('obs','feat',''),('last','feat',''),('feat','ridge',''),('ridge','res',''),('cv','add','+'),('res','add','+'),('add','inv','')]),
'craft_temporal_concepts':(1500,700,[
('frozen','Frozen AO-3\nseed 20260928',35,275,200,100,'#dbeafe'),('act','FFN ReLU activations\ntrain [988,15,128]\nval [432,15,128]',285,250,250,150,'#e0e7ff'),('nmf','Rank-4 NMF bank\nW≥0, H≥0\ntrain-only fit',595,250,225,150,'#ede9fe'),('coef','Concept coefficients\nper window × time\nC1–C4 numeric only',880,250,245,150,'#dcfce7'),('recursive','Selected parent tokens\naligned preceding-layer\nReLU; recursive NMF\n(C2–C4 screened)',1165,85,275,145,'#fef3c7'),('sobol','Independent amplitude\ninterventions on selected\nFFN branch; 4 scalar outputs',1165,280,275,145,'#fce7f3'),('fid','Insertion/deletion\nwithin reconstructed branch\n7/8 screens; one failed',1165,485,275,130,'#fee2e2'),('dev','GP7 development only\nNo official-test access\nNo semantic labels/maps/humans',285,500,535,110,'#f1f5f9')],
[('frozen','act',''),('act','nmf',''),('nmf','coef',''),('coef','recursive',''),('coef','sobol',''),('sobol','fid',''),('dev','sobol','')])}
for name,(W,H,nodes,edges) in diagrams.items():
 nmap={n[0]:n for n in nodes}
 # SVG vector
 s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#526579"/></marker></defs>', '<rect width="100%" height="100%" fill="#fbfcfe"/>']
 for a,b,l in edges:
  A=nmap[a];B=nmap[b];x1=A[2]+A[4];y1=A[3]+A[5]/2;x2=B[2];y2=B[3]+B[5]/2
  if x2<x1: x1=A[2]+A[4]/2;y1=A[3]+A[5];x2=B[2]+B[4]/2;y2=B[3]
  s.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#526579" stroke-width="3" fill="none" marker-end="url(#arrow)"/>')
  if l:s.append(f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-7}" font-family="Arial" font-size="15" text-anchor="middle" fill="#526579">{escape(l)}</text>')
 for id,label,x,y,w,h,c in nodes:
  lines=label.split('\n');s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="{c}" stroke="#718096" stroke-width="2"/>')
  lineh=25;start=y+h/2-(len(lines)-1)*lineh/2
  for i,line in enumerate(lines):s.append(f'<text x="{x+w/2}" y="{start+i*lineh}" text-anchor="middle" dominant-baseline="middle" font-family="Arial" font-size="18" fill="#17324d">{escape(line)}</text>')
 s.append('</svg>');(OUT/f'{name}.svg').write_text('\n'.join(s),encoding='utf-8')
 # editable Mermaid source nlabels='\n'.join(f' {id}["{label.replace(chr(10),"<br/>")}"]' for id,label,*_ in nodes); elines='\n'.join(f' {a} -->|{lab}| {b}' if lab else f' {a} --> {b}' for a,b,lab in edges);(OUT/f'{name}.mmd').write_text('flowchart LR\n'+nlabels+'\n'+elines+'\n',encoding='utf-8')
 # draw.io editable mxGraph nodes/edges
 cells=['<mxCell id="0"/>','<mxCell id="1" parent="0"/>']
 for id,label,x,y,w,h,c in nodes:
  val=escape(label).replace('\n','&#xa;');cells.append(f'<mxCell id="{id}" value="{val}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={c};strokeColor=#718096;fontColor=#17324d;" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
 for i,(a,b,l) in enumerate(edges):cells.append(f'<mxCell id="e{i}" value="{escape(l)}" style="endArrow=block;html=1;strokeColor=#526579;" edge="1" parent="1" source="{a}" target="{b}"><mxGeometry relative="1" as="geometry"/></mxCell>')
 xml=f'<mxfile host="app.diagrams.net"><diagram name="{name}"><mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="10" page="1" pageWidth="{W}" pageHeight="{H}"><root>{"".join(cells)}</root></mxGraphModel></diagram></mxfile>';(OUT/f'{name}.drawio').write_text(xml,encoding='utf-8')
 # Pillow raster equivalent
 im=Image.new('RGB',(W,H),'#fbfcfe');d=ImageDraw.Draw(im)
 for a,b,l in edges:
  A=nmap[a];B=nmap[b];x1=A[2]+A[4];y1=A[3]+A[5]/2;x2=B[2];y2=B[3]+B[5]/2
  if x2<x1:x1=A[2]+A[4]/2;y1=A[3]+A[5];x2=B[2]+B[4]/2;y2=B[3]
  d.line((x1,y1,x2,y2),fill='#526579',width=3);d.polygon([(x2,y2),(x2-12,y2-6),(x2-12,y2+6)],fill='#526579')
  if l:d.text(((x1+x2)/2,(y1+y2)/2-15),l,font=font(15),fill='#526579',anchor='mm')
 for id,label,x,y,w,h,c in nodes:
  d.rounded_rectangle((x,y,x+w,y+h),radius=15,fill=c,outline='#718096',width=2);lines=label.split('\n');lineh=25;start=y+h/2-(len(lines)-1)*lineh/2
  for i,line in enumerate(lines): d.text((x+w/2,start+i*lineh),line,font=font(18),fill='#17324d',anchor='mm')
 im.save(OUT/f'{name}.png')
print('created',len(diagrams),'editable diagram sets')
