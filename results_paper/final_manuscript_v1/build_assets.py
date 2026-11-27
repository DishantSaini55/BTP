from pathlib import Path
import csv, math, statistics, hashlib
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
ROOT=Path.cwd(); OUT=ROOT/'results_paper/final_manuscript_v1'; SRC=ROOT/'results_paper/AO_confirmation_v1'
summary=list(csv.DictReader((SRC/'OFFICIAL_TEST_MODEL_SUMMARY_v3.csv').open(encoding='utf-8-sig',newline='')))
seed=list(csv.DictReader((SRC/'OFFICIAL_TEST_RESULTS_v2/OFFICIAL_TEST_SEED_RESULTS.csv').open(encoding='utf-8-sig',newline='')))
video=list(csv.DictReader((SRC/'OFFICIAL_TEST_RESULTS_v2/OFFICIAL_TEST_PER_VIDEO.csv').open(encoding='utf-8-sig',newline='')))
track=list(csv.DictReader((SRC/'OFFICIAL_TEST_RESULTS_v2/OFFICIAL_TEST_PER_TRACK.csv').open(encoding='utf-8-sig',newline='')))
window=list(csv.DictReader((SRC/'OFFICIAL_TEST_RESULTS_v2/OFFICIAL_TEST_PER_WINDOW.csv').open(encoding='utf-8-sig',newline='')))
macro=list(csv.DictReader((SRC/'OFFICIAL_TEST_MACRO_RESULTS.csv').open(encoding='utf-8-sig',newline='')))
paired=list(csv.DictReader((SRC/'OFFICIAL_TEST_PAIRED_COMPARISONS_v3.csv').open(encoding='utf-8-sig',newline='')))
order=['AO-0','AO-1','AO-2','AO-3','AO-RIDGE','AO-CP']; sm={r['model_id']:r for r in summary}
# Aggregate macro first across run-level macro metrics.
macroagg={}
for m in order:
 rows=[r for r in macro if r['model_id']==m]
 macroagg[m]=(statistics.mean(float(r['macro_video_mse']) for r in rows),statistics.mean(float(r['macro_track_mse']) for r in rows))
# deliver transparent canonical CSVs and human-readable model tables
(OUT/'supporting_data').mkdir(exist_ok=True)
for name in ['OFFICIAL_TEST_PER_VIDEO.csv','OFFICIAL_TEST_PER_TRACK.csv','OFFICIAL_TEST_PER_WINDOW.csv','OFFICIAL_TEST_SEED_RESULTS.csv']:
 (OUT/'supporting_data'/name).write_bytes((SRC/'OFFICIAL_TEST_RESULTS_v2'/name).read_bytes())
(OUT/'supporting_data'/'OFFICIAL_TEST_MACRO_RESULTS.csv').write_bytes((SRC/'OFFICIAL_TEST_MACRO_RESULTS.csv').read_bytes())
(OUT/'supporting_data'/'OFFICIAL_TEST_PAIRED_COMPARISONS_v3.csv').write_bytes((SRC/'OFFICIAL_TEST_PAIRED_COMPARISONS_v3.csv').read_bytes())
with (OUT/'supporting_data'/'MANUSCRIPT_AGGREGATES.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['model','runs','parameters','mse_mean','mse_sd','cmse','cfmse','macro_video_mse_mean_over_runs','macro_track_mse_mean_over_runs','horizon1','horizon15','horizon30','horizon45'])
 for m in order:
  r=sm[m]; w.writerow([m,r['n_completed_runs'],r['parameter_count'],r['mse_mean'],r['mse_sd'],r['cmse_mean'],r['cfmse_mean'],*macroagg[m],r['horizon_1_mse_mean'],r['horizon_15_mse_mean'],r['horizon_30_mse_mean'],r['horizon_45_mse_mean']])
# Need use full metric definitions from frozen evaluator. Plot theme.
W,H=1500,900; BG='#fbfcfe'; NAVY='#17324d'; GRID='#d9e1ea'; COLORS={'AO-0':'#3b82f6','AO-1':'#ef4444','AO-2':'#16a34a','AO-3':'#8b5cf6','AO-RIDGE':'#111827','AO-CP':'#f59e0b'}
def font(size,bold=False):
 candidates=['C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf','C:/Windows/Fonts/calibrib.ttf' if bold else 'C:/Windows/Fonts/calibri.ttf']
 for p in candidates:
  try:return ImageFont.truetype(p,size)
  except:pass
 return ImageFont.load_default()
def frame(title,subtitle=None):
 im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);d.text((80,42),title,font=font(38,True),fill=NAVY)
 if subtitle:d.text((82,93),subtitle,font=font(20),fill='#526579')
 return im,d
def save(im,stem):
 for ext in ['png']:
  im.save(OUT/'figures'/f'{stem}.{ext}',dpi=(220,220))
# grouped bar utility, 6 models x cats. optional legend.
def grouped(title,subtitle,categories,series,stem,ylabel,log=False):
 im,d=frame(title,subtitle);left,top,right,bottom=150,180,1370,730
 vals=[v for r in series.values() for v in r if v is not None];vmax=max(vals)*1.08
 if log:vmin=max(1,min(vals)*.75); lo=math.log(vmin); hi=math.log(vmax)
 else:vmin=0;lo=0;hi=vmax
 def yy(v):return bottom-(math.log(max(v,vmin))-lo)/(hi-lo)*(bottom-top) if log else bottom-v/(hi-lo)*(bottom-top)
 for i in range(6):
  frac=i/5; y=bottom-frac*(bottom-top); d.line((left,y,right,y),fill=GRID,width=2); lab=(f'{math.exp(lo+frac*(hi-lo)):.0f}' if log else f'{vmax*frac:.0f}');d.text((70,y-12),lab,font=font(18),fill='#526579')
 d.text((35,top),ylabel,font=font(17),fill=NAVY)
 groupw=(right-left)/len(categories); bw=min(33,groupw/(len(series)+1))
 for ci,cat in enumerate(categories):
  cx=left+(ci+.5)*groupw; d.text((cx-40,bottom+22),str(cat),font=font(17),fill=NAVY)
  for si,(model,vs) in enumerate(series.items()):
   v=vs[ci]
   if v is None:continue
   x=cx-(len(series)*bw)/2+si*bw;y=yy(v);d.rounded_rectangle((x,y,x+bw-4,bottom),radius=3,fill=COLORS[model])
 # legend
 x=left;y=bottom+75
 for m in series:
  d.rounded_rectangle((x,y,x+22,y+22),radius=3,fill=COLORS[m]);d.text((x+30,y),m,font=font(17),fill=NAVY);x+=170
 save(im,stem)
# Horizon results
horizons=['1','15','30','45']; series={m:[float(sm[m][f'horizon_{i}_mse_mean']) for i in [1,15,30,45]] for m in order}
grouped('Prediction error grows with forecast horizon','Mean raw-pixel MSE at selected future frames (descriptive; across available runs)',horizons,series,'horizon_mse','MSE (px²)')
# per-video mean each model/video over available seed rows
vmap=defaultdict(list)
for r in video:vmap[(r['model_id'],r['video_id'])].append(float(r['mse']))
vids=sorted({r['video_id'] for r in video}); labels=[f'GP{int(v.split("_vid_")[0].replace("gp_set_",""))}-V{int(v.split("_vid_")[1])}' for v in vids]; vseries={m:[statistics.mean(vmap[(m,v)]) for v in vids] for m in order}
with (OUT/'supporting_data'/'VIDEO_INDEX_KEY.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['plot_label','video_id']);w.writerows(zip(labels,vids))
grouped('Performance across official-test videos','Per-video MSE averaged over available seeds; IDs listed in VIDEO_INDEX_KEY.csv',labels,vseries,'per_video_mse','MSE (px^2)')
# track error distributions, average each track across available seed values
trmap=defaultdict(list)
for r in track:trmap[(r['model_id'],r['video_id'],r['track_id'])].append(float(r['mse']))
def quantile(a,p):
 a=sorted(a);idx=(len(a)-1)*p; lo=int(idx);hi=min(lo+1,len(a)-1);return a[lo]*(hi-idx)+a[hi]*(idx-lo)
# custom box plot
im,d=frame('Track-cluster error distribution','Each point of aggregation is a (video, track) cluster; neural errors averaged across seeds per track')
left,top,right,bottom=150,180,1380,730; vv={m:[statistics.mean(vals) for (mm,_,_),vals in trmap.items() if mm==m] for m in order};vmax=quantile([x for a in vv.values() for x in a],.95)*1.1
for i in range(6):
 y=bottom-i*(bottom-top)/5;d.line((left,y,right,y),fill=GRID,width=2);d.text((72,y-12),f'{vmax*i/5:.0f}',font=font(18),fill='#526579')
for i,m in enumerate(order):
 x=left+(i+.5)*(right-left)/6;a=vv[m]; q1,med,q3=[quantile(a,p) for p in [.25,.5,.75]];lo,hi=[quantile(a,p) for p in [.1,.9]]
 yy=lambda z:bottom-z/vmax*(bottom-top)
 d.line((x,yy(hi),x,yy(lo)),fill=COLORS[m],width=4);d.line((x-20,yy(hi),x+20,yy(hi)),fill=COLORS[m],width=4);d.line((x-20,yy(lo),x+20,yy(lo)),fill=COLORS[m],width=4);d.rectangle((x-40,yy(q3),x+40,yy(q1)),outline=COLORS[m],width=4);d.line((x-40,yy(med),x+40,yy(med)),fill=NAVY,width=4);d.text((x-40,bottom+22),m,font=font(18),fill=NAVY)
save(im,'per_track_distribution')
# pooled per-window distribution: average same sample across neural seeds
wmap=defaultdict(list)
for r in window:wmap[(r['model_id'],r['sample_id'])].append(float(r['mse']))
# Histogram percentiles on log scale bins across all data (exclude top 0.2% for chart scale, report full CSV)
# plot all model distributions as quantile line summaries to avoid dense opaque hist
im,d=frame('Distribution of per-window forecast errors','Per-window MSE from canonical v2 records; neural values averaged over five seed predictions; display capped at 99th percentile')
left,top,right,bottom=160,180,1380,730; vals={m:[statistics.mean(v) for (mm,_),v in wmap.items() if mm==m] for m in order}; vmax=quantile([x for aa in vals.values() for x in aa],.99)
for i in range(6):
 y=bottom-i*(bottom-top)/5;d.line((left,y,right,y),fill=GRID,width=2);d.text((90,y-12),str(i*20),font=font(18),fill='#526579')
# scaled empirical CDFs at x sampled MSE grid, vertical axis percentage
for m,a in vals.items():
 pts=[]
 for j in range(101):
  xval=vmax*j/100; pct=sum(z<=xval for z in a)/len(a)*100; x=left+j*(right-left)/100;y=bottom-pct*(bottom-top)/100;pts.append((x,y))
 d.line(pts,fill=COLORS[m],width=4)
for j in range(6):
 x=left+j*(right-left)/5;d.line((x,top,x,bottom),fill='#eef1f5',width=1);d.text((x-20,bottom+18),f'{vmax*j/5:.0f}',font=font(17),fill='#526579')
for i,m in enumerate(order):
 x=170+(i%3)*380;y=780+(i//3)*36;d.line((x,y+10,x+42,y+10),fill=COLORS[m],width=4);d.text((x+52,y),m,font=font(17),fill=NAVY)
d.text((70,160),'Cumulative (%)',font=font(16),fill=NAVY);d.text((580,848),'Per-window MSE (px²)',font=font(17),fill=NAVY)
save(im,'per_window_error_ecdf')
# parameter accuracy
im,d=frame('Accuracy and model size','Trainable parameters vs. official-test mean MSE; deterministic baselines at zero parameters')
left,top,right,bottom=180,180,1350,730; xmax=82000; ymin=2500;ymax=12500
for i in range(6):
 y=bottom-i*(bottom-top)/5;d.line((left,y,right,y),fill=GRID,width=2);d.text((100,y-10),f'{ymax-i*(ymax-ymin)/5:.0f}',font=font(17),fill='#526579')
for i in range(5):
 x=left+i*(right-left)/4;d.line((x,top,x,bottom),fill='#edf1f5',width=1);d.text((x-18,bottom+20),f'{xmax*i/4:.0f}',font=font(17),fill='#526579')
for m in order:
 x=left+int(sm[m]['parameter_count'])/xmax*(right-left);y=top+(float(sm[m]['mse_mean'])-ymin)/(ymax-ymin)*(bottom-top);d.ellipse((x-10,y-10,x+10,y+10),fill=COLORS[m],outline='white',width=2);d.text((x+14,y-16),m,font=font(18,True),fill=COLORS[m])
d.text((550,800),'Trainable parameters',font=font(17),fill=NAVY);d.text((45,160),'MSE (px²)',font=font(17),fill=NAVY)
save(im,'parameter_accuracy')
# Save per-video/track aggregation values
with (OUT/'supporting_data'/'PER_VIDEO_TRACK_SUMMARY.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['model','macro_video_mse_mean_over_runs','macro_track_mse_mean_over_runs','track_clusters','videos'])
 for m in order:w.writerow([m,*macroagg[m],861,13])
print('created quantitative supporting data and figures')
for m in order: print(m, 'macrovideo',round(macroagg[m][0],3),'macrotrack',round(macroagg[m][1],3))


