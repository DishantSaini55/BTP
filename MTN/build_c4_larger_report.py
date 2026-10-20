from pathlib import Path
import json, numpy as np, matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans'})
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(r'D:\IDD-PeD-Evaluation\outputs\weight_sharing_targeted');R=json.loads((ROOT/'C4_larger_validation_results.json').read_text());O=ROOT/'C4_Larger_Scale_Validation_Report.pdf';N='#102A43';T='#00A6A6';B='#2F80ED';P='#F7FAFC';RED='#D64545'
def pg(t,s):
 f=plt.figure(figsize=(11.69,8.27));f.text(.055,.93,t,fontsize=20,fontweight='bold',color=N);f.text(.055,.892,s,fontsize=9,color='#52616B');f.add_artist(plt.Line2D([.055,.945],[.875,.875],color=T,lw=2));return f
def tx(f,x,y,s,z=10,w='normal',c='#263238'):f.text(x,y,s,fontsize=z,fontweight=w,color=c,va='top')
def tab(ax,c,r,w,fs=8):
 ax.axis('off');t=ax.table(cellText=r,colLabels=c,loc='center',cellLoc='left',colLoc='left',colWidths=w);t.auto_set_font_size(False);t.set_fontsize(fs);t.scale(1,1.45)
 for (i,j),q in t.get_celld().items():
  q.set_edgecolor('#D9E2EC');q.set_facecolor(N if i==0 else (P if i%2==0 else 'white'))
  if i==0:q.get_text().set_color('white');q.get_text().set_weight('bold')
def ms(x):return 1000*x
with PdfPages(O) as pdf:
 f=pg('C4 Larger-Scale CPU Validation','Three-seed matched ablation • CPU Farneback flow • not an exact reproduction of the authors’ RAFT pipeline')
 tx(f,.055,.82,'Executive summary',14,'bold',N);tx(f,.055,.785,'C4 reduces trainable parameters by 6.07% in every run. Across three seeds and all 24 locally available held-out\ntrajectories, C4 had lower mean MSE (−5.01%), but the seed-to-seed spread is substantial and it was slower on\naverage CPU inference (+10.14%). This supports further testing, not final architecture selection.',11,'bold',T)
 tx(f,.055,.64,'Exact C4 definition',13,'bold',N);tx(f,.055,.602,'Only EncoderLayer.self_attn Q/K/V/O is shared by object identity across Encoder.layers[0..2]. Context branch,\nencoder src_attn/cross_attn/FFNs, decoder, flows, loss and output head are unchanged.',10.3)
 tx(f,.055,.49,'Fixed experiment',13,'bold',N);c=R['configuration'];tx(f,.055,.452,f"Train: {c['train_trajectories']} trajectories, gp_set_0006 video 2. Validation: {c['validation_trajectories']} trajectories, video 1.\nTest: ALL {c['test_trajectories']} eligible trajectories from separately held-out video 3, source order, no outcome selection.\nSeeds: {', '.join(map(str,c['seeds']))}; Adam 1e-3; batch 8; MSE; {c['epochs']} epoch; same official initialization per B/C4 pair.",10.2)
 tx(f,.055,.30,'Simple architecture view',13,'bold',N);tx(f,.055,.26,'B:  Encoder 1 → self-attention A | Encoder 2 → self-attention B | Encoder 3 → self-attention C\nC4: Encoder 1 ─┐\n    Encoder 2 ─┼→ SAME self-attention Q/K/V/O parameters\n    Encoder 3 ─┘',11,'bold',T);tx(f,.055,.12,'All three attention operations still execute. Parameter reduction does not by itself imply fewer FLOPs.',9.5,'normal',RED);pdf.savefig(f);plt.close(f)
 f=pg('Parameters, sharing verification, and multi-seed metrics','Lower MSE / C-MSE / CF-MSE / latency is better. Mean ± sample standard deviation across three seeds.')
 a=R['aggregate'];ch=R['changes_C4_minus_B'];rows=[]
 for name in ['B','C4']:
  q=a[name];rows.append([name,f"{q['parameters']:,}",f"{q['mse']['mean']:.2f} ± {q['mse']['std']:.2f}",f"{q['cmse']['mean']:.2f} ± {q['cmse']['std']:.2f}",f"{q['cfmse']['mean']:.2f} ± {q['cfmse']['std']:.2f}",f"{ms(q['latency_seconds']['mean']):.2f} ± {ms(q['latency_seconds']['std']):.2f}",f"{q['checkpoint_bytes']['mean']/1048576:.2f}"])
 ax=f.add_axes([.03,.57,.94,.20]);tab(ax,['Model','Parameters','MSE','C-MSE','CF-MSE','Latency ms','Checkpoint MiB'],rows,[.09,.12,.17,.17,.17,.17,.12],7.5)
 tx(f,.055,.48,f"C4 minus B: parameters {ch['parameters']['percent']:+.2f}%; MSE {ch['mse']['percent']:+.2f}%; C-MSE {ch['cmse']['percent']:+.2f}%;\nCF-MSE {ch['cfmse']['percent']:+.2f}%; measured latency {ch['latency_seconds']['percent']:+.2f}%.",11,'bold',T)
 v=R['sharing_verification'];tx(f,.055,.36,'Sharing check: Q, K, V and O passed module identity, weight/bias object identity and storage-pointer equality across\nall three EncoderLayers. Optimizer check: no duplicate independent parameter objects.',10.3,'bold',N)
 tx(f,.055,.25,'Computational interpretation',13,'bold',N);tx(f,.055,.213,'The three encoder self-attention calls retain the same shapes and sequence length, so theoretical operation count/FLOPs\nremain approximately unchanged. Checkpoint size falls because tied weights are stored once. CPU latency is measured,\nnot inferred, and was not reduced in this three-seed validation.',10,'normal',RED);pdf.savefig(f);plt.close(f)
 f=pg('Per-seed and sample-wise evidence','Each B/C4 pair used the same data, seed, checkpoint initialization protocol, flow cache, optimizer, and epoch count.')
 rows=[]
 for s in c['seeds']:
  d=R['seeds'][str(s)];rows.append([str(s),f"{d['B']['metrics']['mse']:.2f}",f"{d['C4']['metrics']['mse']:.2f}",f"{d['C4']['metrics']['mse']-d['B']['metrics']['mse']:+.2f}",f"{ms(d['B']['latency']['mean_seconds']):.2f}",f"{ms(d['C4']['latency']['mean_seconds']):.2f}"])
 ax=f.add_axes([.12,.60,.76,.18]);tab(ax,['Seed','B MSE','C4 MSE','C4−B','B latency ms','C4 latency ms'],rows,[.14,.17,.17,.16,.19,.19],8.5)
 pairs=R['sample_comparison'];cnt=R['sample_counts'];tx(f,.055,.49,f"Across all 72 paired observations (24 trajectories × 3 seeds): C4 better {cnt['C4_better']} ({100*cnt['C4_better']/cnt['total']:.1f}%); B better {cnt['B_better']} ({100*cnt['B_better']/cnt['total']:.1f}%); ties {cnt['ties']}.",11,'bold',T)
 ax=f.add_axes([.10,.16,.80,.22]);x=np.arange(len(pairs));ax.plot(x,[p['B_mse'] for p in pairs],'.',color=B,label='B');ax.plot(x,[p['C4_mse'] for p in pairs],'.',color=T,label='C4');ax.set_xlabel('Paired held-out trajectory observations (seed-major)');ax.set_ylabel('MSE');ax.grid(alpha=.25);ax.legend();pdf.savefig(f);plt.close(f)
 f=pg('Interpretation and recommended next experiment','Evidence-based conclusion, limited to the current CPU/Farneback validation.')
 tx(f,.055,.82,'Answers',13,'bold',N);tx(f,.055,.78,'1. Parameters? Yes—C4 consistently reduces them by 8,448 (6.07%).\n2. Prediction performance? Mean metrics favor C4, but variation is large; evidence is suggestive, not conclusive.\n3. Inference computation/FLOPs? No meaningful theoretical reduction: all three attention executions remain.\n4. Measured latency? No—C4 was slower on average in this run (+10.14%), with high variability.\n5. Consistency across samples/videos? C4 won 56.9% of paired samples; only one held-out video is locally available.\n6. Final architecture recommendation? Not yet. Retain C4 as a candidate for larger evaluation.',10.5)
 tx(f,.055,.50,'Limitations',13,'bold',N);tx(f,.055,.465,'One held-out video, 24 trajectories, three seeds, one training epoch, CPU-only timing, and Farneback instead of RAFT.\nThe larger test set reduces the earlier 10-sample limitation, but does not establish cross-video generalization.',10.2,'normal',RED)
 tx(f,.055,.35,'Recommended next step',13,'bold',N);tx(f,.055,.315,'Add independently held-out videos when verified data are available; retain video-level separation; use the same three-or-more\nseeds with a pre-declared multi-epoch budget; report uncertainty across videos and seeds. Keep C4 unchanged.',10.5,'bold',T)
 tx(f,.055,.17,'Machine-readable artifacts: C4_larger_validation_results.json, C4_larger_validation_summary.csv, and\nC4_larger_validation_config_manifest.json. They preserve every per-trajectory paired result and selection rule.',9.5,'normal','#52616B');pdf.savefig(f);plt.close(f)
print(O)
