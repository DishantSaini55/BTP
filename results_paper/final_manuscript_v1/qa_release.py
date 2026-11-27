from pathlib import Path
import csv,hashlib,zipfile,re,sys
root=Path.cwd();out=root/'results_paper/final_manuscript_v1';src=root/'results_paper/AO_confirmation_v1';fail=[]
def readcsv(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
summary={r['model_id']:r for r in readcsv(src/'OFFICIAL_TEST_MODEL_SUMMARY_v3.csv')}
expected={'AO-0':3997.2373046875,'AO-1':8516.52294921875,'AO-2':3975.058349609375,'AO-3':3962.072998046875,'AO-RIDGE':2889.71826171875,'AO-CP':11558.828125}
for m,v in expected.items():
 if abs(float(summary[m]['mse_mean'])-v)>0.001:fail.append(f'metric mismatch {m}')
seed=readcsv(src/'OFFICIAL_TEST_RESULTS_v2/OFFICIAL_TEST_SEED_RESULTS.csv')
pervideo=readcsv(src/'OFFICIAL_TEST_RESULTS_v2/OFFICIAL_TEST_PER_VIDEO.csv')
pertrack=readcsv(src/'OFFICIAL_TEST_RESULTS_v2/OFFICIAL_TEST_PER_TRACK.csv')
perwindow=readcsv(src/'OFFICIAL_TEST_RESULTS_v2/OFFICIAL_TEST_PER_WINDOW.csv')
macro=readcsv(src/'OFFICIAL_TEST_MACRO_RESULTS.csv')
if len(seed)!=18:fail.append('seed records not 18')
if len(pervideo)!=18*13:fail.append(f'per-video rows {len(pervideo)}')
if len(pertrack)!=18*861:fail.append(f'per-track rows {len(pertrack)}')
if len(perwindow)!=18*2615:fail.append(f'per-window rows {len(perwindow)}')
if len(macro)!=18:fail.append(f'macro rows {len(macro)}')
tex=(out/'main.tex').read_text(encoding='utf-8-sig'); log=(out/'compile_release4.log').read_text(encoding='utf-8',errors='replace')
for token in ['3997.237','8516.523','3975.058','3962.073','2889.718','11558.828','27.707\\%','0.175750','0.371239','0.0027753']:
 if token not in tex:fail.append('missing manuscript token '+token)
if 'undefined citation' in log or 'undefined references' in log or re.search(r'Overfull \\hbox',log):fail.append('compile log warning gate failed')
for p in [out/'FINAL_RESEARCH_PAPER.pdf',out/'OVERLEAF_SOURCE.zip']:
 if not p.is_file() or p.stat().st_size==0:fail.append('missing output '+str(p))
with zipfile.ZipFile(out/'OVERLEAF_SOURCE.zip') as z:
 names=set(z.namelist())
 for required in ['main.tex','references.bib','figures/horizon_mse.png','diagrams/cv_anchored_transformer.svg','diagrams/architecture_diagrams.pdf']:
  if required not in names:fail.append('missing zip member '+required)
v2sha=sha(src/'OFFICIAL_TEST_RESULTS_v2.json')
expectedsha='6a96d14b857452a454ceeab76954bdb5f9dfc15d11de18c4ff6500e7d8558b03'
if v2sha!=expectedsha:fail.append('canonical v2 hash differs from addendum')
print('v2_sha256',v2sha)
print('record_rows',{'seed':len(seed),'per_video':len(pervideo),'per_track':len(pertrack),'per_window':len(perwindow),'macro':len(macro)})
print('pdf_bytes',(out/'FINAL_RESEARCH_PAPER.pdf').stat().st_size,'zip_bytes',(out/'OVERLEAF_SOURCE.zip').stat().st_size)
print('status','FAIL' if fail else 'PASS')
for x in fail:print('FAIL:',x)
if fail:sys.exit(1)
