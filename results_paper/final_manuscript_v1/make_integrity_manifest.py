from pathlib import Path
import hashlib,csv,zipfile,sys
root=Path.cwd();out=root/'results_paper/final_manuscript_v1';manifest=out/'PROVENANCE_AND_INTEGRITY_MANIFEST.csv'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
# Verify the actual zipped source is the exact final source.
with zipfile.ZipFile(out/'OVERLEAF_SOURCE.zip') as z:
 for rel in ['main.tex','references.bib']:
  local=(out/rel).read_bytes(); packaged=z.read(rel)
  if hashlib.sha256(local).digest()!=hashlib.sha256(packaged).digest():
   raise SystemExit(f'ZIP stale source: {rel}')
source_rel=[
'results_paper/AO_confirmation_v1/OFFICIAL_TEST_RESULTS_v2.json',
'results_paper/AO_confirmation_v1/OFFICIAL_TEST_RESULTS_v2/OFFICIAL_TEST_SEED_RESULTS.csv',
'results_paper/AO_confirmation_v1/OFFICIAL_TEST_RESULTS_v2/OFFICIAL_TEST_PER_VIDEO.csv',
'results_paper/AO_confirmation_v1/OFFICIAL_TEST_RESULTS_v2/OFFICIAL_TEST_PER_TRACK.csv',
'results_paper/AO_confirmation_v1/OFFICIAL_TEST_RESULTS_v2/OFFICIAL_TEST_PER_WINDOW.csv',
'results_paper/AO_confirmation_v1/OFFICIAL_TEST_MODEL_SUMMARY_v3.csv',
'results_paper/AO_confirmation_v1/OFFICIAL_TEST_PAIRED_COMPARISONS_v3.csv',
'results_paper/AO_confirmation_v1/OFFICIAL_TEST_TRACK_BOOTSTRAP_v3.csv',
'results_paper/AO_confirmation_v1/OFFICIAL_TEST_MACRO_RESULTS.csv',
'results_paper/AO_confirmation_v1/OFFICIAL_TEST_FINAL_REPORT_v3.md',
'results_paper/PHASE_6B_EXECUTION_AND_INTEGRITY_ADDENDUM_v1.md',
'results_paper/FINAL_CHECKPOINT_HASHES.csv',
'results_paper/phase_6c_literature_v1/VERIFIED_REFERENCE_DATABASE.csv',
'results_paper/phase_6c_literature_v1/PHASE_6C_FINAL_REPORT.md',
'results_paper/phase_6d_craft_stage1_v1/STAGE1_REPORT.md',
'results_paper/phase_6d_craft_stage2b_v1/STAGE2B_REPORT.md',
'results_paper/phase_6d_craft_stage3_v1/CRAFT_FINAL_EXPERIMENT_REPORT.md',
'results_paper/qualitative/HIVE_DESIGN.md']
rows=[]
for rel in source_rel:
 p=root/rel
 if p.is_file(): rows.append(('canonical_input',rel,p.stat().st_size,sha(p)))
 else: print('source not present for hash manifest:',rel)
for p in sorted(out.rglob('*')):
 if p.is_file() and p!=manifest:
  rel=p.relative_to(root).as_posix(); rows.append(('delivery_artifact',rel,p.stat().st_size,sha(p)))
with manifest.open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['role','path','size_bytes','sha256']);w.writerows(rows)
print('manifest rows',len(rows))
print('main.tex sha',sha(out/'main.tex'))
print('final pdf sha',sha(out/'FINAL_RESEARCH_PAPER.pdf'))
print('overleaf zip sha',sha(out/'OVERLEAF_SOURCE.zip'))

