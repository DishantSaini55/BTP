"""Matched CPU POC runner for Version F and F+C4; uses existing fixed 32/8/24 split."""
import json,torch
from torch.utils.data import DataLoader
import cpu_c4_larger_validation as large
import cpu_targeted_weight_sharing as common
from dynamic_fusion_TF import FusionContextIndividualTF
from c4_encoder_shared import make_c4_encoder_self_attention_shared
def fnew(strategy):
 m=FusionContextIndividualTF(15,4,4,4,N=3,d_model=32,d_ff=128,h=4,dropout=.1)
 o=torch.load(common.OFFICIAL,map_location='cpu');s=m.state_dict();n=0
 for k in s:
  q=k.removeprefix('base.')
  if q in o and o[q].shape==s[k].shape:s[k]=o[q];n+=1
 m.load_state_dict(s)
 if strategy=='encoder_self_attn':make_c4_encoder_self_attention_shared(m)
 return m,n
def main():
 common.new=fnew
 tr=large.load(common.FLOW/'train.pkl');va=large.load(common.FLOW/'validation.pkl');te=large.load(large.FLOW_LARGE/'test_full.pkl');manifest=json.loads((large.FLOW_LARGE/'test_full_manifest.json').read_text());mean,std,ms,ss=common.stats(tr);td=common.build(tr,'train',mean,std,ms,ss);vd=common.build(va,'validation',mean,std,ms,ss);ed=common.build(te,'test',mean,std,ms,ss);batches=list(DataLoader(ed,batch_size=1,shuffle=False));out={}
 for seed in large.SEEDS:
  out[str(seed)]={'F':large.run('F','none',seed,td,vd,ed,mean,std,batches,manifest),'F_C4':large.run('F_C4','encoder_self_attn',seed,td,vd,ed,mean,std,batches,manifest)};print(seed,out[str(seed)]['F']['metrics'],out[str(seed)]['F_C4']['metrics'],flush=True)
 (large.OUT/'fusion_validation_results.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
