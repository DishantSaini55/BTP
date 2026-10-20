"""DynMM-style trajectory/context fusion for MTN (Version F)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
 from .context_TF import SceneContextEncoder
 from .individual_TF import IndividualTF
 from .transformer.multihead_attention import MultiHeadAttention
 from .transformer.pointerwise_feedforward import PointerwiseFeedforward
except ImportError:
 from context_TF import SceneContextEncoder
 from individual_TF import IndividualTF
 from transformer.multihead_attention import MultiHeadAttention
 from transformer.pointerwise_feedforward import PointerwiseFeedforward

class FusionSceneContextEncoder(nn.Module):
 """Keeps six native context sources both as tokens and as a global vector."""
 def __init__(self,d_model,embedding_dim=16,dropout=.1):
  super().__init__()
  try: from .context_schema import SCENE_VOCAB_SIZES,OCCLUSION_VOCAB_SIZE,OCCLUSION_UNKNOWN_VALUE
  except ImportError: from context_schema import SCENE_VOCAB_SIZES,OCCLUSION_VOCAB_SIZE,OCCLUSION_UNKNOWN_VALUE
  self.sizes=SCENE_VOCAB_SIZES;self.occ_unknown=OCCLUSION_UNKNOWN_VALUE
  self.scene_embeddings=nn.ModuleList(nn.Embedding(n,embedding_dim) for n in SCENE_VOCAB_SIZES);self.occlusion_embedding=nn.Embedding(OCCLUSION_VOCAB_SIZE,embedding_dim)
  self.token_projection=nn.Linear(embedding_dim,d_model)
  self.global_projection=nn.Sequential(nn.LayerNorm(6*embedding_dim),nn.Linear(6*embedding_dim,d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model,d_model))
 def forward(self,scene,occlusion):
  pieces=[emb(scene[:,i].long().clamp(0,self.sizes[i]-1)) for i,emb in enumerate(self.scene_embeddings)]
  pieces.append(self.occlusion_embedding(occlusion.long().clamp(0,self.occ_unknown)).mean(1))
  raw=torch.stack(pieces,1) # [B,6,16]
  return self.token_projection(raw),self.global_projection(raw.flatten(1))

class DynamicMultimodalFusion(nn.Module):
 """Per-sample hard selection among identity, additive, and cross-attention fusion."""
 def __init__(self,d_model=32,h=4,d_ff=128,dropout=.1,gate_mode='hard'):
  super().__init__();self.gate_mode=gate_mode
  self.add_norm=nn.LayerNorm(d_model)
  self.cross_attn=MultiHeadAttention(h,d_model,dropout)
  self.cross_norm=nn.LayerNorm(d_model);self.cross_ff=PointerwiseFeedforward(d_model,d_ff,dropout);self.cross_ff_norm=nn.LayerNorm(d_model)
  self.gate=nn.Sequential(nn.Linear(2*d_model,d_model),nn.GELU(),nn.Linear(d_model,3))
  self.dropout=nn.Dropout(dropout);self.last_gate=None;self.last_shapes={}
 def forward(self,trajectory,context,global_context):
  # trajectory [B,T,D], context [B,6,D]; gate is per sample, not per token.
  if trajectory.ndim!=3 or context.ndim!=3 or trajectory.shape[0]!=context.shape[0] or trajectory.shape[2]!=context.shape[2] or context.shape[1]!=6: raise ValueError('Expected trajectory [B,T,D] and six context tokens [B,6,D].')
  identity=trajectory
  additive=self.add_norm(trajectory+global_context.unsqueeze(1).expand(-1,trajectory.shape[1],-1))
  cross=self.cross_norm(trajectory+self.dropout(self.cross_attn(trajectory,context,context)))
  cross=self.cross_ff_norm(cross+self.dropout(self.cross_ff(cross)))
  logits=self.gate(torch.cat((trajectory.mean(1),global_context),dim=-1))
  if self.gate_mode=='soft': weights=logits.softmax(-1)
  elif self.training: weights=F.gumbel_softmax(logits,tau=1.,hard=True,dim=-1)
  else: weights=F.one_hot(logits.argmax(-1),3).to(trajectory.dtype)
  out=sum(weights[:,i,None,None]*op for i,op in enumerate((identity,additive,cross)))
  self.last_gate=weights.detach();self.last_shapes={'trajectory':tuple(trajectory.shape),'context':tuple(context.shape),'identity':tuple(identity.shape),'additive':tuple(additive.shape),'cross_attention':tuple(cross.shape),'output':tuple(out.shape),'gate_logits':tuple(logits.shape)}
  return out

class FusionContextIndividualTF(nn.Module):
 """Version F: MTN + native context encoder + genuine DynMM-style fusion."""
 def __init__(self,inp_l,enc_inp_size,dec_inp_size,dec_out_size,context_dim=16,gate_mode='hard',**options):
  super().__init__();self.base=IndividualTF(inp_l,enc_inp_size,dec_inp_size,dec_out_size,**options);d=options.get('d_model',512)
  self.context_encoder=FusionSceneContextEncoder(d,context_dim,options.get('dropout',.1));self.fusion=DynamicMultimodalFusion(d,options.get('h',8),options.get('d_ff',2048),options.get('dropout',.1),gate_mode)
 def forward(self,residual,inp,obd_spd,ego_flow,ped_flow,tgt,src_mask,obd_mask,spd_mask,ped_mask,scene_context,observed_occlusion):
  ctx,global_ctx=self.context_encoder(scene_context,observed_occlusion)
  speed=self.base.model.enc_extra_embed_I(obd_spd.permute(0,2,1));memory,mix=self.base.model.encode(inp,speed,src_mask,obd_mask)
  fused=self.fusion(memory,ctx,global_ctx)
  ego=self.base.model.enc_extra_embed_II(ego_flow.permute(0,2,1));ped=self.base.model.enc_extra_embed_III(ped_flow.permute(0,2,1))
  out=self.base.model.decode(tgt,fused,mix,ego,ped,src_mask,spd_mask,ped_mask);return self.base.model.generator(out)+residual
 def forward_batch(self,batch):
  boxes=batch.get('boxes',batch.get('enc_input')).float();speed=batch.get('speed',batch.get('obd_speed')).float();ego=batch['ego_flow'].float();ped=batch['ped_flow'].float();B,T=boxes.shape[:2]
  if ego.ndim==4: ego=ego[:,:T-1].permute(0,1,3,2).reshape(B,2*(T-1),-1)
  if ped.ndim==4: ped=ped[:,:T-1].permute(0,1,3,2).reshape(B,2*(T-1),-1)
  L=batch.get('predict_length',45);res=boxes[:,-1:,:].repeat(1,L,1);tgt=torch.zeros_like(res);dev=boxes.device
  return self(res,boxes,speed[:,:T],ego,ped,tgt,torch.ones(B,1,T,device=dev),torch.ones(B,1,1,device=dev),torch.ones(B,1,ego.shape[-1],device=dev),torch.ones(B,1,ped.shape[-1],device=dev),batch['scene_context'].to(dev),batch['observed_occlusion'].to(dev))
