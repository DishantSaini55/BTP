"""Context-aware MTN for the IDD-PeD scene-information ablation.

The model is deliberately an additive branch over the unchanged MTN baseline:
scene metadata creates one memory token and a DynMM-inspired hard gate decides
per sample whether that token is available to decoder cross-attention.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .context_schema import (OCCLUSION_UNKNOWN_VALUE, OCCLUSION_VOCAB_SIZE,
                                 SCENE_FIELDS, SCENE_VOCAB_SIZES)
    from .individual_TF import IndividualTF
except ImportError:  # Supports original scripts run from MTN/.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from context_schema import (OCCLUSION_UNKNOWN_VALUE, OCCLUSION_VOCAB_SIZE,
                                SCENE_FIELDS, SCENE_VOCAB_SIZES)
    from individual_TF import IndividualTF


class SceneContextEncoder(nn.Module):
    """Encodes static scene labels and observed occlusion labels only."""

    def __init__(self, d_model, embedding_dim=16, dropout=0.1):
        super().__init__()
        self.scene_embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim) for size in SCENE_VOCAB_SIZES
        )
        self.occlusion_embedding = nn.Embedding(OCCLUSION_VOCAB_SIZE, embedding_dim)
        self.projection = nn.Sequential(
            nn.LayerNorm(embedding_dim * (len(SCENE_FIELDS) + 1)),
            nn.Linear(embedding_dim * (len(SCENE_FIELDS) + 1), d_model),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )

    def forward(self, scene_context, observed_occlusion):
        if scene_context.ndim != 2 or scene_context.shape[1] != len(SCENE_FIELDS):
            raise ValueError(
                f"scene_context must have shape [batch, {len(SCENE_FIELDS)}] in "
                f"this order: {', '.join(SCENE_FIELDS)}"
            )
        if observed_occlusion.ndim != 2:
            raise ValueError("observed_occlusion must have shape [batch, observed_steps]")
        scene_tokens = []
        for index, embedding in enumerate(self.scene_embeddings):
            values = scene_context[:, index].long().clamp(
                min=0, max=SCENE_VOCAB_SIZES[index] - 1
            )
            scene_tokens.append(embedding(values))
        occlusion = observed_occlusion.long().clamp(
            min=0, max=OCCLUSION_UNKNOWN_VALUE
        )
        # Static labels stay distinct; mean pooling summarizes visibility history.
        occlusion_token = self.occlusion_embedding(occlusion).mean(dim=1)
        return self.projection(torch.cat(scene_tokens + [occlusion_token], dim=-1))


class ContextIndividualTF(nn.Module):
    """MTN plus an adaptively admitted scene-context memory token."""

    def __init__(self, inp_l, enc_inp_size, dec_inp_size, dec_out_size,
                 context_dim=16, gate_mode="hard", **model_options):
        super().__init__()
        if gate_mode not in {"hard", "soft"}:
            raise ValueError("gate_mode must be 'hard' or 'soft'")
        self.base = IndividualTF(
            inp_l, enc_inp_size, dec_inp_size, dec_out_size, **model_options
        )
        d_model = model_options.get("d_model", 512)
        self.context_encoder = SceneContextEncoder(
            d_model, context_dim, model_options.get("dropout", 0.1)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model + enc_inp_size, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )
        self.gate_mode = gate_mode
        self.last_context_gate = None
        self._context_gate_for_loss = None

    def _context_gate(self, context_token, observed_boxes):
        logits = self.gate(torch.cat((context_token, observed_boxes[:, -1]), dim=-1))
        if self.gate_mode == "soft":
            use_context = logits.softmax(dim=-1)[:, 1]
        elif self.training:
            use_context = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)[:, 1]
        else:
            use_context = logits.argmax(dim=-1).eq(1).to(context_token.dtype)
        self.last_context_gate = use_context.detach()
        self._context_gate_for_loss = use_context
        return use_context

    def context_cost_loss(self):
        """Expected auxiliary-branch use, for DynMM resource regularisation."""
        if self._context_gate_for_loss is None:
            raise RuntimeError("Run a forward pass before requesting context_cost_loss().")
        return self._context_gate_for_loss.mean()

    def forward(self, residual, inp, obd_spd, ego_flow, ped_flow, tgt,
                src_mask, obd_enc_mask, spd_mask, ped_mask, scene_context,
                observed_occlusion):
        context_token = self.context_encoder(scene_context, observed_occlusion)
        gate = self._context_gate(context_token, inp)
        output = self.base.model(
            inp, obd_spd, ego_flow, ped_flow, tgt, src_mask, obd_enc_mask,
            spd_mask, ped_mask, context=context_token * gate.unsqueeze(-1),
            context_mask=gate.view(-1, 1, 1),
        )
        return self.base.model.generator(output) + residual

    def forward_batch(self, batch):
        boxes = batch.get("boxes", batch.get("enc_input")).float()
        speed = batch.get("speed", batch.get("obd_speed")).float()
        ego_flow = batch["ego_flow"].float()
        ped_flow = batch["ped_flow"].float()
        scene_context = batch["scene_context"].to(boxes.device)
        observed_occlusion = batch["observed_occlusion"].to(boxes.device)
        predict_length = batch.get("predict_length", 45)
        observe_length, batch_size = boxes.shape[1], boxes.shape[0]
        if ego_flow.ndim == 4:
            ego_flow = ego_flow[:, :observe_length - 1].permute(0, 1, 3, 2)
            ego_flow = ego_flow.reshape(batch_size, 2 * (observe_length - 1), -1)
        if ped_flow.ndim == 4:
            ped_flow = ped_flow[:, :observe_length - 1].permute(0, 1, 3, 2)
            ped_flow = ped_flow.reshape(batch_size, 2 * (observe_length - 1), -1)
        residual = boxes[:, -1:, :].repeat(1, predict_length, 1)
        target = torch.zeros_like(residual)
        src_mask = torch.ones(batch_size, 1, observe_length, device=boxes.device)
        obd_mask = torch.ones(batch_size, 1, 1, device=boxes.device)
        ego_mask = torch.ones(batch_size, 1, ego_flow.shape[-1], device=boxes.device)
        ped_mask = torch.ones(batch_size, 1, ped_flow.shape[-1], device=boxes.device)
        return self(residual, boxes, speed[:, :observe_length], ego_flow, ped_flow,
                    target, src_mask, obd_mask, ego_mask, ped_mask,
                    scene_context, observed_occlusion)
