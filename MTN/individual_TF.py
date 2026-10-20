import torch
import torch.nn as nn
try:
    from .transformer.decoder import Decoder, SharedDecoder
    from .transformer.multihead_attention import MultiHeadAttention
    from .transformer.positional_encoding import PositionalEncoding
    from .transformer.pointerwise_feedforward import PointerwiseFeedforward
    from .transformer.encoder_decoder import EncoderDecoder
    from .transformer.encoder import Encoder, SharedEncoder
    from .transformer.encoder_layer import EncoderLayer
    from .transformer.decoder_layer import DecoderLayer
except ImportError:  # Supports the original "run from MTN" scripts.
    from transformer.decoder import Decoder, SharedDecoder
    from transformer.multihead_attention import MultiHeadAttention
    from transformer.positional_encoding import PositionalEncoding
    from transformer.pointerwise_feedforward import PointerwiseFeedforward
    from transformer.encoder_decoder import EncoderDecoder
    from transformer.encoder import Encoder, SharedEncoder
    from transformer.encoder_layer import EncoderLayer
    from transformer.decoder_layer import DecoderLayer
import copy
import math


class IndividualTF(nn.Module):
    def __init__(self, inp_l,enc_inp_size, dec_inp_size, dec_out_size, N=1,
                   d_model=512, d_ff=2048, h=8, dropout=0.1, share_layers=False,
                   share_encoder=False, share_decoder=False):
        super(IndividualTF, self).__init__()
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        generator=Generator(d_model, dec_out_size)

        if share_layers:
            share_encoder = share_decoder = True
        encoder_layer = EncoderLayer(d_model, c(attn),c(attn),c(attn),c(ff), dropout)
        decoder_layer = DecoderLayer(d_model, c(attn), c(attn),c(attn), c(ff), dropout)
        encoder = SharedEncoder(encoder_layer, N) if share_encoder else Encoder(encoder_layer, N)
        decoder = SharedDecoder(decoder_layer, N) if share_decoder else Decoder(decoder_layer, N)
        self.model = EncoderDecoder(
            encoder, decoder,
            nn.Sequential(LinearEmbedding(enc_inp_size,d_model), c(position)),
            nn.Sequential(LinearEmbedding(dec_inp_size,d_model),c(position)),
            nn.Sequential(LinearEmbedding_sp(inp_l,d_model)),
            nn.Sequential(LinearEmbedding_sp(2*(inp_l-1),d_model)),
            nn.Sequential(LinearEmbedding_sp(2*(inp_l-1),d_model)),
            generator)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,residual,inp,obd_spd,ego_flow,ped_flow,tgt,src_mask,obd_enc_mask,spd_mask,ped_mask):
        output=self.model(inp,obd_spd,ego_flow,ped_flow,tgt,src_mask,obd_enc_mask,spd_mask,ped_mask)
        out_lane=self.model.generator(output)
        output=out_lane+residual
        return output

    def forward_batch(self, batch):
        """Run the shared MTN batch contract without consuming future inputs."""
        boxes = batch.get("boxes", batch.get("enc_input")).float()
        speed = batch.get("speed", batch.get("obd_speed")).float()
        ego_flow = batch["ego_flow"].float()
        ped_flow = batch["ped_flow"].float()
        predict_length = batch.get("predict_length", 45)
        observe_length = boxes.shape[1]
        device = boxes.device
        batch_size = boxes.shape[0]
        if ego_flow.ndim == 4:
            ego_flow = ego_flow[:, :observe_length - 1].permute(0, 1, 3, 2)
            ego_flow = ego_flow.reshape(batch_size, 2 * (observe_length - 1), -1)
        if ped_flow.ndim == 4:
            ped_flow = ped_flow[:, :observe_length - 1].permute(0, 1, 3, 2)
            ped_flow = ped_flow.reshape(batch_size, 2 * (observe_length - 1), -1)
        residual = boxes[:, -1:, :].repeat(1, predict_length, 1)
        target = torch.zeros_like(residual)
        src_mask = torch.ones(batch_size, 1, observe_length, device=device)
        obd_mask = torch.ones(batch_size, 1, 1, device=device)
        ego_mask = torch.ones(batch_size, 1, ego_flow.shape[-1], device=device)
        ped_mask = torch.ones(batch_size, 1, ped_flow.shape[-1], device=device)
        return self(residual, boxes, speed[:, :observe_length], ego_flow, ped_flow,
                    target, src_mask, obd_mask, ego_mask, ped_mask)

class LinearEmbedding_sp(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding_sp, self).__init__()
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x)

class LinearEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, out_size):
        super(Generator, self).__init__()
        self.proj=nn.Linear(d_model,out_size)

    def forward(self, x):
        return self.proj(x)
