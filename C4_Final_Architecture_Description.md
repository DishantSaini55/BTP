# C4 Final POC Architecture

## Simple Hinglish

MTN observed pedestrian boxes ko encode karta hai. Context branch road/intersection type, signal status, location, motion direction aur observed occlusion ko 16-D embeddings mein convert karta hai. In six representations ko concatenate karke 96-D context feature banaya jata hai, phir 32-D context token aur original admission gate se decoder memory mein add kiya jata hai. C4 mein sirf encoder self-attention ke Q/K/V/O weights teen encoder layers mein same hain; baaki encoder, decoder aur context modules alag hi rehte hain.

## Technical description

Input boxes `[B,15,4]` are embedded to `[B,15,32]` and processed by `Encoder.layers[0..2]`. C4 aliases only `self_attn.linears[0..3]` (Q/K/V/O) across these layers. The encoder returns trajectory memory `[B,15,32]`. Five static context labels plus observed occlusion yield six 16-D embeddings, concatenated to `[B,96]` and projected to `[B,32]`. The original B admission gate creates a gated context token `[B,1,32]`, appended to produce augmented decoder memory `[B,16,32]`. The unchanged MTN decoder and output head predict `[B,45,4]`.

## Scope

C4 reduces parameters by 6.07% in the unified CPU Farneback POC. It does not reduce FLOPs because all three encoder self-attention executions remain. Measured latency was +3.93% versus B. This is a POC architecture, not a final benchmark claim.
