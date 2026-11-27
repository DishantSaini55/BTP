# Annotation-only architecture comparison

All learned models consume only observed bounding boxes and output 45 future boxes. They are separate implementations; none is described as an exact reproduction of full-input MTN.

| ID | Model | Anchor | Encoder self-attention | Scene context |
|---|---|---|---|---|
| AO-0 | Deterministic constant velocity | Last-three-observed-box constant-velocity extrapolation | None | None |
| AO-1 | Annotation-only transformer | Constant-position residual | Unshared | None |
| AO-2 | Matched annotation-only transformer | Constant-velocity residual | Unshared | None |
| AO-3 | AO-2 with SATIE-style sharing | Constant-velocity residual | Same Q/K/V/O module shared by encoder layers 1–3 | None |
| AO-4 | Skipped | — | — | GP7 video 0005 has no POI scene labels; prediction-time availability is unverified |

The learned models use three encoder and three decoder layers, `d_model=32`, `d_ff=128`, four attention heads, and dropout 0.1. Decoder layers cross-attend to trajectory memory and apply a feed-forward block; they do not add target self-attention. A single residual anchor is added to the learned correction.

AO-1's anchor is the last observed box repeated. AO-2/AO-3 and AO-0 use the same coordinate-wise formula `b[T+k] = b[T] + k * (b[T] - b[T-2]) / 2`, for `k=1..45`.

Measured unique trainable parameter counts are AO-0: 0, AO-1: 76,804, AO-2: 76,804, and AO-3: 68,356. Sharing the encoder attention module reduces AO-3 by 8,448 parameters; this count deduplicates shared parameter objects.

The historical full-input MTN uses speed and optical-flow streams alongside trajectory features, and its decoder attends to the trajectory and flow memories. The AO variants instead embed only observed boxes into a trajectory memory and cross-attend decoder queries to that memory. They are trained from scratch on the new split; they do not feed synthetic zeros to unavailable flow inputs. AO-1 uses the constant-position residual; AO-2 and AO-3 use a constant-velocity residual, so those models intentionally differ from the historical residual behavior as well.

AO-3 shares only encoder self-attention; decoder attention, feed-forward blocks, normalization, embeddings, and output layers remain independent. Actual unique trainable parameter counts are recorded by the runner for every configuration.

See `MTN/ao_models_v1.py` for the versioned implementation and `tests/test_ao_models_v1.py` for shape, residual, gradient, and parameter-identity checks.
