# MTN Context Weight-Sharing Audit and CPU POC

## Scope

This audit compares the completed **Version B** context-aware MTN with **Version C**, a layer-tied context-aware MTN. It uses the same IDD-PeD 32-train / 8-validation / 10-held-out-test POC split, cached CPU Farneback flow tensors, seed (20260928), optimizer, MSE loss, and 12-epoch training budget. The context branch and input/output contract are unchanged.

This is a CPU proof-of-concept, not a published-benchmark reproduction.

## 1. Current Version B audit

| Item | Actual configuration / count |
|---|---:|
| Total parameters | 139,142 |
| Trainable parameters | 139,142 |
| Encoder total | 64,096 |
| Decoder total | 63,904 |
| Self-attention projections | 12,672 |
| Cross-attention projections | 63,360 |
| FFN layers | 50,112 |
| Context branch (context encoder + gate) | 5,442 |
| Context encoder alone | 4,816 |
| Encoder layers | 3 |
| Decoder layers | 3 |
| Attention modules | 18 (9 encoder + 9 decoder) |
| Attention module size | 4,224 parameters each |
| `d_model` / heads / `d_ff` | 32 / 4 / 128 |

The encoder has three attention operations per layer (`self_attn`, `src_attn`, `cross_attn`). The decoder has three (`ped_attn`, `src_attn`, `spd_attn`). Each stack uses independent copies in Version B. The context branch itself adds no new attention module: it encodes labels into one context token, which is appended to decoder memory and consumed through existing decoder source attention.

The parameter categories `self-attention`, `cross-attention`, and `FFN` above are exclusive ownership groups. Encoder/decoder totals include these groups plus their layer-normalization parameters.

## 2. Sharing strategies considered

| Candidate | Unique parameters | Reduction | Reduction % | Assessment |
|---|---:|---:|---:|---|
| B: current context MTN | 139,142 | 0 | 0.0% | Reference |
| Encoder layers tied | 96,454 | 42,688 | 30.7% | Safe interface; encoder executes 3 times |
| Decoder layers tied | 96,582 | 42,560 | 30.6% | Safe interface; decoder executes 3 times |
| Attention-only, same-role layers tied | 88,454 | 50,688 | 36.4% | Leaves FFNs independent; not selected |
| All attention projections globally tied | 67,334 | 71,808 | 51.6% | Shape-compatible but semantically aggressive; not selected |
| C: encoder + decoder layers tied | 53,894 | 85,248 | 61.3% | Implemented and measured |

The first three sharing counts were instantiated from the actual model. Attention-only totals are calculated from the actual 18 attention modules, each with 4,224 parameters: same-role sharing retains six unique attention modules; global attention tying retains one. The latter can share shapes (`d_model -> d_model`) across encoder/decoder attention projections, but attention roles differ and it was not the safe first experiment.

The context branch remains independent in Version C. Its 96-to-32 context projection and 32-to-32 projection are not semantically interchangeable with MTN Q/K/V/O projections. Tying them would conflate scene-token formation with attention projections, so it was not selected.

## 3. Selected implementation: Version C

Version C has one encoder layer and one decoder layer. Each is executed three times with the original loop structure, sharing its own parameters across depth. The model still has three encoder and three decoder **executions**. It keeps the same context labels, gate, context token, flow tensors, prediction target, and output shape as B.

For C, 94 compatible official MTN tensors were loaded. Tied layers are initialized from official layer 0 tensors; the context branch is then trained during the same 12-epoch POC budget. The C context branch received a first-batch L1 gradient of **1.45948**, confirming it was trained.

## 4. Parameter and computation implications

| Property | B | C | Change |
|---|---:|---:|---:|
| Parameters | 139,142 | 53,894 | -85,248 (-61.3%) |
| Saved checkpoint | 1,941,515 bytes | 1,541,251 bytes | -400,264 bytes (-20.6%) |
| Encoder executions | 3 | 3 | unchanged |
| Decoder executions | 3 | 3 | unchanged |
| Attention operations per forward | 18 | 18 | unchanged |
| FFN calls per forward | 9 | 9 | unchanged |

**Key distinction:** weight sharing reduces the number of unique parameters and optimizer state, but the same encoder/decoder computations still run. Therefore, inference FLOPs remain approximately unchanged. Actual FLOP reduction would require fewer executed layers, reduced sequence lengths, reduced `d_model`/`d_ff`, or fewer attention operations—not weight tying alone.

## 5. Measured matched results

| Metric | B: current context MTN | C: layer-tied context MTN | C vs B |
|---|---:|---:|---:|
| MSE | 1113.21 | 1339.31 | +226.10 (+20.3%) |
| C-MSE | 1094.57 | 1326.93 | +232.36 (+21.2%) |
| CF-MSE | 3646.51 | 4530.43 | +883.92 (+24.2%) |
| Best validation MSE | 11.35 | 9.78 | Different validation result; held-out test is decisive here |
| C lower-MSE test samples | — | 5/10 | — |

### Warmed repeated CPU latency benchmark

Both saved models were run on the identical 10 test inputs after five warm-up passes, with 200 timed forwards per model.

| Latency metric | B | C | C vs B |
|---|---:|---:|---:|
| Mean per-forward latency | 47.20 ms | 50.13 ms | +2.93 ms (+6.2%) |
| Standard deviation | 4.98 ms | 8.04 ms | — |

Version C did not show a latency reduction. This is consistent with the unchanged layer-execution and attention-operation counts. The small observed slowdown should not be interpreted as a general property of sharing; it is a measurement from this CPU POC.

## 6. Recommendation

Layer tying is technically safe for the interface and gives a **61.3% parameter reduction**, but in this small matched POC it increases held-out MSE by **20.3%**, C-MSE by **21.2%**, CF-MSE by **24.2%**, and does not reduce measured CPU latency.

**Do not replace Version B with this fully layer-tied Version C for the next accuracy-focused stage.** Keep Version B as the accuracy reference. If parameter footprint is the priority, investigate less aggressive encoder-only or decoder-only tying next; they each reduce parameters by about 30.6% while preserving the context branch. To reduce computational complexity, separately test fewer executed layers or smaller attention/FFN dimensions, and retrain/evaluate them under the same matched protocol.

**Bottom line:** weight sharing reduces parameter count by **61.3%**, while inference computation remains approximately unchanged because the same 18 attention operations and six layer executions are retained. In this POC, measured inference latency changed by **+6.2%** and prediction accuracy regressed.
