# Baseline compatibility record

Status date: 2026-11-21. This is a data/flow-contract audit, not a result table.
Published values and re-run values will remain in separate tables. The frozen
from-scratch GP6 protocol uses 15 observed and 45 future bounding boxes,
per-frame OBD speed, CPU Farneback pooled flow, five static scene codes,
observed occlusion codes, and source-frame paths. It has no public baseline
repository checked out locally and no adapter for the frozen manifests.

## B2: published IDD-PeD baselines

| Method | Public implementation / stated IDD-PeD entry point | Required data / output contract | Can directly consume the local frozen cache? | Decision |
|---|---|---|---|---|
| PIETraj (PIEPredict) | The [official IDD-PeD repository](https://github.com/Ruthvik9/IDD-PeD) documents `PIEPredict/train_test.py` and an IDD-PeD environment. The code is not present under `D:\IDD-PeD-Evaluation`. | Its PIETraj pipeline trains separate intention, speed and trajectory components and needs IDD-PeD image data plus its own dataset assembly; the documented training path notes substantial image-model memory use. It does not consume MTN pooled-flow tensors. | No. The cache provides frame paths, boxes and speed but not the PIETraj-prepared image/intention/speed training records or adapter. Building one would change the frozen matched protocol. | Do not run now. Retain only a separately labelled published block if cited. |
| BiTraP | [Official IDD-PeD repository](https://github.com/Ruthvik9/IDD-PeD) documents `BITRAP/tools/train.py --config_file configs/bitrap_np_IDDP.yml`; upstream [BiTraP](https://github.com/umautobots/bidirection-trajectory-predicter) supports JAAD/PIE/ETH-UCY configurations. Neither checkout nor IDD-PeD adapter is local. | Goal-conditioned stochastic CVAE; the fair comparison would need its IDD-PeD preprocessing and both its best-of-20 and one stated single-hypothesis output. It does not use MTN Farneback flow tensors. | No direct reader/adapter exists for the manifest or `.pkl` flow cache. | Do not run now. Do not mix its published best-of-20 result with re-run single-trajectory metrics. |
| SGNet | [Official IDD-PeD repository](https://github.com/Ruthvik9/IDD-PeD) documents `SGNet/tools/iddp/train_cvae.py` and `eval_cvae.py`; upstream [SGNet](https://github.com/ChuhuaW/SGNet.pytorch) is a goal-driven stochastic predictor. No local checkout or adapter is present. | Requires SGNet's IDD-PeD data layout, CVAE training/evaluation pipeline and stochastic sampling. It is not an MTN-flow-cache consumer. | No direct reader/adapter exists. | Do not run now. Keep any published best-of-20 number in a clearly separate block. |

## B3: additional candidates

| Candidate | Public-code status checked | Required inputs versus frozen cache | Runnable without changing the matched protocol? | Decision |
|---|---|---|---|---|
| ENCORE — *A Novel Benchmarking Paradigm and a Scale- and Motion-Aware Model for Egocentric Pedestrian Trajectory Prediction* | No official public implementation was identified in the checked paper/index sources. | Egocentric image/scale/motion benchmark pipeline; no verified IDD-PeD adapter or frozen-cache reader. | No. | Not run; do not put a number in an IDD-PeD table. |
| TrajMamba — *An Ego-Motion-Guided Mamba Model for Pedestrian Trajectory Prediction from an Egocentric Perspective* | No verified official repository for the cited arXiv identifier was located. A similarly named 2026 RGB-D/ROS project is a different task and is not usable as evidence. | Claimed PIE/JAAD setting, not an IDD-PeD cache contract. | No. | Not run. |
| PedFormer | No verified official training repository/IDD-PeD adapter was located. The paper evaluates PIE and JAAD. | RGB, bounding boxes, ego motion, semantic attentive interactions and multitask crossing labels; the frozen cache lacks the PedFormer-prepared visual/interaction/segmentation inputs. | No. | Not run. |
| Zhang, Ding & Tian, *Decouple Ego-View Motions for Predicting Pedestrian Trajectory and Intention* (TIP 2024) | No verified public implementation/IDD-PeD adapter was located. | Two towers use ego actions/speed, initial position, prior pedestrian trajectory and visual features, plus intention task data; no direct frozen-cache reader. | No. | Not run. |
| Munir & Kucner, *Context-aware Multi-task Learning for Pedestrian Intent and Trajectory Prediction* (PTINet) | Code is linked by the paper at [munirfarzeen/PTINet](https://github.com/munirfarzeen/PTINet), but it is not checked out locally and reports JAAD/PIE/TITAN rather than IDD-PeD. | Images and dense flow plus keypoints, head orientation, demographic/behavioural attributes, scene data and intent targets. The frozen cache lacks several required local attributes and the code's dataset adapter. | No. | Not run. |
| *A Multi-Stage Goal-Driven Network for Pedestrian Trajectory Prediction* | No verified official code/IDD-PeD adapter was located during the initial audit. | Goal-stage supervision and its published dataset pipeline have not been shown compatible with the manifest. | No. | Not run. |

Consequently B2/B3 will not be launched unless an official codebase and a
lossless frozen-manifest adapter are available. No published and re-run numbers
will share a table block.
