# SOTA protocol compatibility report

## Categories

**A — directly comparable.** AO-0, AO-CP, AO-1, AO-2, AO-3 and AO-RIDGE use
the same 2,615 official-test windows, 15/45 box protocol, raw-pixel metrics,
train-only normalization and deterministic single forecast. These are the only
valid numerical competitors in this study.

**B — published IDD-PeD results, unmatched.** Published/released IDD-PeD MTN,
BiTraP and SGNet families are benchmark-relevant but may use ego speed, optical
flow, different split/setup code, goals or stochastic candidates. They can be
tabulated as published context only after quoting their original setup.

**C — related methods, not numerically comparable.** Social LSTM, Social GAN,
Trajectron++, AgentFormer, PECNet, Y-Net and EigenTrajectory largely forecast
world/ground-plane centres in fixed-camera/multi-agent/map settings with
ADE/FDE, often best-of-K. Their numbers must not appear in the same ranked table.

## Mandatory comparison disclosures

Phase 6B is annotation-only: first 15 absolute-pixel `xyxy` boxes; no raw
video, optical flow, ego motion, scene map, other-agent state, or future input.
It predicts 45 future boxes at 30 FPS and reports pooled coordinate MSE,
C-MSE and CF-MSE. It includes all eligible stride-30 windows over GP3/5/8/9.

Published methods can differ in training/test partitions, extraction/frame
rates, visibility filtering, centre versus box output, pixel versus world units,
observation/future duration, deterministic versus best-of-K scoring and metric
code. Any one difference blocks a direct rank claim. Do not compare parameter
counts across different encoders/perception stacks as computational equivalence.

## Feasibility

No external method should be trained in this phase. CPU-only reproduction is
not fair without a frozen conversion/eligibility protocol, released compatible
inputs, fixed stochastic K/seeds, and a resource budget. The local data does
not provide full-input official-test optical flow; producing it would violate
the current no-raw-video-download boundary.
