# Phase 6C literature review

## Scope and verification

This review uses primary publisher, proceedings, arXiv and official-project
pages checked on 2026-11-27. The reference database records URLs rather than
unverified numerical claims. It is a targeted review of methods relevant to
egocentric pedestrian box forecasting, not an assertion that every 2026 paper
has been exhaustively indexed.

## What is directly relevant

MTN is the closest architectural antecedent: it predicts future pedestrian
boxes from egocentric boxes, ego speed and optical flow, with MSE/C-MSE/CF-MSE
on PIE/JAAD. IDD-PeD supplies MTN, BiTraP and SGNet implementations/checkpoints
as baseline families. PIETraj, BiTraP and SGNet are relevant first-person
trajectory baselines, but their input, stochasticity and preprocessing must be
verified before numbers are compared to AO.

Classical constant velocity and residual regression are essential baselines:
the Phase 6B result shows a fixed train-fitted ridge residual is better than
all frozen neural candidates. This is an empirical finding, not evidence that
CV-residual learning is a new general method; residual learning, kinematic
anchors and linear motion models are longstanding forecasting practice.

## Context only, not numerical competitors

Social LSTM/Social GAN/Trajectron++/AgentFormer model multi-agent centre paths,
usually in fixed-camera or map-rich settings. PECNet, Y-Net and EigenTrajectory
use endpoint, map or multi-hypothesis formulations and commonly best-of-K
ADE/FDE. They help explain the field's social, goal and stochastic approaches;
they cannot establish superiority or inferiority relative to 45-step,
single-forecast, raw-pixel box MSE on IDD-PeD.

## Implications

The defensible literature gap is narrow: a reproducible annotation-only,
single-forecast IDD-PeD evaluation with an explicit classical-versus-neural
comparison. It is not a claim to a universal SOTA trajectory predictor, a
replacement for full-input MTN, or a direct challenge to best-of-K papers.

See `VERIFIED_REFERENCE_DATABASE.csv` for titles, authors, venue, URLs,
modalities, representation, evaluation family and implementation links.
