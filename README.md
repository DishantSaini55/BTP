# Context-Aware MTN for IDD-PeD

This repository contains the code, architecture documentation, editable Draw.io diagrams, and Overleaf paper draft for a CPU-based proof-of-concept extending MTN for pedestrian trajectory prediction on IDD-PeD.

## Current proposed POC architecture

The candidate augments the MTN trajectory memory with a gated context token derived from six native IDD-PeD context sources: intersection type, signalized type, road type, location type, motion direction, and observed occlusion. C4 selectively shares only encoder self-attention Q/K/V/O projections across the three encoder layers.

The completed matched CPU proof-of-concept found a 6.07% parameter reduction and 5.01% lower mean MSE versus the local Context-Aware MTN reference, with 3.93% higher CPU latency. It is not an official RAFT-based IDD-PeD benchmark reproduction.

## Repository layout

- `MTN/` — MTN variants, CPU POC scripts, preprocessing, and model code.
- `MTN/overleaf_draft/` — editable paper draft and TikZ figures.
- `MTN/overleaf_draft/figures/C4_Paper_Figures_With_Arrows.drawio` — multi-page editable Draw.io paper-figure set.
- Root `*.drawio` files — editable architecture diagrams.

## Reproducibility boundary

Large datasets, checkpoints, optical-flow caches, virtual environments, logs, archives, and generated experiment outputs are intentionally excluded from Git. The source code and paper assets are retained.
