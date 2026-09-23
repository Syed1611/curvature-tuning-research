# Curvature Tuning - Original Paper Reproduction

MS Research Project - Fall 2026

This branch contains a reproduction of experiments from the NeurIPS 2025 paper:

**Curvature Tuning: Provable Training-free Model Steering From a Single Parameter**

The purpose of this branch is to establish a verified baseline before developing any research extensions.

---

## Base Paper

**Paper:** Curvature Tuning: Provable Training-free Model Steering From a Single Parameter  
**Authors:** Leyang Hu, Matteo Gamba, Randall Balestriero  
**Conference:** NeurIPS 2025

Paper:

https://proceedings.neurips.cc/paper_files/paper/2025/hash/57cc9268e06500c7681e58829dca4a07-Abstract-Conference.html

Original implementation:

https://github.com/Leon-Leyang/curvature-tuning

---

## Objective

This branch focuses on reproducing the original Curvature Tuning transfer-learning experiment using:

- ImageNet-pretrained ResNet-18
- Beans as the downstream dataset
- Baseline ReLU model
- Single-Parameter Curvature Tuning (S-CT)
- Seeds 42, 43, and 44

No Stage-Wise Curvature Tuning modifications are introduced in this branch.

---

## Curvature Tuning

Curvature Tuning modifies the curvature of the activation functions of a pretrained network without updating the original pretrained backbone weights.

For Single-Parameter Curvature Tuning (S-CT), one shared curvature parameter β is used throughout the network.

```text
ResNet-18

Stem   ─┐
Layer1 ─┤
Layer2 ─┤
Layer3 ─┤── shared β
Layer4 ─┘
