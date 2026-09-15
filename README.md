\# Stage-Wise Curvature Tuning Research



MS Research Project — Fall 2026



This repository contains my master's research based on the NeurIPS 2025 paper:



\*\*Curvature Tuning: Provable Training-free Model Steering From a Single Parameter\*\*



The project first reproduces the authors' original Curvature Tuning results and then investigates a parameter-efficient extension using stage-wise curvature parameters.



\---



\## Base Paper



\*\*Paper:\*\* Curvature Tuning: Provable Training-free Model Steering From a Single Parameter  

\*\*Conference:\*\* NeurIPS 2025



Original paper:

https://proceedings.neurips.cc/paper\_files/paper/2025/hash/57cc9268e06500c7681e58829dca4a07-Abstract-Conference.html



Original GitHub repository:

https://github.com/Leon-Leyang/curvature-tuning



\---



\## Research Motivation



The original paper introduces Curvature Tuning (CT), which adapts pretrained neural networks by modifying activation-function curvature rather than updating the original pretrained weights.



The paper proposes two main variants:



\- \*\*S-CT (Steering Curvature Tuning):\*\* uses one shared curvature parameter, `beta`, across the network.

\- \*\*T-CT (Trainable Curvature Tuning):\*\* uses many trainable curvature parameters and provides greater flexibility.



This research investigates a middle ground:



\### Stage-Wise Curvature Tuning



Instead of using one global `beta` for the entire network, each major network stage will use its own curvature parameter.



For a ResNet architecture:



```text

S-CT:

beta -> all stages



Proposed Stage-Wise CT:

Stage 1 -> beta\_1

Stage 2 -> beta\_2

Stage 3 -> beta\_3

Stage 4 -> beta\_4

