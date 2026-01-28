# DSTA: Dynamic Spatial Topology and Confidence-Aware Soft Alignment

This is the official implementation of the paper: **"DSTA: Dynamic Spatial Topology and Confidence-Aware Soft Alignment for Cross-dataset EEG Emotion Recognition"**.

## 🌟 Core Features
DSTA addresses the fundamental challenges of structural incompatibility and distributional misalignment in cross-dataset EEG emotion recognition through two primary modules:

1. **DSTT (dstt.py)**: Dynamic Spatial Topology Transformer.
   - **Dynamic Channel-wise Tokenization**: Reformulates EEG channels as independent semantic tokens to capture localized neural dynamics.
   - **Topological Reconstruction**: Adaptively models global functional connectivities, circumventing the constraints of rigid, hand-crafted spatial priors.
   
2. **CASA (alignment.py)**: Confidence-Aware Soft Alignment.
   - **Uncertainty-guided Weighting**: Preserves continuous emotional manifolds during domain adaptation by gating alignments with prediction confidence.
   - **Decoupled Alignment Loss**: A multi-objective optimization framework integrating Self-Augmentation, Intra-domain Supervised, and Inter-domain Confidence losses.

## 📁 Repository Structure
- `dstt.py`: Implementation of the DSTT backbone and Dynamic Hybrid Masking.
- `alignment.py`: Implementation of the Soft Alignment heads and CASA mechanisms.
- `main.py`: The unified training logic and multi-objective optimization workflow.
- `datapipe.py`: Data processing and loading utilities for EEG datasets.

## 📊 Performance
DSTA establishes new state-of-the-art benchmarks in challenging cross-dataset scenarios:
- **SEED → SEED-IV**: 63.25% Accuracy
- **SEED-IV → SEED**: 62.92% Accuracy

*Our learned spatial patterns reveal stable hemispheric lateralization consistent with Frontal EEG Asymmetry theory, bridging the gap between engineering efficacy and cognitive interpretability.*

## 🚀 Quick Start
1. **Requirements**:
   ```bash
   pip install torch torch_geometric numpy pandas scikit-learn
