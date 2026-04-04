# SoftMET: Adaptive Soft 3Trees for Mixed-Effects Modeling

This project implements **SoftMET** (Soft Mixed-Effects Trees), a principled framework that integrates **Softmax Attention Mechanisms** into hierarchical mixed-effects models. 

## 🚀 Key Features
- **Soft Routing:** Replaces traditional hard, axis-aligned splits with probabilistic leaf assignments for smooth and differentiable regression surfaces.
- **Multi-Level Architecture:** Decomposes the non-parametric component into three dedicated trees:
  - **Individual-level** (Level 1)
  - **Cluster-level** (Level-2)
  - **Cross-level interactions**
- **Adaptive Attention:** Learned inter-tree weights to calibrate different sources of variation.

## 📊 Results
In our Monte Carlo simulations, the **Soft 3Trees** model significantly outperformed traditional approaches:
- **RMSE Improvement:** ~13.07% reduction in error compared to standard Linear Mixed Models (LMM).
- **AIC Optimization:** Improved model parsimony (AIC: 1606.8 vs. 1712.3).
- **Continuity:** Successfully eliminated boundary discontinuities found in original hard-partitioning tree models.

## 🛠 Tech Stack
- **Language:** R
- **Key Libraries:** `lme4`, `nnet`, `Matrix`

## 📁 Project Structure
- `SoftMET_Simulation.R`: Main script containing the estimators and simulation logic.
- `SoftMET_Simulation.Rproj`: RStudio project configuration.

## 👤 Author
- **Nastaran Nikpour**
