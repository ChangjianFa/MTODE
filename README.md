# Multi-task Learning of Complex Networks via Nonlinear Ordinary Differential Equations

This repository contains the simulation code, analysis scripts, and comparison experiments accompanying the manuscript:

**“Multi-task Learning of Complex Networks via Nonlinear Ordinary Differential Equations.”**

---

## 📁 Repository Structure

### **Simulation & Analysis**

- **`computer_simulation.R`**  
  Simulation experiments for a 20-variable nonlinear dynamical system.

- **`Plasmodium_falciparum_gene_expression_analysis.R`**  
  Analysis of *Plasmodium falciparum* strain D6 gene expression data.

### **Method Comparisons**

- **`comparison_variable_selection_methods.R`**  
  Comparison of variable selection methods (Lasso, Sparse Group Lasso, etc.).

- **`comparison_MTL_vs_STL.R`**  
  Evaluation of multi-task learning (MTL) vs. single-task learning (STL).

- **`comparison_integral_vs_nonintegral_methods.R`**  
  Comparison between integral-based and non-integral modeling approaches.

- **`Compare_with_pySINDY_dynGenie3_nonlinearODE.py`**  
  Python script comparing our method to pySINDy, dynGENIE3, and nonlinear ODE models.

---

## ⚙️ Requirements

### R Packages
- `glmnet`
- `splines`
- `ggplot2`
- `tidyverse`
- `doParallel`

### Python Packages (for comparison script)
- `pysindy`
- `numpy`
- `scipy`
- `sklearn`
- `matplotlib`

---

## 🚀 Usage

Run any script directly in R or Python:

```bash
Rscript computer_simulation.R
python Compare_with_pySINDY_dynGenie3_nonlinearODE.py
