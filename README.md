# iML_SFAuDA

Stability of Feature Attributions Under Data Augmentation.

This project studies how training-time data augmentation affects the stability of post-hoc explanations such as SHAP and LIME.

We run controlled experiments on small benchmark datasets (tabular and image) and compare explanation stability across multiple training runs.

## Setup
1. Create a Python 3.10 conda environment
```bash
conda create -n iml-sfauda python=3.10
```
2. Activate the conda environment
```bash
conda activate iml-sfauda
```
3. Install required dependencies
```bash
pip install -r requirements.txt
```

## Project structure
```bash
iml-explanation-stability/
├── results/
│   # Experiment outputs
├────── stability_scores.csv
│       # Output of stability scores
│
├── LICENSE
│
├── README.md
│   # Project overview and setup instructions
│
├── augmentations.py
│   # Data augmentation and perturbation methods.
│
├── explanations.py
│   # SHAP and LIME explanation logic.
│
├── iML_Project_Proposal.pdf
│   # Initial project proposal.
│
├── main.py
│   # Runs experiments.
│
├── metrics.py
│   # Explanation stability metrics.
│
├── models.py
│   # Model definitions and training.
│
├── requirements.txt
│   # Python dependencies
│
├── utils.py
│   # Shared helper functions.
```

## Status
Project setup and baseline experiments in progress.