# iML_SFAuDA

Stability of Feature Attributions Under Data Augmentation.

This project studies how training-time data augmentation affects the stability of post-hoc explanations such as SHAP and LIME.

We run controlled experiments on small benchmark datasets (tabular and image) and compare explanation stability across multiple training runs.

## Setup
1. Create a Python 3.10 virtual environment
```bash
python3.10 -m venv venv
```
2. Activate the virtual environment
```bash
source venv/bin/activate
```
3. Install required dependencies
```bash
pip install -r requirements.txt
```

## Project structure
```bash
iml-explanation-stability/
├── README.md
│   # Project overview and setup instructions
│
├── requirements.txt
│   # Python dependencies
│
├── data/
│   # Datasets used in experiments
│   ├── raw/
│   └── processed/
│
├── src/
│   # Core project code
│   ├── datasets/
│   ├── models/
│   ├── augmentations/
│   ├── explainers/
│   ├── metrics/
│   └── utils/
│
├── experiments/
│   # Reproducible experiment scripts
│
├── results/
│   # Experiment outputs
│   ├── figures/
│   └── tables/
```

## Status
Project setup and baseline experiments in progress.