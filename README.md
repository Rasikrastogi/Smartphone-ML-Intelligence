# Smartphone ML Intelligence

An applied machine learning project for analysing smartphone specifications, predicting price and benchmark performance, studying feature relationships and generalization, identifying market segments, estimating specification-based value, and building an explainable recommendation system.

## Project Overview

This project uses a custom-curated smartphone dataset containing hardware, benchmark, display, camera, battery, build, launch, and pricing attributes.

The workflow is structured as a sequence of reproducible notebooks covering:

1. Data audit and cleaning
2. Exploratory data analysis
3. Smartphone price prediction
4. Smartphone performance prediction
5. Market segmentation
6. Specification-based value analysis
7. Explainable recommendation

The project emphasizes leakage-safe preprocessing, repeated cross-validation, controlled model comparison, feature ablation, grouped validation, and interpretable machine learning experiments.

## Repository Structure

```text
smartphone-ml-intelligence/
├── data/
│   ├── raw/
│   │   └── Phones.csv
│   └── processed/
│       └── phones_clean.csv
│
├── notebooks/
│   ├── archive/
│   │   └── 00_original_phones_analysis.ipynb
│   ├── 01_data_audit_and_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_price_prediction.ipynb
│   └── 04_performance_prediction.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Completed Work

### 1. Data Audit and Cleaning

Built a reproducible data-quality workflow covering:

- schema inspection
- duplicate detection
- categorical consistency checks
- numerical range validation
- logical feature checks
- date and Boolean normalization
- derived feature creation
- cleaned dataset generation

The raw dataset is preserved separately from the processed analytical dataset.

### 2. Exploratory Data Analysis

Analysed:

- brand distribution
- smartphone price distribution
- market price segments
- brand positioning
- benchmark distributions
- price-performance relationships
- battery and charging characteristics
- display and camera attributes
- feature correlations
- pricing outliers

### 3. Smartphone Price Prediction

Built leakage-safe regression pipelines using repeated cross-validation.

Models evaluated include:

- Dummy Regressor
- Linear Regression
- Ridge Regression
- Random Forest
- Extra Trees
- Gradient Boosting

Key experiments include:

- comparison of linear and nonlinear regression models
- regularized regression
- raw-price versus log-price modelling
- specification-only price prediction
- benchmark-enhanced price prediction
- controlled comparison of benchmark contribution

The final workflow retains the specification-based model when benchmark features do not provide meaningful additional predictive information.

### 4. Smartphone Performance Prediction

Modelled Antutu benchmark performance from smartphone hardware characteristics.

Models evaluated include:

- Ridge Regression
- Random Forest
- Extra Trees
- Gradient Boosting

Key experiments include:

#### Experiment A: Hardware-Only Performance Prediction

Predicted Antutu performance from processor and hardware-related characteristics.

#### Experiment B: Hardware + Geekbench Prediction

Added Geekbench single-core and multi-core scores to measure how much related benchmark information improves Antutu prediction.

This experiment is treated as benchmark-assisted prediction rather than purely hardware-based prediction.

#### Experiment C: Predicting Without Thickness & Screen Size

Removed physical dimensions to test whether they meaningfully contribute to performance prediction.

The experiment showed that most predictive information is retained without these features.

#### Experiment D: Predicting with Just Processors

Used only processor and processor type to measure how much benchmark performance can be explained from processor identity alone.

The comparison showed that processor information is highly predictive, while fabrication process and memory/storage technologies provide additional useful information.

#### Processor-Grouped Validation

Used grouped cross-validation with processor identity as the grouping variable to test generalization to smartphones using processors not seen during training.

This provides a stricter evaluation than standard cross-validation and helps distinguish processor memorization from broader hardware-performance learning.

## Planned Modules

### 5. Smartphone Market Segmentation

Identify meaningful groups of smartphones based on pricing, performance, display, battery, camera, and hardware characteristics.

### 6. Specification-Based Value Analysis

Compare actual market price with model-estimated specification-based price to identify devices priced above or below their expected specification-based value.

### 7. Explainable Recommendation System

Build a content-based recommendation layer using structured smartphone specifications, user constraints, and similarity-based ranking.

## Dataset
The raw dataset currently contains around 270 smartphone records and hardware,
benchmark, display, camera, battery, build, launch, and price attributes.

The dataset is custom-curated and continuously expanded with additional smartphone records.

The schema covers attributes including:

- processor and fabrication
- RAM and storage technologies
- benchmark scores
- display specifications
- camera configuration
- battery and charging
- build characteristics
- launch information
- pricing

The dataset is maintained separately from the modelling logic so that additional smartphone records can be added and all downstream analyses can be rerun reproducibly.

Before public redistribution, the original data sources and their usage or licensing terms should be verified and documented.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook, Git, GitHub

## Project Status

Completed:

- Data audit and cleaning
- Exploratory data analysis
- Price prediction
- Performance prediction

In progress / planned:

- Market segmentation
- Specification-based value analysis
- Explainable recommendation system

