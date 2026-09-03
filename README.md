# Smartphone ML Intelligence

An applied machine learning project for analysing smartphone specifications,
predicting price and benchmark performance, identifying market segments,
estimating specification-based value, and building an explainable
recommendation system.

## Current repository state

This first commit preserves the original exploratory notebook and raw dataset
before the project is refactored into reproducible machine learning pipelines.

## Repository structure

```text
smartphone-ml-intelligence/
├── data/
│   └── raw/
│       └── Phones.csv
├── notebooks/
│   └── archive/
│       └── 00_original_phones_analysis.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Important note

The archived notebook is intentionally unchanged and contains the original
local Windows dataset path. A new reproducible working notebook will be created
separately using:

```python
pd.read_csv("../data/raw/Phones.csv")
```

## Planned machine learning modules

1. Data validation and exploratory analysis
2. Leakage-safe smartphone price prediction
3. Hardware and benchmark performance modelling
4. Smartphone market segmentation
5. Specification-based value analysis
6. Explainable content-based recommendation

## Dataset

The raw dataset currently contains more than 265 smartphone records and hardware,
benchmark, display, camera, battery, build, launch, and price attributes.

Before publishing the repository, confirm that the dataset's source
and licence permit redistribution. Add the source and licence information here
once verified.
