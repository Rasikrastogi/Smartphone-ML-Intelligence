# Smartphone ML Intelligence

An end-to-end smartphone market intelligence and machine learning project built around a custom-curated smartphone dataset.

The project combines technical product research, data cleaning, SQL-based filtering, exploratory analysis, supervised learning, unsupervised market segmentation, and specification-based value analysis. The next stages extend this work into a preference-aware recommendation engine and a natural-language end-user application.

## Project Objectives

The project is designed to answer practical smartphone-market questions such as:

- Which specifications are most strongly associated with smartphone pricing?
- How accurately can smartphone price and benchmark performance be estimated from technical specifications?
- How well do models generalize to previously unseen processor families?
- What natural market segments emerge from smartphone specifications?
- Which smartphones appear to offer stronger specification-based value relative to their market price?
- How can users express requirements in natural language and receive ranked smartphone recommendations?

## Data

The dataset was independently curated through multi-source smartphone research rather than taken as a ready-made machine learning dataset.

It contains smartphone attributes covering areas such as:

- pricing and market availability
- processors and fabrication process
- RAM and storage technologies
- AnTuTu and Geekbench benchmarks
- display characteristics
- battery capacity and charging speed
- camera specifications
- physical dimensions and form factor
- launch information
- sales / market-position categories

The dataset is continuously maintained as smartphone availability and market information change.

## SQL-Based Availability Filtering

The analytical notebooks use **DuckDB** as a lightweight SQL layer on top of the processed smartphone dataset.

Before analysis or modelling, the dataset is registered as a DuckDB table and filtered so that experiments operate only on smartphones that are currently available:

```python
from pathlib import Path

import duckdb
import pandas as pd

DATA_PATH = Path("../data/processed/phones_clean.csv")

data = pd.read_csv(
    DATA_PATH,
    parse_dates=["launch_date"]
)

con = duckdb.connect()
con.register("phones", data)

query = """
SELECT *
FROM phones
WHERE avaibility = 1
"""

phones = con.execute(query).df()
```

This makes the analytical workflow more representative of the current smartphone market and also provides a SQL-based data-access layer that can later support the natural-language recommendation application.

## Project Workflow

```text
Custom Smartphone Research & Dataset Creation
                    ↓
        Data Audit and Cleaning
                    ↓
     DuckDB Availability Filtering
                    ↓
      Exploratory Data Analysis
                    ↓
          Price Prediction
                    ↓
       Performance Prediction
                    ↓
        Market Segmentation
                    ↓
 Specification-Based Value Analysis
                    ↓
 Preference-Aware Recommendation Engine
                    ↓
 Natural Language → SQL → Recommendation App
```

## Repository Structure

```text
smartphone-ml-intelligence/
│
├── data/
│   ├── raw/
│   │   └── Phones.csv
│   │
│   └── processed/
│       ├── phones_clean.csv
│       └── phones_with_value_scores.csv
│
├── notebooks/
│   ├── archive/
│   │   └── 00_original_phones_analysis.ipynb
│   │
│   ├── 01_data_audit_and_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_price_prediction.ipynb
│   ├── 04_performance_prediction.ipynb
│   ├── 05_market_segmentation.ipynb
│   └── 06_value_analysis.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Completed Work

### 1. Data Audit and Cleaning

Built a reproducible cleaning workflow for the custom smartphone dataset.

Key work includes:

- schema and datatype inspection
- duplicate and logical consistency checks
- standardization of column names and values
- date parsing
- boolean and categorical cleanup
- derived smartphone attributes
- preservation of raw data separately from processed analytical data

The processed dataset is stored as `data/processed/phones_clean.csv`.

### 2. Exploratory Data Analysis

Explored the smartphone market across brands, pricing, processors, benchmark performance, battery and charging, displays, cameras, physical characteristics, correlations, and outliers.

The analysis focuses on understanding relationships in the dataset before applying predictive models.

### 3. Smartphone Price Prediction

Developed leakage-aware machine learning pipelines to estimate smartphone price from technical specifications.

The notebook compares multiple regression approaches, including baseline regression, regularized linear models, Random Forest, Extra Trees, and Gradient Boosting. Repeated cross-validation is used to compare models more reliably.

A second experiment evaluates whether benchmark scores provide additional pricing information beyond the smartphone specifications themselves.

### 4. Smartphone Performance Prediction

Modelled AnTuTu performance from smartphone hardware characteristics.

The experiments investigate:

- hardware-only performance prediction
- benchmark-assisted prediction using Geekbench scores
- processor-grouped validation for unseen processor identities
- contribution of fabrication process and memory / storage technologies
- processor-only prediction versus broader hardware information

Processor-grouped validation is used to test whether model performance depends heavily on seeing the same processor families during training.

### 5. Smartphone Market Segmentation

Performed unsupervised market segmentation using standardized numerical smartphone characteristics across price, performance, display refresh rate, battery capacity, charging speed, main-camera resolution, useful camera count, and device size.

Multiple cluster counts were evaluated using inertia, Silhouette Score, Davies-Bouldin Score, and Calinski-Harabasz Score.

The final segmentation was interpreted using cluster profiles, representative smartphones, brand distributions, and sales-category distributions.

This allows the project to identify naturally occurring smartphone market groups rather than relying only on manually defined price brackets.

### 6. Specification-Based Value Analysis

Extended the price-prediction work into a market-value analysis.

For every smartphone, repeated out-of-fold predictions are used to estimate a specification-based expected price without evaluating a phone using a model that was trained on that same phone.

The analysis derives:

- expected specification-based price
- absolute value gap
- percentage value gap
- brand-level value patterns
- value patterns across price ranges
- a relative 0–100 value score

The value gap is interpreted as the difference between the model's specification-based price estimate and the actual market price.

It is **not** treated as an objective measure of whether a smartphone is overpriced or underpriced, because factors such as software support, ecosystem, after-sales service, brand positioning, design, and resale value may not be fully represented in the dataset.

The resulting value score is saved for use by the recommendation engine.

## Planned Work

### 7. Preference-Aware Recommendation Engine

Build a recommendation layer that separates:

- **hard constraints**, such as maximum budget or required brand
- **soft preferences**, such as performance, camera quality, battery, charging speed, thinness, or value

Eligible smartphones will first be filtered and then ranked using normalized feature scores, specification-based value signals, similarity, and user-derived preference weights.

### 8. Natural-Language Smartphone Recommendation Application

The final application will allow users to express requirements naturally, for example:

> "I want the slimmest phones with good performance under Rs. 40,000."

The planned flow is:

```text
Natural-language request
        ↓
Structured intent extraction
        ↓
Validated SQL generation
        ↓
DuckDB candidate filtering
        ↓
Preference-aware ranking
        ↓
Top smartphone recommendations
        ↓
Human-readable explanation and trade-offs
```

Part 8 will also investigate more meaningful representations for selected categorical attributes.

For genuinely ordered categorical variables, ordinal modelling can be used to derive continuous market-position or technology-position signals instead of assigning arbitrary equally spaced integer values.

Nominal features will remain categorical or use an appropriate non-ordinal representation.

## Technologies

- Python
- SQL
- DuckDB
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Git
- GitHub

Planned application components may additionally include Streamlit and an LLM-based structured intent layer.

## Methodological Principles

The project follows several principles throughout the notebooks:

- preserve raw data and generate processed analytical datasets separately
- filter current smartphone availability through SQL before market analysis and modelling
- avoid target leakage
- use cross-validation rather than relying on a single train/test split
- evaluate model generalization beyond headline metrics
- treat model predictions as analytical estimates rather than objective market truth
- keep recommendation constraints separate from ranking preferences
- use validated SQL generation rather than executing unrestricted model-generated SQL

## Current Status

Completed:

- Data audit and cleaning
- Exploratory data analysis
- Price prediction
- Performance prediction
- Market segmentation
- Specification-based value analysis
- DuckDB-based availability filtering across the analytical workflow

Next:

- Preference-aware recommendation engine
- Natural-language smartphone recommendation application
