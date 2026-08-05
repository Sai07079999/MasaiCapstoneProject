# Analytics

A complete data science case study on the catalog built by
`data_pipeline/`: EDA, feature engineering, multi-model comparison
with hyperparameter tuning, evaluation, and business interpretation.

## Business question

Can we predict whether a book will be a **high performer** (rating
≥ 4) using only attributes known before reviews accumulate — price,
category, and stock behavior? This is a realistic pre-launch
merchandising signal.

## Structure

```
analytics/
├── notebooks/analysis.ipynb   # Full narrative case study (executed, outputs baked in)
├── src/
│   ├── data_loader.py         # Reads directly from data_pipeline's SQLite DB
│   ├── eda.py                 # Distribution/correlation/category visualizations
│   ├── feature_engineering.py # Missing values, outliers, encoding, target definition
│   ├── modeling.py            # 3-model comparison + GridSearchCV + cross-validation
│   └── evaluation.py          # Metrics, confusion matrix, ROC, feature importance, serialization
├── outputs/                   # Generated charts, model_comparison.csv, best_model.pkl
├── tests/                     # Unit tests for feature engineering (no I/O)
└── run_analysis.py            # Runs the entire workflow as a script
```

## Running it

```bash
# 1. Populate the database (real scrape or synthetic seed)
python -m data_pipeline.main                      # live scrape
# or, if you don't have unrestricted network access:
python -m data_pipeline.seed_synthetic_data        # 600 synthetic products

# 2. Run the full analysis
python -m analytics.run_analysis

# or open the narrative version
jupyter notebook analytics/notebooks/analysis.ipynb
```

## Models compared

| Model | Why it's included |
|---|---|
| Logistic Regression | Interpretable baseline; coefficients are directly explainable |
| Random Forest | Captures non-linear feature interactions |
| Gradient Boosting | Typically the strongest tabular performer |

Each is tuned via 5-fold stratified `GridSearchCV` optimizing ROC-AUC,
then compared on held-out test accuracy, precision, recall, F1, and
ROC-AUC (`outputs/model_comparison.csv`).

## Key result (on the synthetic seed dataset)

Logistic Regression won on this dataset (test ROC-AUC ≈ 0.62) —
genre (one-hot category) and `price_vs_category_avg` (price relative
to its own genre's average, not raw price) were the strongest
predictors. Full interpretation and business recommendations are in
Section 8 of the notebook.

> **Note on data:** these specific numbers reflect the synthetic seed
> dataset (see root README) — re-running against a real scrape from
> `data_pipeline/main.py` will produce different, real-world numbers
> using the exact same code path.

## Tests

```bash
pytest analytics/tests/ -v
```
