# Bike-Sharing Hourly Demand Forecasting

Predicting the hourly number of bike rentals for a city bike-sharing system, to
support fleet planning and rebalancing decisions.

End-to-end regression project on a **time-series** problem: framing, temporal
validation, feature engineering, model selection, hyperparameter tuning, honest
test-set evaluation, and a business-facing report.

---

## Results at a glance

| Metric | Value | Meaning |
|--------|-------|---------|
| **Test RMSE** | **70 bikes/hour** (95% CI: 66.7 - 73.9) | Average error on unseen data |
| **vs. baseline** | **-70%** | vs. predicting the historical average (~233 bikes error) |
| **R²** | **0.90** | Share of demand variability explained |
| Model | Gradient Boosting Regressor | Tuned via time-series cross-validation |

The model generalizes well: the cross-validated RMSE (67.3) and the test RMSE
(70.0) are close, so there is no overfitting to the validation folds. One detail
worth flagging: average demand in the (more recent) test period is ~40% higher
than in training. The model captures this growth trend, which is why a naive
average-based baseline does so poorly.

---

## Problem statement

A city bike-sharing operator needs to anticipate **how many bikes will be rented
each hour** so it can position the fleet, schedule rebalancing trucks and staff
appropriately. Under-supply means lost revenue and unhappy users; over-supply
wastes operating cost. The goal is an accurate, honest hourly demand estimate the
operations team can act on.

## Data

[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- hourly rentals for Washington D.C. (2011-2012), 17,379 rows. The target `cnt`
ranges from 1 to 977 bikes/hour (mean ~ 190). See [`data/README.md`](data/README.md).

## Approach

The project follows a structured ML workflow:

1. **Frame the problem** - hourly regression; success measured by RMSE in bikes.
2. **Explore** - demand is driven by hour of day, weather and a long-term growth trend.
3. **Temporal split** - the most recent 20% of data is held out as the test set.
4. **Feature engineering** - cyclical time encodings + domain features.
5. **Preprocessing pipeline** - one-hot, scaling and feature creation in a single `Pipeline`.
6. **Model shortlist** - Linear Regression, Decision Tree, Random Forest, Gradient Boosting.
7. **Fine-tuning** - `GridSearchCV` over a `TimeSeriesSplit`.
8. **Evaluation** - test-set RMSE, MAPE, bootstrap CI, error analysis by segment.

## Key technical decisions

A few decisions did most of the work here:

- **Temporal validation, not random.** This is a time series, so both the
  train/test split and cross-validation use chronological ordering
  (`TimeSeriesSplit`). A random split would leak future information and inflate
  the score.
- **Cyclical encoding of time.** Hour, month and weekday are circular; encoding
  them as `sin`/`cos` pairs captures that hour 23 is adjacent to hour 0. The hour
  features turned out to be the single most important predictor (~44% of model
  importance).
- **Domain feature engineering.** A working-day rush-hour flag and a
  temperature/humidity "pleasantness" interaction were added - both rank in the
  top 5 most important features.
- **Target transformation tested and rejected.** A `log(cnt)` transform was
  evaluated but *worsened* the absolute RMSE: it optimizes relative error, while
  the business metric is absolute. Tree ensembles also model the skewed target
  well without it.
- **Knowing when to stop tuning.** Tuning improved the RMSE from 75.2 -> 67.3;
  further grid expansion yielded only ~1 bike of improvement, so tuning was
  stopped (diminishing returns).

## Results in detail

### Model comparison (cross-validated RMSE)

| Model | CV RMSE (default) | CV RMSE (tuned) |
|-------|------------------:|----------------:|
| Linear Regression | 104.9 | - |
| Decision Tree | 95.0 | - |
| Random Forest | 75.2 | 72.7 |
| **Gradient Boosting** | 77.6 | **67.3** |

### Where the model is reliable

| Segment | Reliability |
|---------|-------------|
| High-demand hours (peak) | Best - error ~ 24% of demand |
| Working days vs. weekends | More accurate on working days (66 vs. 78 bikes RMSE) |
| Low-demand hours (night) | Small absolute error but high % error |

> Note on MAPE: the test MAPE is ~56%, but this is **inflated by low-demand night
> hours** (being off by a few bikes is a huge percentage when demand is ~5). The
> model is much more accurate during the high-demand hours that matter for
> operations.

### What drives predictions

1. **Hour of day** (dominant, ~44%)
2. **Weather** - temperature/humidity interaction
3. **Long-term system growth**
4. **Rush hour on working days**

## Project structure

```
bike-sharing-demand-forecasting/
├── README.md                 # This case study
├── requirements.txt
├── reproduce.py              # Train + evaluate the final model end to end
├── predict.py                # Predicted vs. actual demand for a sample day
├── data/
│   ├── README.md             # Dataset source and schema
│   └── hour.csv
├── notebooks/
│   └── 01_analysis.ipynb     # Narrated end-to-end analysis
├── src/
│   ├── data.py               # Loading + temporal split
│   ├── features.py           # Feature engineering + preprocessing pipeline
│   ├── train.py              # Final model definition + training
│   └── evaluate.py           # Metrics, bootstrap CI, segment analysis
└── models/                   # Saved model artifact (regenerated by train.py)
```

## Reproduce it

All commands run from the **repository root** (so the `src` package resolves):

```bash
# 0. (first time only) install dependencies
pip install -r requirements.txt

# 1. Train the final model and print the test-set metrics
python reproduce.py

# 2. (optional) Inspect predicted vs. actual demand for a sample day
python predict.py
```

`reproduce.py` trains the tuned Gradient Boosting model, saves it to
`models/model.joblib`, and prints:

```
Test-set performance
--------------------------------
RMSE      : 70.0 bikes/hour
RMSE 95%CI: [66.7, 73.9]
R^2       : 0.899
MAPE      : 56%
```

`predict.py` loads that saved model and prints predicted vs. actual demand, hour
by hour, for one day of the held-out test set.

The full narrated analysis lives in
[`notebooks/01_analysis.ipynb`](notebooks/01_analysis.ipynb).

## Limitations & next steps

- **No special events.** Festivals, games or road closures that spike demand on a
  given day are not modeled - human judgment should override the model then.
- **No lag features.** The model does not use the previous hour's/day's demand,
  usually a strong predictor in time series. **This is the most promising next
  improvement.**
- **Weaker on weekends and night hours.**
- **Partial seasonal evaluation.** The temporal test window did not cover every
  season (summer is absent), so seasonal behavior is only partially validated.

## Tech stack

Python | pandas | scikit-learn | scipy | matplotlib | Jupyter
