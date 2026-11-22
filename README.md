# statistical-learning-bike-demand Modelling

**BWIB621 / STTN627 — Statistical Learning Practical Project**

A comparative study of statistical, machine learning, and deep learning models for
predicting **daily bike-rental demand** using the UCI Bike Sharing Dataset.

---

## Authors

| Name & Surname | Student Number |
|---|---|
| Shawn Cloete | 41142276 |
| Luyanda Senotlela | 56471955 |
| Mlungisi Mthetwa | 45984530 |

---

## Project Overview

This project investigates whether flexible nonlinear models (GAMs), tree ensembles,
and neural architectures provide meaningful improvements over simpler statistical
models when forecasting daily bike-rental demand.

The response variable is **`cnt`** — the total number of daily bike rentals
(casual + registered). Predictors include weather conditions, seasonal indicators,
and calendar variables from the Capital Bikeshare system in Washington, D.C.
(2011–2012).

**Eight models** are compared on predictive accuracy and model complexity:

| # | Part | Model |
|---|------|-------|
| 1 | A1 | Ordinary linear regression (baseline) |
| 2 | A1 | Additive GAM |
| 3 | A2 | GAM with temperature × windspeed tensor interaction |
| 4 | B1 | Random forest |
| 5 | B2 | XGBoost |
| 6 | C1 | Linear AR(7) |
| 7 | C2 | Feedforward neural network |
| 8 | C3 | Simple RNN |

---

## Dataset

- **Source:** [UCI Machine Learning Repository — Bike Sharing Dataset (ID 275)](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- **Access:** Programmatically via `ucimlrepo`
- **Raw granularity:** Hourly → **aggregated to daily** in the script
- **Final size:** 731 daily observations (1 Jan 2011 – 31 Dec 2012)

### Variables used

| Category | Variables | Description |
|---|---|---|
| Target | `cnt` | Total daily bike rentals |
| Weather | `temp`, `atemp`, `hum`, `windspeed`, `weathersit` | Temperature, apparent temperature, humidity, windspeed, weather condition |
| Calendar | `season`, `yr`, `mnth`, `holiday`, `weekday`, `workingday` | Seasonal and calendar indicators |
| Engineered | `doy` | Day of year (derived from `dteday`) |

---

## Train / Test Split

A **chronological split** is used (no shuffling) to avoid look-ahead bias.

| Set | Period | Observations |
|---|---|---|
| **Train** | 1 Jan 2011 – 31 Aug 2012 | 609 |
| **Test** | 1 Sep 2012 – 31 Dec 2012 | 122 |

**Cutoff date:** `2012-09-01`

All hyperparameter tuning uses `TimeSeriesSplit(n_splits=5)`.
**Random seed:** `2026` (NumPy + TensorFlow) for full reproducibility.

### Evaluation metric

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

RMSE is expressed in daily bike rentals.

---

## Results Summary

| Rank | Model | Part | Main Input | Test RMSE |
|:---:|---|:---:|---|---:|
| 1 | **XGBoost** | B2 | Full feature set (incl. `yr`, `atemp`) | **984.44** |
| 2 | Random forest | B1 | Full feature set (incl. `yr`, `atemp`) | 1105.52 |
| 3 | Linear AR(7) | C1 | 7 lags of `cnt`, `temp`, `windspeed` | 1240.58 |
| 4 | Feedforward NN | C2 | Standardised 21 lag features | 1463.83 |
| 5 | Simple RNN | C3 | 7 time steps × 3 channels | 1953.73 |
| 6 | Additive GAM | A1 | Weather, `doy`, season, weekday | 2139.40 |
| 7 | Interaction GAM | A2 | Temperature × windspeed tensor smooth | 2158.37 |
| 8 | Linear regression | A1 | Same six predictors as additive GAM | 2291.08 |

### Key improvements

- **XGBoost vs Random Forest:** ≈ **10.95%** lower RMSE
- **XGBoost vs Linear AR(7):** ≈ **20.64%** lower RMSE
- **Additive GAM vs Linear Regression:** ≈ **6.62%** lower RMSE

---

## Key Findings

**Part A — Beyond linearity**
- The additive GAM improved on linear regression, confirming that several
  weather–demand relationships are nonlinear (e.g. demand rises with temperature
  but flattens at extremes).
- Partial effects show **temperature ↑ demand**, **humidity ↓ demand**, and
  **windspeed ↓ demand**.
- The temperature × windspeed interaction **did not generalise** to the test
  period (worse RMSE) and reduced interpretability — a clear bias–variance trade-off.

**Part B — Tree ensembles**
- Tree models dominated. They captured nonlinearities, interactions, and the
  **long-term demand growth (`yr`)** automatically.
- `temp` and `atemp` were the most important features, followed by `yr`.
- XGBoost achieved the **lowest RMSE of all eight models**.

**Part C — Sequence models**
- **Linear AR(7)** was the best sequence model, showing that recent demand history
  carries strong predictive signal.
- The feedforward NN and simple RNN **overfit** on ~600 observations (see learning
  curves); validation loss flattened or rose while training loss kept falling.
- Note: sequence models use *lagged* weather, not same-day weather, which makes
  them a fairer representation of true operational forecasting.

---

## Recommendation

> **XGBoost** is the recommended primary model (lowest test RMSE, 984.44).

| Use case | Recommended model |
|---|---|
| Maximum accuracy | **XGBoost** |
| Interpretable ML alternative | **Random forest** (feature importance) |
| No weather forecast available | **Linear AR(7)** — simple, fast, transparent |
| Understanding driver effects | **Additive GAM** (partial-effect plots) |

Neural networks are **not recommended** based on this experiment due to the small
sample size and limited tuning.

---

## Repository Structure
