# Demand Forecasting — Melamine Tableware (3-Month Operational Forecast)

A short-horizon (3-month-ahead) demand forecast to drive production planning (PPIC),
raw-material procurement, and safety-stock decisions in a melamine tableware factory.

## Why a 3-month horizon?
Most procurement and production-planning decisions in this factory operate on a rolling
quarter: the imported melamine powder has a ~21-day lead time, and shift/material plans are
locked 4–12 weeks out. A 3-month-ahead forecast matches the actual decision window, so the
error metric reflects decisions the planner really makes.

## Method
- Data: 60 months of monthly demand (trend + Lebaran Mar–May and year-end Nov–Dec seasonality + demand shocks).
- Models implemented from scratch in NumPy: Naive, Seasonal Naive, Moving Average(3),
  Linear Regression (trend + month dummies + lag-12), Holt-Winters additive (grid-searched).
- Validation: **rolling-origin (walk-forward) backtest** of the 3-month-ahead forecast across
  all origins from month 36 onward — not a single holdout — so the metric is averaged over many
  forecast windows and is far more honest than a one-shot test.
- Selection: best backtest MAPE, refit on the full series, forecast next 3 months (Jun–Aug 2026)
  with an 80% prediction interval derived from per-horizon backtest residuals.

## Results (rolling-origin backtest, 3-month-ahead)
Holt-Winters additive won on every metric; ~61% more accurate than the naive baseline.
See `results.json` for the full table and the generated forecast.

## Files
- `forecast.py` — full pipeline (data, models, backtest, charts).
- `demand.csv` — the dataset.
- `results.json` — metrics + 3-month forecast.
- `charts/` — timeseries, seasonality, backtest, forecast PNGs.

## To turn this into real portfolio proof
Swap the synthetic data for a public dataset (e.g. Kaggle “Store Item Demand Forecasting”)
or your own sanitized factory data, push to GitHub with this README, and link from your CV/LinkedIn.
