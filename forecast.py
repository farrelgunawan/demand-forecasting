import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json

rng = np.random.default_rng(42)
OUT = "/data/proj"
os.makedirs(f"{OUT}/charts", exist_ok=True)

# ---------------------------------------------------------------
# 1. Synthetic but realistic monthly demand (melamine tableware)
#    5 years (60 months), ending 2026-05 so the next 3 months are
#    Jun-Aug 2026 (current, forward-looking).
# ---------------------------------------------------------------
n = 60
start = "2021-06-01"
all_dates = pd.date_range(start, periods=n + 6, freq="MS")  # +6 buffer for forecasting
dates = all_dates[:n]
cal_month = all_dates.month.values

# calendar-month seasonal multipliers: Lebaran (Mar-May) + year-end (Nov-Dec)
seas = {1:0.92,2:0.95,3:1.18,4:1.25,5:1.12,6:0.98,7:0.97,8:0.96,9:0.99,10:1.03,11:1.15,12:1.22}
base, slope = 20000.0, 95.0
level_noise = 0.0
units, promo = [], []
for i, d in enumerate(dates):
    m = d.month
    p = 1 if (m in (3,4,11,12) and rng.random() < 0.5) else 0
    # persistent demand shocks (AR-like) + irregular month-to-month noise
    level_noise = 0.5*level_noise + rng.normal(0, 900)
    shock = rng.normal(0, 2500) if rng.random() < 0.12 else 0.0
    val = (base + slope*i) * seas[m] * (1 + 0.08*p) + level_noise + rng.normal(0, 700) + shock
    units.append(int(round(max(val, 1000)))); promo.append(p)
units = np.array(units, dtype=float)
df = pd.DataFrame({"date": dates, "units": units.astype(int), "avg_price_idr": (38000 + np.arange(n)*40).astype(int), "promo": promo})
df.to_csv(f"{OUT}/demand.csv", index=False)

H = 3            # forecast horizon (months ahead)
PERIOD = 12
MIN_TRAIN = 36   # rolling-origin minimum training window
Z80 = 1.2816

# ---------------------------------------------------------------
# 2. Models (implemented from scratch in NumPy)
# ---------------------------------------------------------------
def naive(train, h):
    return np.full(h, train[-1], dtype=float)

def snaive(train, h):
    y = np.asarray(train, float)
    return np.array([y[len(y)-PERIOD + (k-1)] for k in range(1, h+1)])

def ma3(train, h):
    return np.full(h, np.mean(train[-3:]), dtype=float)

def lr_forecast(train, h, month_idx):
    # month_idx: calendar months aligned to absolute positions 0..len+h-1
    y = np.asarray(train, float); nlag = 12; L = len(y)
    X, Y = [], []
    for t in range(nlag, L):
        row = [1.0, float(t)]
        md = [0.0]*11; mo = month_idx[t]
        if mo != 1: md[mo-2] = 1.0
        row += md; row.append(y[t-nlag]); X.append(row); Y.append(y[t])
    X, Y = np.array(X), np.array(Y)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    hist = list(y); fc = []
    for k in range(1, h+1):
        t = L + k - 1
        row = [1.0, float(t)]; md = [0.0]*11; mo = month_idx[t]
        if mo != 1: md[mo-2] = 1.0
        row += md; row.append(hist[t-nlag])
        v = float(np.dot(row, beta)); fc.append(v); hist.append(v)
    return np.array(fc)

def holt_winters_add(train, h, alpha, beta_, gamma):
    y = np.asarray(train, float)
    L = y[:PERIOD].mean()
    T = (y[PERIOD:2*PERIOD].mean() - y[:PERIOD].mean())/PERIOD if len(y) >= 2*PERIOD else 0.0
    S = [y[i] - L for i in range(PERIOD)]
    for t in range(len(y)):
        s = S[t % PERIOD]; lastL = L
        L = alpha*(y[t]-s) + (1-alpha)*(L+T)
        T = beta_*(L-lastL) + (1-beta_)*T
        S[t % PERIOD] = gamma*(y[t]-L) + (1-gamma)*s
    return np.array([L + k*T + S[(len(y)+k-1) % PERIOD] for k in range(1, h+1)])

# ---------------------------------------------------------------
# 3. Rolling-origin backtest (3-month-ahead)
# ---------------------------------------------------------------
def rolling(model_fn):
    recs = []
    for origin in range(MIN_TRAIN, n - H + 1):
        train = units[:origin]
        fc = model_fn(train, H)
        act = units[origin:origin+H]
        for j in range(H):
            recs.append((act[j], fc[j], origin, j))
    return recs

def metrics(recs):
    a = np.array([r[0] for r in recs]); p = np.array([r[1] for r in recs])
    mae = np.mean(np.abs(a-p)); rmse = np.sqrt(np.mean((a-p)**2)); mape = np.mean(np.abs((a-p)/a))*100
    return mae, rmse, mape

# grid search HW on backtest MAPE
best = None
for a in np.round(np.arange(0.1,0.96,0.1),2):
    for b in np.round(np.arange(0.0,0.41,0.1),2):
        for g in np.round(np.arange(0.1,0.96,0.1),2):
            recs = rolling(lambda tr,h,a=a,b=b,g=g: holt_winters_add(tr,h,a,b,g))
            _,_,mape = metrics(recs)
            if best is None or mape < best[0]:
                best = (mape,(a,b,g))
HW = best[1]

month_idx_full = cal_month  # absolute positions
models = {
    "Naive": naive,
    "Seasonal Naive": snaive,
    "Moving Average (3)": ma3,
    "Linear Reg (trend+season+lag12)": lambda tr,h: lr_forecast(tr,h,month_idx_full),
    "Holt-Winters (additive)": lambda tr,h: holt_winters_add(tr,h,*HW),
}
results = {}
for name, fn in models.items():
    results[name] = metrics(rolling(fn))

# ---------------------------------------------------------------
# 4. Final 3-month forecast with HW + 80% PI from backtest by horizon
# ---------------------------------------------------------------
hw_recs = rolling(lambda tr,h: holt_winters_add(tr,h,*HW))
std_by_h = {}
for j in range(H):
    e = np.array([r[0]-r[1] for r in hw_recs if r[3]==j])
    std_by_h[j] = e.std(ddof=1)
final_fc = holt_winters_add(units, H, *HW)
fc_dates = all_dates[n:n+H]
lo = [final_fc[j]-Z80*std_by_h[j] for j in range(H)]
hi = [final_fc[j]+Z80*std_by_h[j] for j in range(H)]

# ---------------------------------------------------------------
# 5. Charts
# ---------------------------------------------------------------
plt.rcParams.update({"figure.dpi":120,"font.size":11})
# 01 timeseries
plt.figure(figsize=(10,4))
plt.plot(dates, units, color="#0f4c75", lw=1.8)
plt.title("Permintaan Bulanan Melamine Tableware (60 bulan)")
plt.ylabel("units"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{OUT}/charts/01_timeseries.png"); plt.close()
# 02 seasonality
bym = pd.Series(units, index=dates.month).groupby(level=0).mean()
plt.figure(figsize=(9,4))
plt.bar([pd.Timestamp(2020,m,1).strftime('%b') for m in bym.index], bym.values, color="#1b9aaa")
plt.title("Rata-rata Permintaan per Bulan Kalender"); plt.ylabel("units"); plt.grid(axis='y',alpha=.3); plt.tight_layout(); plt.savefig(f"{OUT}/charts/02_seasonality.png"); plt.close()
# 03 backtest: 3-step-ahead predictions vs actual
pts = [(origin+2, holt_winters_add(units[:origin],H,*HW)[2]) for origin in range(MIN_TRAIN, n-H+1)]
bx = [dates[i] for i,_ in pts]; bp = [v for _,v in pts]
plt.figure(figsize=(10,4))
plt.plot(dates, units, color="#0f4c75", lw=1.6, label="Aktual")
plt.plot(bx, bp, color="#e4572e", lw=1.8, ls="--", marker="o", ms=3, label="Prediksi HW (3 bln ke depan)")
plt.title("Backtest Rolling-Origin: Prediksi 3 Bulan ke Depan vs Aktual")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{OUT}/charts/03_backtest.png"); plt.close()
# 04 forecast
tail = 18
plt.figure(figsize=(10,4))
plt.plot(dates[-tail:], units[-tail:], color="#0f4c75", lw=1.8, marker="o", ms=3, label="Aktual")
plt.plot(fc_dates, final_fc, color="#e4572e", lw=2, marker="s", ms=5, label="Forecast 3 bln")
plt.fill_between(fc_dates, lo, hi, color="#e4572e", alpha=.18, label="Interval 80%")
plt.title("Forecast Permintaan 3 Bulan ke Depan (Jun-Aug 2026)")
plt.ylabel("units"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{OUT}/charts/04_forecast.png"); plt.close()

# ---------------------------------------------------------------
# 6. Dump results
# ---------------------------------------------------------------
summary = {
  "hw_params": {"alpha":HW[0],"beta":HW[1],"gamma":HW[2]},
  "metrics": {k:{"MAE":round(v[0],1),"RMSE":round(v[1],1),"MAPE":round(v[2],2)} for k,v in results.items()},
  "forecast": [{"month":fc_dates[j].strftime('%Y-%m'),"point":int(round(final_fc[j])),"lo80":int(round(lo[j])),"hi80":int(round(hi[j]))} for j in range(H)],
  "naive_mape": round(results["Naive"][2],2),
  "hw_mape": round(results["Holt-Winters (additive)"][2],2),
}
with open(f"{OUT}/results.json","w") as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
