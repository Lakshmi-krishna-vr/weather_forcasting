"""
Chaos-Aware Explainable Adaptive Weather Forecasting — Streamlit app (v3, multi-variable LSTM→GRU)
Forecasts temperature, pressure, relative humidity and wind speed 1 hour ahead (6 × 10 min),
scores how chaotic the current weather is, adapts the forecast horizon, and shows explainability results.

Run locally:   pip install -r requirements.txt   then   streamlit run app.py
"""
import json, os, urllib.request, urllib.parse
from datetime import timedelta
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
def P(name): return os.path.join(BASE, name)

st.set_page_config(page_title="Chaos-Aware Weather Forecasting", page_icon="🌦️", layout="wide")

NAMES = {"T": "Temperature", "p": "Pressure", "rh": "Relative humidity", "wv": "Wind speed"}
RENAME = {"Date Time": "date", "T (degC)": "T", "p (mbar)": "p", "rh (%)": "rh",
          "Tdew (degC)": "Tdew", "wv (m/s)": "wv", "wd (deg)": "wd"}
RAW_COLS = ["T", "p", "rh", "Tdew", "wv", "wd", "rain", "SWDR"]
POLICY = {"Stable": (6, "10 minutes"), "Moderately Unpredictable": (3, "5 minutes"), "Highly Unpredictable": (1, "1 minute")}
LOCATIONS = {   # name: (lat, lon)
    "Jena, Germany (training station)": (50.9271, 11.5892),
    "Kottayam, Kerala": (9.5916, 76.5222),
    "Kochi, Kerala": (9.9312, 76.2673),
    "Thiruvananthapuram, Kerala": (8.5241, 76.9366),
    "Custom latitude / longitude": None,
}
STATE_COLOR = {"Stable": "#2e7d32", "Moderately Unpredictable": "#ef6c00", "Highly Unpredictable": "#c62828"}


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading model…")
def load_assets():
    import tensorflow as tf
    cfg = json.load(open(P("config.json")))
    model = tf.keras.models.load_model(P("weather_lstm_gru_multivar.keras"), compile=False)
    return (cfg, model, joblib.load(P("feature_scaler.pkl")),
            joblib.load(P("target_scaler.pkl")), joblib.load(P("chaos_scaler.pkl")))

@st.cache_data
def load_csv(name, **kw):
    path = P(name)
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


# ----------------------------------------------------------------------------- live data (Open-Meteo, free, no API key)
OM_VARS = {"temperature_2m": "T", "surface_pressure": "p", "relative_humidity_2m": "rh",
           "dew_point_2m": "Tdew", "wind_speed_10m": "wv", "wind_direction_10m": "wd"}

@st.cache_data(ttl=600, show_spinner="Fetching live weather…")
def fetch_live(lat, lon):
    """Last ~28 h of 15-minute readings + the next 2 h of Open-Meteo's own forecast, in local time."""
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "minutely_15": ",".join(OM_VARS), "current": "temperature_2m",
        "past_minutely_15": 28 * 4, "forecast_minutely_15": 8, "wind_speed_unit": "ms", "timezone": "auto"})
    with urllib.request.urlopen(f"https://api.open-meteo.com/v1/forecast?{q}", timeout=20) as r:
        js = json.load(r)
    return parse_live(js)

def parse_live(js):
    m = js["minutely_15"]
    df = pd.DataFrame({"date": pd.to_datetime(m["time"])})
    for k, v in OM_VARS.items(): df[v] = pd.to_numeric(pd.Series(m[k]), errors="coerce")
    now = pd.Timestamp(js["current"]["time"])
    obs = df[df.date <= now].dropna(subset=list(OM_VARS.values()), how="all").reset_index(drop=True)
    nwp = df[df.date > now].reset_index(drop=True)
    return obs, nwp, js.get("timezone", ""), now


# ----------------------------------------------------------------------------- preprocessing (same as the notebook)
def prepare(raw):
    df = raw.rename(columns={c: RENAME.get(str(c).strip(), str(c).strip()) for c in raw.columns})
    if "date" not in df.columns:
        raise ValueError("The CSV needs a 'date' (or 'Date Time') column.")
    fmt = "%d.%m.%Y %H:%M:%S" if df["date"].astype(str).str.match(r"^\d{2}\.\d{2}\.\d{4}").all() else None
    df["date"] = pd.to_datetime(df["date"], format=fmt, errors="coerce")
    keep = ["date"] + [c for c in RAW_COLS if c in df.columns]
    df = df[keep].dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    num = keep[1:]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "wv" in df: df.loc[df["wv"] < 0, "wv"] = 0.0
    if "rh" in df: df["rh"] = df["rh"].clip(0, 100)
    df = df.set_index("date").resample("10min").mean().interpolate(limit_direction="both").reset_index()
    return df

def add_features(df):
    df = df.copy()
    if {"wv", "wd"} <= set(df.columns):
        rad = np.deg2rad(df["wd"])
        df["Wx"] = df["wv"] * np.cos(rad); df["Wy"] = df["wv"] * np.sin(rad)
    hour = df.date.dt.hour + df.date.dt.minute / 60
    doy = df.date.dt.dayofyear
    df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    df["doy_sin"], df["doy_cos"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    df["temperature_rolling_mean"] = df["T"].rolling(6).mean()
    df["temperature_rolling_std"] = df["T"].rolling(6).std()
    df["pressure_rolling_mean"] = df["p"].rolling(6).mean()
    df["humidity_rolling_mean"] = df["rh"].rolling(6).mean()
    df["temperature_change"] = df["T"].diff()
    df["temperature_change_1h"] = df["T"].diff(6)
    df["pressure_change_1h"] = df["p"].diff(6)
    df["humidity_change"] = df["rh"].diff()
    if "wv" in df:
        df["wind_rolling_mean"] = df["wv"].rolling(6).mean()
        df["wind_rolling_std"] = df["wv"].rolling(6).std()
        df["wind_change_1h"] = df["wv"].diff(6)
    return df.dropna().reset_index(drop=True)


# ----------------------------------------------------------------------------- model, chaos, confidence
def forecast(i, df, cfg, model, scaler, target_scaler):
    L, S, targets = cfg["input_steps"], cfg["output_steps"], cfg["targets"]
    window = scaler.transform(df[cfg["feature_columns"]].iloc[i - L + 1 : i + 1]).astype("float32")[None]
    delta = target_scaler.inverse_transform(np.asarray(model.predict(window, verbose=0))).reshape(S, len(targets))
    pred = df[targets].iloc[i].values[None, :].astype(float) + delta
    for j, v in enumerate(targets):
        if v in cfg.get("phys_limits", {}):
            lo, hi = cfg["phys_limits"][v]
            pred[:, j] = np.clip(pred[:, j], lo, hi)
    return pred                                                    # (steps, targets)

def _safe(fn, x):
    try:
        v = fn(np.asarray(x, float)); return float(v) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0

def _acf1(x):
    v = np.corrcoef(x[:-1], x[1:])[0, 1]
    return float(v) if np.isfinite(v) else 0.0

def chaos_now(i, df, cfg, chaos_scaler):
    import nolds
    L, targets = cfg["input_steps"], cfg["targets"]
    raw = {}
    for v in targets:
        w = df[v].iloc[i - L + 1 : i + 1].values
        raw.update({f"{v}_rolling_std": np.std(w), f"{v}_autocorrelation": _acf1(w),
                    f"{v}_sample_entropy": _safe(nolds.sampen, w), f"{v}_lyapunov_exponent": _safe(nolds.lyap_r, w)})
    cols = [f"{v}_{m}" for v in targets for m in ["rolling_std", "sample_entropy", "lyapunov_exponent"]]
    s = dict(zip(cols, chaos_scaler.transform(pd.DataFrame([raw])[cols]).clip(0, 1)[0]))
    per = {}
    for v in targets:
        acf = (np.clip(raw[f"{v}_autocorrelation"], -1, 1) + 1) / 2
        per[v] = float(np.clip(0.30*s[f"{v}_rolling_std"] + 0.25*s[f"{v}_sample_entropy"]
                               + 0.25*s[f"{v}_lyapunov_exponent"] + 0.20*(1 - acf), 0, 1))
    return float(np.mean(list(per.values()))), per, raw

def classify(score, cfg):
    lo, hi = cfg["chaos_thresholds"]
    return "Stable" if score < lo else ("Moderately Unpredictable" if score < hi else "Highly Unpredictable")

def confidence(v, score_v, cfg):
    mae, sd = cfg["model_mae"][v], cfg["error_std"][v]
    return float(np.clip(100 / (1 + mae / (sd + 1e-8)) * (1 - 0.5 * score_v), 0, 100))

def alerts(score, cfg, fc, row):
    out = []
    if score >= cfg["chaos_thresholds"][1]: out.append("High unpredictability detected — shorter horizon, faster monitoring.")
    if row.get("pressure_change_1h", 0) < -2: out.append("Rapid pressure drop in the last hour (possible front / storm).")
    if abs(row.get("temperature_change_1h", 0)) > 3: out.append("Rapid temperature change in the last hour.")
    if "wv" in fc and max(fc["wv"]) > 10: out.append("Strong wind forecast (> 10 m/s).")
    if "rh" in fc and max(fc["rh"]) > 95: out.append("Near-saturation humidity forecast (fog / rain likely).")
    if "p" in fc and fc["p"][-1] - fc["p"][0] < -1: out.append("Pressure forecast to fall > 1 mbar within the hour.")
    return out


# ----------------------------------------------------------------------------- UI
def main():
    try:
        cfg, model, scaler, target_scaler, chaos_scaler = load_assets()
    except Exception as e:
        st.error(f"Could not load the model files next to app.py: {e}")
        st.stop()
    targets, units, L, S = cfg["targets"], cfg["units"], cfg["input_steps"], cfg["output_steps"]

    st.title("🌦️ Chaos-Aware Explainable Adaptive Weather Forecasting")
    st.caption(f"Hybrid LSTM→GRU · forecasts {', '.join(NAMES.get(v, v) for v in targets)} for the next "
               f"{S * 10} minutes from the last {L // 6} hours · dataset: {cfg.get('dataset', '')} ({cfg.get('split', '')})")

    # ---- sidebar: data source and forecast time
    st.sidebar.header("Input data")
    source = st.sidebar.radio("Data source", ["Live (today)", "Demo (test-year data)", "Upload CSV"])
    raw, live, nwp = None, False, None
    if source == "Live (today)":
        loc = st.sidebar.selectbox("Location", list(LOCATIONS))
        if LOCATIONS[loc] is None:
            lat = st.sidebar.number_input("Latitude", value=9.5916, format="%.4f")
            lon = st.sidebar.number_input("Longitude", value=76.5222, format="%.4f")
        else:
            lat, lon = LOCATIONS[loc]
        if st.sidebar.button("🔄 Refresh now"): fetch_live.clear()
        try:
            raw, nwp, tzname, now = fetch_live(lat, lon)
            live = True
            st.sidebar.success(f"Live data up to {now:%d %b %Y %H:%M} ({tzname})")
            if not loc.startswith("Jena"):
                st.sidebar.warning("The model was trained on Jena (Germany). For other climates, e.g. Kerala, "
                                   "treat the forecast as experimental — values outside the training range reduce accuracy.")
        except Exception as e:
            st.sidebar.error(f"Live data unavailable ({e}). Showing demo data instead.")
    elif source == "Upload CSV":
        up = st.sidebar.file_uploader("10-minute weather CSV", type="csv",
                                      help="Columns: date, T, p, rh, Tdew, wv, wd (+ rain, SWDR if the model was trained "
                                           "with them). Jena-style headers also work. Needs ≥ 25 hours of data.")
        if up is not None: raw = pd.read_csv(up)
        else: st.sidebar.info("Upload a CSV, or switch back to the demo data.")
    if raw is None:
        raw = load_csv("demo_history.csv")
        if raw is None:
            st.error("demo_history.csv not found next to app.py."); st.stop()

    try:
        df = add_features(prepare(raw))
    except Exception as e:
        st.error(f"Could not process the data: {e}"); st.stop()
    missing = [c for c in cfg["feature_columns"] if c not in df.columns]
    if missing:
        st.error(f"The data is missing columns the model needs: {missing}"); st.stop()
    if len(df) < L:
        st.error(f"Need at least {L} rows (24 h) after preprocessing; got {len(df)}."); st.stop()

    t_min, t_max = df.date.iloc[L - 1].to_pydatetime(), df.date.iloc[-1].to_pydatetime()
    default_t = df.date.iloc[max(L - 1, len(df) - 1 - S)].to_pydatetime()
    if live:
        t_sel = t_max                                                   # live: forecast from the latest reading
        st.sidebar.caption("Live mode: the forecast is issued from the latest reading (now).")
    else:
        t_sel = st.sidebar.slider("Forecast issued at", min_value=t_min, max_value=t_max, value=default_t,
                                  step=timedelta(minutes=10), format="YYYY-MM-DD HH:mm",
                                  help="The moment treated as 'now': the model reads the 24 h before it and forecasts the next hour.")
    i = int(np.clip(df.date.searchsorted(pd.Timestamp(t_sel), side="right") - 1, L - 1, len(df) - 1))
    adaptive = st.sidebar.toggle("Chaos-adaptive horizon", value=True,
                                 help="Shorten the forecast horizon when the weather is unpredictable.")

    # ---- compute
    pred = forecast(i, df, cfg, model, scaler, target_scaler)
    score, per_chaos, raw_chaos = chaos_now(i, df, cfg, chaos_scaler)
    state = classify(score, cfg)
    horizon, monitor = POLICY[state] if adaptive else (S, "10 minutes")
    future_times = pd.date_range(df.date.iloc[i] + pd.Timedelta(minutes=10), periods=S, freq="10min")
    actual = df.set_index("date")[targets].reindex(future_times)
    ref = None
    if live and nwp is not None and len(nwp):
        both = pd.concat([raw[["date"] + targets].tail(1), nwp[["date"] + targets]]).set_index("date")
        ref = both.reindex(both.index.union(future_times)).interpolate(method="time").reindex(future_times)

    tabs = st.tabs(["Forecast", "Model performance", "Chaos analysis", "Explainability", "About"])

    # ================= Forecast
    with tabs[0]:
        c = st.columns(4)
        c[0].markdown(f"**{'Now (live)' if live else 'Forecast issued at'}**<br><span style='font-size:1.4rem;font-weight:600'>"
                      f"{df.date.iloc[i]:%d %b %Y, %H:%M}</span>", unsafe_allow_html=True)
        c[1].markdown(f"**Weather state**<br><span style='color:{STATE_COLOR[state]};font-size:1.4rem;font-weight:600'>"
                      f"{state}</span>", unsafe_allow_html=True)
        c[2].metric("Overall chaos score", f"{score:.3f}",
                    help=f"Thresholds: Stable < {cfg['chaos_thresholds'][0]:.3f} ≤ Moderate < {cfg['chaos_thresholds'][1]:.3f} ≤ Highly")
        c[3].metric("Horizon · monitoring", f"{horizon * 10} min", monitor, delta_color="off")

        st.subheader(f"Forecast for the next {horizon * 10} minutes")
        c = st.columns(len(targets))
        for j, v in enumerate(targets):
            now, end = float(df[v].iloc[i]), float(pred[horizon - 1, j])
            c[j].metric(f"{NAMES.get(v, v)} (+{horizon * 10} min)", f"{end:.2f} {units[v]}",
                        f"{end - now:+.2f} {units[v]} vs now")
            c[j].caption(f"Confidence {confidence(v, per_chaos[v], cfg):.1f}% · chaos {per_chaos[v]:.2f}")

        hist = df.iloc[i - L + 1 : i + 1]
        fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 3.4))
        axes = np.atleast_1d(axes)
        for j, (ax, v) in enumerate(zip(axes, targets)):
            ax.plot(hist.date, hist[v], lw=1.2, label="last 24 h")
            ax.plot(future_times[:horizon], pred[:horizon, j], "o-", lw=1.6, ms=3, label="forecast")
            if actual[v].notna().any():
                ax.plot(future_times[:horizon], actual[v].values[:horizon], "x--", lw=1, label="actual")
            if ref is not None and v in ref:
                ax.plot(future_times[:horizon], ref[v].values[:horizon], "s:", lw=1, ms=3, label="Open-Meteo forecast")
            ax.set_title(f"{NAMES.get(v, v)} ({units[v]})"); ax.grid(alpha=.3); ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        axes[0].legend(fontsize=8); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        table = pd.DataFrame(pred[:horizon], columns=[f"{NAMES.get(v, v)} ({units[v]})" for v in targets],
                             index=[t.strftime("%H:%M") for t in future_times[:horizon]])
        if actual.notna().all(axis=None):
            for j, v in enumerate(targets):
                table[f"actual {v}"] = actual[v].values[:horizon]
        if ref is not None:
            for v in targets:
                if v in ref: table[f"Open-Meteo {v}"] = ref[v].values[:horizon]
            st.caption("Open-Meteo forecast = a professional numerical weather model, shown for comparison.")
        st.dataframe(table.round(2), use_container_width=True)

        fc = {v: pred[:horizon, j].tolist() for j, v in enumerate(targets)}
        al = alerts(score, cfg, fc, df.iloc[i].to_dict())
        st.subheader("Alerts")
        if al:
            for a in al: st.warning(a)
        else:
            st.success("No major alerts.")

    # ================= Model performance
    with tabs[1]:
        res = load_csv("test_results.csv")
        if res is not None:
            st.subheader("Test-year results (unseen data)")
            st.dataframe(res.round(4), use_container_width=True, hide_index=True)
            sk = res[res.model.isin(["Ridge", "LSTM→GRU"])].pivot(index="variable", columns="model", values="Skill vs persistence")
            sk = sk.reindex([v for v in targets if v in sk.index])
            fig, ax = plt.subplots(figsize=(8, 3.4))
            (100 * sk).plot.bar(ax=ax, rot=0); ax.axhline(0, c="k", lw=.8)
            ax.set_ylabel("skill vs persistence (%)"); ax.set_title("Skill score by variable (higher is better)"); ax.grid(alpha=.3, axis="y")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        ph = load_csv("per_horizon_mae.csv", header=[0, 1], index_col=0)
        if ph is not None:
            st.subheader("MAE by forecast horizon")
            st.dataframe(ph.round(3), use_container_width=True)
        tl = load_csv("training_log.csv")
        if tl is not None and {"loss", "val_loss"} <= set(tl.columns):
            st.subheader("Training curve")
            st.line_chart(tl.set_index("epoch")[["loss", "val_loss"]] if "epoch" in tl else tl[["loss", "val_loss"]])

    # ================= Chaos analysis
    with tabs[2]:
        st.subheader("Chaos metrics for the current 24 h window")
        rows = [{"variable": NAMES.get(v, v), "volatility σ": raw_chaos[f"{v}_rolling_std"],
                 "lag-1 autocorr": raw_chaos[f"{v}_autocorrelation"], "sample entropy": raw_chaos[f"{v}_sample_entropy"],
                 "Lyapunov λ": raw_chaos[f"{v}_lyapunov_exponent"], "chaos score": per_chaos[v]} for v in targets]
        st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)
        ct = load_csv("chaos_metrics_test.csv", parse_dates=["date"])
        if ct is not None:
            fig, ax = plt.subplots(figsize=(14, 3.4))
            ax.plot(ct.date, ct.chaos_score, lw=.8, c="k")
            for thr, col in zip(cfg["chaos_thresholds"], ["green", "red"]): ax.axhline(thr, ls="--", c=col)
            ax.set_title("Overall chaos score over the test year (every 6 h)"); ax.grid(alpha=.3)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.write("Share of test periods per regime:",
                     (100 * ct.weather_state.value_counts(normalize=True)).round(1).astype(str).add(" %").to_dict())
        br = load_csv("error_by_chaos_regime.csv", header=[0, 1], index_col=0)
        if br is not None:
            st.subheader("Forecast error by chaos regime (test year)")
            st.dataframe(br.round(3), use_container_width=True)

    # ================= Explainability
    with tabs[3]:
        sh = load_csv("shap_importance.csv", index_col=0)
        pm = load_csv("permutation_importance.csv", index_col=0)
        c = st.columns(2)
        if sh is not None:
            c[0].subheader("SHAP — share of mean |SHAP| (%)")
            c[0].dataframe(sh.head(15).rename(columns=NAMES).style.background_gradient(cmap="Blues").format("{:.1f}"),
                           use_container_width=True)
        if pm is not None:
            c[1].subheader("Permutation — MAE increase (%)")
            c[1].dataframe(pm.head(15).rename(columns=NAMES).style.background_gradient(cmap="Reds").format("{:.1f}"),
                           use_container_width=True)
        if sh is None and pm is None:
            st.info("No explainability files found.")

    # ================= About
    with tabs[4]:
        st.markdown(f"""
**Model.** {cfg.get('architecture', f"Input (144 × {len(cfg['feature_columns'])}) → LSTM(64) → LayerNorm → Dropout → GRU(48) → Dropout → Dense(128) → Dense({S} × {len(targets)})")}. The network predicts the **change** from the last observation for every variable
(residual target), so persistence is its starting point and it only learns the correction.

**Chaos score.** For each variable over the last 24 h: volatility, lag-1 autocorrelation, sample entropy and the largest
Lyapunov exponent, min–max scaled on training data and combined as
C = 0.30σ + 0.25·SampEn + 0.25·λ + 0.20·(1 − ρ). The overall score is the mean over variables.

**Adaptive policy.** Stable → 60 min horizon · Moderately unpredictable → 30 min · Highly unpredictable → 10 min.

**Dataset.** Jena Climate 2009–2016, Max Planck Institute for Biogeochemistry
([Kaggle](https://www.kaggle.com/datasets/mnassrib/jena-climate)).

**Features used:** {', '.join(cfg['feature_columns'])}
""")


if __name__ == "__main__":
    main()
