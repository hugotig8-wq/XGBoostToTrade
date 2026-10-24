"""
ing_avanzada_features.py
========================
Feature engineering para noticias + barras.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Features base del clasificador (orden original del JSON). 
# ¡OJO! NO agregar 'abstract' ni textos aquí.
# ---------------------------------------------------------------------------
BASE_FEATURES = [
    "sentiment_score", "impact_magnitude", "confidence_score", "asset_exposure_score",
    "close", "high", "low", "open", "volume", "sma_20", "ema_20", "ema_50", "macd",
    "macd_signal", "rsi_14", "bb_upper", "bb_lower", "bb_width", "bb_percent_b",
    "atr_14", "atr_pct", "obv", "relative_volume", "vwap", "price_to_vwap",
    "last_swing_high_60", "last_swing_low_60", "swing_range_60", "wave_strength_60",
    "fib_tp_60", "fib_sl_60", "elliott_phase_60", "structural_sl_60", "structural_tp_60",
    "last_swing_high_80", "last_swing_low_80", "swing_range_80", "wave_strength_80",
    "fib_tp_80", "fib_sl_80", "elliott_phase_80", "structural_sl_80", "structural_tp_80",
    "last_swing_high_100", "last_swing_low_100", "swing_range_100", "wave_strength_100",
    "fib_tp_100", "fib_sl_100", "elliott_phase_100", "structural_sl_100", "structural_tp_100",
    "dominant_elliott_phase",
]

def _parse_list(x) -> list:
    if isinstance(x, list):
        return [str(t).strip().lower() for t in x if str(t).strip()]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() == "nan": return []
        try:
            v = json.loads(s)
            if isinstance(v, list): return [str(t).strip().lower() for t in v if str(t).strip()]
            if isinstance(v, str): return [v.strip().lower()]
        except (json.JSONDecodeError, ValueError):
            pass
        return [t.strip().lower() for t in s.split(",") if t.strip()]
    return []

def _jaccard(a, b) -> float:
    a, b = set(a) if a else set(), set(b) if b else set()
    return len(a & b) / len(a | b) if a or b else 0.0

def _normalize_rows(E: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return E / n

def _embed_texts_fallback(texts) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    clean = [str(t) if pd.notna(t) and str(t).strip() else "empty" for t in texts]
    vec = TfidfVectorizer(max_features=512, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(clean).toarray().astype(np.float32)
    return _normalize_rows(X)

def _agg_window(ts, s, c, m, sw, bar_ts, N, half_life_h, max_age_h):
    left_ts = bar_ts - np.timedelta64(int(max_age_h), "h")
    left = np.searchsorted(ts, left_ts, side="left")
    right = np.searchsorted(ts, bar_ts, side="right")
    if right <= left: return None
    start = max(left, right - N)
    idx = np.arange(start, right)
    age = (bar_ts - ts[idx]) / np.timedelta64(1, "h")
    w = (0.5 ** (age / half_life_h)) * np.clip(sw[idx], 0.0, None)
    W = float(w.sum())
    if W <= 0: return None
    
    s_i, c_i, m_i = s[idx], c[idx], m[idx]
    s_agg = float((w * s_i).sum() / W)
    c_agg = float((w * c_i).sum() / W)
    m_agg = float((w * m_i).sum() / W)
    disp = float(np.sqrt((w * (s_i - s_agg) ** 2).sum() / W))
    c_adj = float(np.clip(c_agg * (1.0 - disp / 2.0), 0.0, 1.0))
    signed_impact = float((w * s_i * m_i).sum() / W)
    
    return {
        "sent": s_agg, "conf": c_agg, "impact": m_agg, "signed_impact": signed_impact,
        "disp": disp, "conf_adj": c_adj, "max_impact": float(m_i.max()),
        "age_last": float(age.min()), "count": int(len(idx)), "has_news": 1,
        "sent_x_conf": s_agg * c_adj, "sent_x_impact": s_agg * m_agg, "conf_x_impact": c_adj * m_agg,
    }

_EMPTY_WINDOW = {
    "sent": 0.0, "conf": 0.0, "impact": 0.0, "signed_impact": 0.0,
    "disp": 0.0, "conf_adj": 0.0, "max_impact": 0.0,
    "age_last": None, "count": 0, "has_news": 0,
    "sent_x_conf": 0.0, "sent_x_impact": 0.0, "conf_x_impact": 0.0,
}

def normalize_news(news: pd.DataFrame, sentiment_col="sentiment_raw", confidence_col="confidence_raw", impact_col="impact_raw", impact_tau=None) -> pd.DataFrame:
    out = news.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_localize(None)
    out = out.sort_values(["asset", "timestamp"]).reset_index(drop=True)

    out["sentiment_raw"] = pd.to_numeric(out[sentiment_col], errors="coerce").fillna(0.0).clip(-1, 1)
    out["confidence_raw"] = pd.to_numeric(out[confidence_col], errors="coerce").fillna(0.0).clip(0, 1)
    out["impact_raw"] = pd.to_numeric(out[impact_col], errors="coerce").fillna(0.0).clip(lower=0)

    if impact_tau is None:
        pos = out["impact_raw"][out["impact_raw"] > 0]
        impact_tau = float(pos.median()) if len(pos) else 1.0
    out["impact_norm"] = 1.0 - np.exp(-out["impact_raw"] / max(impact_tau, 1e-9))

    out["sample_weight"] = 1.0
    if "cluster_id" not in out.columns: out["cluster_id"] = np.arange(len(out))
    if "cluster_size" not in out.columns: out["cluster_size"] = 1
    return out

# AQUÍ ESTÁ LA ACTUALIZACIÓN PARA ACEPTAR 'abstract'
def cluster_news(news: pd.DataFrame, time_window_h=6.0, sim_threshold=0.85, w_actor=0.30, w_topic=0.25, w_embed=0.30, w_asset=0.15, text_cols=("title", "summary", "abstract")) -> pd.DataFrame:
    out = news.sort_values(["asset", "timestamp"]).reset_index(drop=True).copy()
    actors = out["actors_raw"].apply(_parse_list).tolist() if "actors_raw" in out.columns else [[] for _ in range(len(out))]
    topics = out["topics_raw"].apply(_parse_list).tolist() if "topics_raw" in out.columns else [[] for _ in range(len(out))]

    texts = []
    for _, row in out.iterrows():
        # Concatena las columnas de texto que existan (title, summary, abstract)
        parts = [str(row.get(col, "")) for col in text_cols if col in row and pd.notna(row.get(col))]
        texts.append(" ".join(parts).strip())
    E = _embed_texts_fallback(texts)

    n = len(out)
    cluster_ids = -np.ones(n, dtype=int)
    cents_ts, cents_E, cents_actors, cents_topics, cents_asset, sizes = [], [], [], [], [], []
    ts = out["timestamp"].values.astype("datetime64[ns]")
    assets = out["asset"].values

    for i in range(n):
        best_c, best_sim = -1, -1.0
        for c in range(len(cents_ts)):
            if cents_asset[c] != assets[i]: continue
            dt_h = (ts[i] - cents_ts[c]) / np.timedelta64(1, "h")
            if dt_h < 0 or dt_h > time_window_h: continue
            sim_a = _jaccard(actors[i], cents_actors[c])
            sim_t = _jaccard(topics[i], cents_topics[c])
            sim_e = float(E[i] @ cents_E[c])
            sim = w_actor * sim_a + w_topic * sim_t + w_embed * sim_e + w_asset * 1.0
            if sim > best_sim: best_sim, best_c = sim, c

        if best_sim >= sim_threshold and best_c >= 0:
            cluster_ids[i] = best_c
            k = sizes[best_c] + 1
            cents_E[best_c] = (cents_E[best_c] * (k - 1) + E[i]) / k
            cents_actors[best_c] = cents_actors[best_c] | set(actors[i])
            cents_topics[best_c] = cents_topics[best_c] | set(topics[i])
            sizes[best_c] = k
        else:
            new_id = len(cents_ts)
            cluster_ids[i] = new_id
            cents_ts.append(ts[i]); cents_E.append(E[i].copy())
            cents_actors.append(set(actors[i])); cents_topics.append(set(topics[i]))
            cents_asset.append(assets[i]); sizes.append(1)

    out["cluster_id"] = cluster_ids
    out["cluster_size"] = [sizes[c] for c in cluster_ids]
    out["sample_weight"] = 1.0 / out["cluster_size"].astype(float)
    return out.sort_values(["asset", "timestamp"]).reset_index(drop=True)

def collapse_clusters(news: pd.DataFrame, method="max_conf") -> pd.DataFrame:
    rows = []
    for cid, g in news.groupby("cluster_id", sort=False):
        if method == "max_conf": rep = g.loc[g["confidence_raw"].idxmax()]
        elif method == "mean":
            rep = g.iloc[0].copy()
            rep["sentiment_raw"], rep["confidence_raw"], rep["impact_raw"] = g["sentiment_raw"].mean(), g["confidence_raw"].mean(), g["impact_raw"].mean()
            rep["impact_norm"], rep["timestamp"] = g["impact_norm"].mean(), g["timestamp"].min()
        rep = rep.copy()
        rep["cluster_size"] = len(g)
        rep["sample_weight"] = 1.0
        rows.append(rep)
    return pd.DataFrame(rows).sort_values(["asset", "timestamp"]).reset_index(drop=True)

def build_news_features(news: pd.DataFrame, bars: pd.DataFrame, windows=(3, 5), half_life_h=12.0, max_age_h=120.0, impact_tau=None) -> pd.DataFrame:
    news = normalize_news(news, impact_tau=impact_tau)
    bars = bars.sort_values(["asset", "timestamp"]).reset_index(drop=True).copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_localize(None)
    out_rows = []

    for asset, gbars in bars.groupby("asset", sort=False):
        gnews = news[news["asset"] == asset].sort_values("timestamp")
        if len(gnews) == 0:
            for _, bar in gbars.iterrows():
                rec = {"asset": asset, "timestamp": bar["timestamp"]}
                for N in windows:
                    for k, v in _EMPTY_WINDOW.items(): rec[f"{k}_n{N}"] = float(max_age_h) if k == "age_last" else v
                out_rows.append(rec)
            continue

        ts = gnews["timestamp"].values.astype("datetime64[ns]")
        s, c, m = gnews["sentiment_raw"].values, gnews["confidence_raw"].values, gnews["impact_norm"].values
        sw = gnews["sample_weight"].values if "sample_weight" in gnews.columns else np.ones(len(gnews))

        for _, bar in gbars.iterrows():
            bts = np.datetime64(bar["timestamp"], "ns")
            rec = {"asset": asset, "timestamp": bar["timestamp"]}
            for N in windows:
                res = _agg_window(ts, s, c, m, sw, bts, N, half_life_h, max_age_h)
                if res is None:
                    for k, v in _EMPTY_WINDOW.items(): rec[f"{k}_n{N}"] = float(max_age_h) if k == "age_last" else v
                else:
                    for k, v in res.items(): rec[f"{k}_n{N}"] = v
            out_rows.append(rec)

    return bars.merge(pd.DataFrame(out_rows), on=["asset", "timestamp"], how="left")
