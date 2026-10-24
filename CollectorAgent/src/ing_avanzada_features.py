import numpy as np
import pandas as pd

BASE_FEATURES = [
    "sentiment_score","impact_magnitude","confidence_score","asset_exposure_score",
    "close","high","low","open","volume","sma_20","ema_20","ema_50","macd",
    "macd_signal","rsi_14","bb_upper","bb_lower","bb_width","bb_percent_b",
    "atr_14","atr_pct","obv","relative_volume","vwap","price_to_vwap",
    "last_swing_high_60","last_swing_low_60","swing_range_60","wave_strength_60",
    "fib_tp_60","fib_sl_60","elliott_phase_60","structural_sl_60","structural_tp_60",
    "last_swing_high_80","last_swing_low_80","swing_range_80","wave_strength_80",
    "fib_tp_80","fib_sl_80","elliott_phase_80","structural_sl_80","structural_tp_80",
    "last_swing_high_100","last_swing_low_100","swing_range_100","wave_strength_100",
    "fib_tp_100","fib_sl_100","elliott_phase_100","structural_sl_100","structural_tp_100",
    "dominant_elliott_phase"
]

def _agg_window(ts, s, c, m, bar_ts, N, half_life_h, max_age_h):
    left_ts = bar_ts - np.timedelta64(int(max_age_h), "h")
    left = np.searchsorted(ts, left_ts, side="left")
    right = np.searchsorted(ts, bar_ts, side="right")
    if right <= left:
        return None
    start = max(left, right - N)
    idx = np.arange(start, right)
    age = (bar_ts - ts[idx]) / np.timedelta64(1, "h")
    w = 0.5 ** (age / half_life_h)
    W = w.sum()
    if W <= 0:
        return None
    s_i, c_i, m_i = s[idx], c[idx], m[idx]
    s_agg = float((w * s_i).sum() / W)
    c_agg = float((w * c_i).sum() / W)
    m_agg = float((w * m_i).sum() / W)
    disp = float(np.sqrt((w * (s_i - s_agg) ** 2).sum() / W))
    c_adj = float(np.clip(c_agg * (1.0 - disp / 2.0), 0.0, 1.0))
    signed_impact = float((w * s_i * m_i).sum() / W)
    return {
        "sent": s_agg,
        "conf": c_agg,
        "impact": m_agg,
        "signed_impact": signed_impact,
        "disp": disp,
        "conf_adj": c_adj,
        "max_impact": float(m_i.max()),
        "age_last": float(age.min()),
        "count": int(len(idx)),
        "has_news": 1,
        "sent_x_conf": s_agg * c_adj,
        "sent_x_impact": s_agg * m_agg,
        "conf_x_impact": c_adj * m_agg,
    }

def build_news_features(news, bars, windows=(3, 5), half_life_h=12,
                        max_age_h=120, impact_tau=None):
    news = news.sort_values(["asset", "timestamp"]).copy()
    bars = bars.sort_values(["asset", "timestamp"]).copy()
    news["sentiment_raw"] = news["sentiment_raw"].clip(-1, 1)
    news["confidence_raw"] = news["confidence_raw"].clip(0, 1)
    news["impact_raw"] = news["impact_raw"].clip(lower=0)

    if impact_tau is None:
        pos = news.loc[news["impact_raw"] > 0, "impact_raw"]
        impact_tau = float(pos.median()) if len(pos) else 1.0

    news["impact_norm"] = 1.0 - np.exp(-news["impact_raw"] / impact_tau)

    out = []
    for asset, gbars in bars.groupby("asset", sort=False):
        gnews = news[news["asset"] == asset]
        ts = gnews["timestamp"].values.astype("datetime64[ns]")
        s = gnews["sentiment_raw"].values
        c = gnews["confidence_raw"].values
        m = gnews["impact_norm"].values

        for _, bar in gbars.iterrows():
            bts = np.datetime64(bar["timestamp"], "ns")
            rec = {"asset": asset, "timestamp": bar["timestamp"]}
            for N in windows:
                res = _agg_window(ts, s, c, m, bts, N, half_life_h, max_age_h)
                if res is None:
                    res = {
                        "sent": 0.0, "conf": 0.0, "impact": 0.0,
                        "signed_impact": 0.0, "disp": 0.0, "conf_adj": 0.0,
                        "max_impact": 0.0, "age_last": float(max_age_h),
                        "count": 0, "has_news": 0,
                        "sent_x_conf": 0.0, "sent_x_impact": 0.0,
                        "conf_x_impact": 0.0,
                    }
                for k, v in res.items():
                    rec[f"{k}_n{N}"] = v
            out.append(rec)

    feats = pd.DataFrame(out)
    return bars.merge(feats, on=["asset", "timestamp"], how="left")

# Uso:
# df = build_news_features(news, bars, windows=(3,5), half_life_h=12, max_age_h=120)
