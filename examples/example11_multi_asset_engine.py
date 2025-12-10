from textwrap import dedent
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import base64
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import tpqoa
from tavily import TavilyClient

from agno.agent import Agent
from agno.models.openai import OpenAIResponses


# --------------------------------------------------------------------
# 0. Asset configuration (Hybrid Engine)
# --------------------------------------------------------------------
ASSETS = {
    # FX majors / EM
    "EURUSD": {
        "instrument": "EUR_USD",
        "label": "EUR/USD",
        "tavily_query": "latest macro and FX news about EURUSD euro dollar currency pair",
    },
    "USDMXN": {
        "instrument": "USD_MXN",
        "label": "USD/MXN",
        "tavily_query": "latest macro and FX news about USD/MXN US dollar Mexican peso Banxico Federal Reserve carry trade",
    },
    # Equity index (global risk proxy)
    "SPX": {
        "instrument": "SPX500_USD",
        "label": "US SPX 500",
        "tavily_query": "latest macro and market news about the S&P 500 US stock market risk sentiment earnings Fed policy",
    },
    # Gold (macro hedge)
    "GOLD": {
        "instrument": "XAU_USD",
        "label": "Gold (XAU/USD)",
        "tavily_query": "latest macro and commodities news about gold XAU USD inflation real yields safe haven demand",
    },
}


# --------------------------------------------------------------------
# 1. Data access
# --------------------------------------------------------------------
def get_oanda_history(
    instrument: str = "EUR_USD",
    start: str = "2023-01-01",
    end: str = "2023-01-10",
    granularity: str = "D",
    price: str = "M",
):
    """
    Fetch OHLCV data from Oanda via tpqoa.
    Returns a DataFrame indexed by datetime.
    """
    config_path = os.getenv("OANDA_CFG_PATH", os.path.expanduser("~/oanda.cfg"))
    api = tpqoa.tpqoa(config_path)
    df = api.get_history(
        instrument=instrument,
        start=start,
        end=end,
        granularity=granularity,
        price=price,
    )

    # Normalize column names
    df = df.copy()
    col_map = {}
    if "o" in df.columns:
        col_map["o"] = "open"
    if "h" in df.columns:
        col_map["h"] = "high"
    if "l" in df.columns:
        col_map["l"] = "low"
    if "c" in df.columns:
        col_map["c"] = "close"
    if "v" in df.columns:
        col_map["v"] = "volume"
    df = df.rename(columns=col_map)

    return df


# --------------------------------------------------------------------
# 2. Quant features
# --------------------------------------------------------------------
def rsi_wilder(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Classic Wilder RSI implementation.
    """
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)

    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def add_quant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic quant features + moving averages + Bollinger bands.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    out = df.copy()
    price = out["close"]

    # Returns
    out["ret"] = price.pct_change()

    # Realized vol (annualised)
    out["rv_10"] = out["ret"].rolling(10).std() * np.sqrt(252)
    out["rv_20"] = out["ret"].rolling(20).std() * np.sqrt(252)

    # ATR(14)
    if {"high", "low", "close"}.issubset(out.columns):
        high = out["high"]
        low = out["low"]
        close = out["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        out["atr_14"] = tr.rolling(14).mean()
    else:
        out["atr_14"] = np.nan

    # RSI
    out["rsi_14"] = rsi_wilder(price, window=14)

    # Regression slopes
    def rolling_slope(x: pd.Series) -> float:
        idx = np.arange(len(x))
        if len(x.dropna()) < len(x):
            return np.nan
        coef = np.polyfit(idx, x.values, 1)
        return coef[0]

    out["slope_5"] = price.rolling(5).apply(rolling_slope, raw=False)
    out["slope_20"] = price.rolling(20).apply(rolling_slope, raw=False)

    # Moving averages
    out["ma_20"] = price.rolling(20).mean()
    out["ma_50"] = price.rolling(50).mean()

    # Bollinger bands
    roll_std_20 = price.rolling(20).std()
    out["bb_upper"] = out["ma_20"] + 2 * roll_std_20
    out["bb_lower"] = out["ma_20"] - 2 * roll_std_20
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["ma_20"]

    return out


# --------------------------------------------------------------------
# 2a. Signal Strength Score (0–100)
# --------------------------------------------------------------------
def compute_signal_score(row: pd.Series) -> float:
    """
    Compute a 0–100 directional signal score:

    - Higher scores → more bullish.
    - Lower scores  → more bearish.
    - ~50           → neutral.
    """
    # 1) RSI component
    rsi = row.get("rsi_14", np.nan)
    if np.isnan(rsi):
        rsi_score = 50.0
    else:
        # RSI < 50 → bullish, RSI > 50 → bearish
        rsi_dir = (50.0 - rsi) / 25.0
        rsi_dir = max(-1.0, min(1.0, rsi_dir))
        rsi_score = 50.0 + 30.0 * rsi_dir
        rsi_score = max(0.0, min(100.0, rsi_score))

    # 2) Trend component (slopes)
    slope5 = row.get("slope_5", 0.0)
    slope20 = row.get("slope_20", 0.0)
    denom = 0.002 if 0.002 != 0 else 1.0
    trend_raw = slope5 + 0.5 * slope20
    trend_norm = trend_raw / denom
    trend_norm = max(-1.0, min(1.0, trend_norm))
    trend_score = 50.0 + 30.0 * trend_norm
    trend_score = max(0.0, min(100.0, trend_score))

    # 3) Volatility regime component (Bollinger width)
    bb_width = row.get("bb_width", np.nan)
    if np.isnan(bb_width):
        vol_score = 50.0
    else:
        vol_norm = (bb_width - 0.01) / 0.07
        vol_norm = max(-1.0, min(1.0, vol_norm))
        vol_score = 50.0 + 10.0 * vol_norm
        vol_score = max(0.0, min(100.0, vol_score))

    # 4) Bollinger band position component
    close = row.get("close", np.nan)
    bb_upper = row.get("bb_upper", np.nan)
    bb_lower = row.get("bb_lower", np.nan)

    if np.isnan(close) or np.isnan(bb_upper) or np.isnan(bb_lower) or bb_upper == bb_lower:
        band_score = 50.0
    else:
        pos = (close - bb_lower) / (bb_upper - bb_lower)
        pos = max(0.0, min(1.0, pos))
        band_dir = (0.5 - pos) / 0.5
        band_dir = max(-1.0, min(1.0, band_dir))
        band_score = 50.0 + 30.0 * band_dir
        band_score = max(0.0, min(100.0, band_score))

    score = (
        0.35 * rsi_score +
        0.35 * trend_score +
        0.15 * vol_score +
        0.15 * band_score
    )
    return float(round(score, 2))


# --------------------------------------------------------------------
# 2b. Regime Classification
# --------------------------------------------------------------------
def classify_regime(df: pd.DataFrame) -> str:
    df_q = df.dropna(subset=["close", "rsi_14", "rv_10", "rv_20",
                             "ma_20", "bb_upper", "bb_lower", "bb_width"])
    if df_q.empty:
        return "Regime unavailable (insufficient data)."

    last = df_q.iloc[-1]

    close = last["close"]
    rsi = last["rsi_14"]
    slope20 = last["slope_20"]
    rv10 = last["rv_10"]
    rv20 = last["rv_20"]
    ma20 = last["ma_20"]
    bb_u = last["bb_upper"]
    bb_l = last["bb_lower"]
    bb_w = last["bb_width"]

    mid_band = (bb_u + bb_l) / 2

    slope_trend = 0.00020
    bb_compress = 0.02
    bb_expansion = 0.06

    if (
        slope20 > slope_trend and
        close > ma20 and
        rsi > 55 and
        close > mid_band
    ):
        return "Trending Up"

    if (
        slope20 < -slope_trend and
        close < ma20 and
        rsi < 45 and
        close < mid_band
    ):
        return "Trending Down"

    if (
        bb_w < bb_compress and
        rv10 < rv20 and
        40 < rsi < 60
    ):
        return "Volatility Compression (Coil)"

    if (
        bb_w > bb_expansion or
        close >= bb_u or
        close <= bb_l
    ):
        return "High Volatility Expansion"

    return "Range-Bound"


def compute_regime_series(df: pd.DataFrame) -> pd.Series:
    regimes = []
    for _, row in df.iterrows():
        if any(pd.isna(row.get(c, np.nan)) for c in [
            "close", "rsi_14", "rv_10", "rv_20",
            "ma_20", "bb_upper", "bb_lower", "bb_width"
        ]):
            regimes.append("Unknown")
            continue

        close = row["close"]
        rsi = row["rsi_14"]
        slope20 = row["slope_20"]
        rv10 = row["rv_10"]
        rv20 = row["rv_20"]
        ma20 = row["ma_20"]
        bb_u = row["bb_upper"]
        bb_l = row["bb_lower"]
        bb_w = row["bb_width"]
        mid_band = (bb_u + bb_l) / 2

        slope_trend = 0.00020
        bb_compress = 0.02
        bb_expansion = 0.06

        if (
            slope20 > slope_trend and
            close > ma20 and
            rsi > 55 and
            close > mid_band
        ):
            regimes.append("Trending Up")
        elif (
            slope20 < -slope_trend and
            close < ma20 and
            rsi < 45 and
            close < mid_band
        ):
            regimes.append("Trending Down")
        elif (
            bb_w < bb_compress and
            rv10 < rv20 and
            40 < rsi < 60
        ):
            regimes.append("Volatility Compression (Coil)")
        elif (
            bb_w > bb_expansion or
            close >= bb_u or
            close <= bb_l
        ):
            regimes.append("High Volatility Expansion")
        else:
            regimes.append("Range-Bound")

    return pd.Series(regimes, index=df.index, name="regime")


# --------------------------------------------------------------------
# 3. Quant Summary
# --------------------------------------------------------------------
def summarize_quant(df: pd.DataFrame) -> str:
    df_q = df.dropna(subset=[
        "close", "rsi_14", "rv_10", "rv_20",
        "atr_14", "slope_5", "slope_20",
        "ma_20", "ma_50", "bb_upper", "bb_lower", "bb_width"
    ])
    if df_q.empty:
        return "Quant snapshot unavailable: insufficient data."

    last = df_q.iloc[-1]
    date_str = last.name.strftime("%Y-%m-%d")

    close = float(last["close"])
    rsi = float(last["rsi_14"])
    rv10 = float(last["rv_10"])
    rv20 = float(last["rv_20"])
    atr = float(last["atr_14"])
    slope5 = float(last["slope_5"])
    slope20 = float(last["slope_20"])
    ma20 = float(last["ma_20"])
    ma50 = float(last["ma_50"])
    bb_u = float(last["bb_upper"])
    bb_l = float(last["bb_lower"])
    bb_w = float(last["bb_width"])

    rel_ma20 = (close - ma20) / ma20 if ma20 != 0 else np.nan
    rel_ma50 = (close - ma50) / ma50 if ma50 != 0 else np.nan

    mid_band = (bb_u + bb_l) / 2
    if close >= bb_u:
        band_pos = "at/above the upper Bollinger band"
    elif close <= bb_l:
        band_pos = "at/below the lower Bollinger band"
    elif close > mid_band:
        band_pos = "in the upper half of the band"
    else:
        band_pos = "in the lower half of the band"

    regime = classify_regime(df)
    signal_score = compute_signal_score(last)

    text = f"""
    Quant snapshot as of {date_str}:

    - Spot close: **{close:.4f}**
    - 14d RSI: **{rsi:.1f}**
    - Realised vol (ann.): **RV10 = {rv10:.2%}**, **RV20 = {rv20:.2%}**
    - ATR(14): **{atr:.1f} pips**
    - Slopes: **5d = {slope5:.5f}**, **20d = {slope20:.5f}**

    Moving averages & bands:
    - 20d MA: **{ma20:.4f}**
    - 50d MA: **{ma50:.4f}**
    - Price vs MA20: **{rel_ma20:+.2%}**
    - Price vs MA50: **{rel_ma50:+.2%}**
    - Bollinger (20d, 2σ): lower **{bb_l:.4f}**, upper **{bb_u:.4f}**
    - Band width: **{bb_w:.2%}**  
    - Price is **{band_pos}**

    **Regime Classification:** {regime}

    **Signal Strength Score (0–100): {signal_score:.1f}**  
    - Higher values → more bullish  
    - Lower values → more bearish  
    - Around 50 → broadly neutral / range-bound
    """.strip()

    return text


# --------------------------------------------------------------------
# 4. Plotting with regime overlays
# --------------------------------------------------------------------
def plot_quant(df: pd.DataFrame, instrument: str, label: str) -> Path:
    """
    3-panel chart: Price+MAs with regime-colored overlays, RSI, RV10.
    """
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{instrument}_daily_quant.png"

    df_plot = df.dropna(subset=["close"])
    if df_plot.empty:
        return out_path

    regime_series = compute_regime_series(df_plot)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    regime_colors = {
        "Trending Up":      "#c7e9c0",
        "Trending Down":    "#fdd0a2",
        "Volatility Compression (Coil)": "#d9d9d9",
        "High Volatility Expansion": "#fcae91",
        "Range-Bound":      "#deebf7",
        "Unknown":          "#ffffff",
    }

    if not regime_series.empty:
        current_regime = regime_series.iloc[0]
        block_start = regime_series.index[0]

        for ts, reg in regime_series.iloc[1:].items():
            if reg != current_regime:
                color = regime_colors.get(current_regime, "#ffffff")
                for ax in axes:
                    ax.axvspan(block_start, ts, facecolor=color, alpha=0.15, linewidth=0)
                current_regime = reg
                block_start = ts

        last_ts = regime_series.index[-1]
        color = regime_colors.get(current_regime, "#ffffff")
        for ax in axes:
            ax.axvspan(block_start, last_ts, facecolor=color, alpha=0.15, linewidth=0)

    axes[0].plot(df_plot.index, df_plot["close"], label="Close", linewidth=1.2)
    if "ma_20" in df_plot.columns:
        axes[0].plot(df_plot.index, df_plot["ma_20"], "--", label="MA20", linewidth=1.0)
    if "ma_50" in df_plot.columns:
        axes[0].plot(df_plot.index, df_plot["ma_50"], ":", label="MA50", linewidth=1.0)
    if {"bb_upper", "bb_lower"}.issubset(df_plot.columns):
        axes[0].plot(df_plot.index, df_plot["bb_upper"], linewidth=0.8, alpha=0.7, label="BB Upper")
        axes[0].plot(df_plot.index, df_plot["bb_lower"], linewidth=0.8, alpha=0.7, label="BB Lower")
    axes[0].set_title(f"{label} – Daily Price, MAs & Regimes")
    axes[0].legend(loc="upper left", fontsize=8)

    if "rsi_14" in df_plot.columns:
        axes[1].plot(df_plot.index, df_plot["rsi_14"], linewidth=1.0)
    axes[1].axhline(70, ls="--", linewidth=0.8)
    axes[1].axhline(30, ls="--", linewidth=0.8)
    axes[1].set_ylabel("RSI(14)")

    if "rv_10" in df_plot.columns:
        axes[2].plot(df_plot.index, df_plot["rv_10"], linewidth=1.0)
    axes[2].set_ylabel("RV10 (ann.)")
    axes[2].set_xlabel("Date")

    legend_patches = [
        Patch(facecolor=color, alpha=0.4, label=label_)
        for label_, color in regime_colors.items()
        if label_ != "Unknown"
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved chart to {out_path}")
    return out_path


# --------------------------------------------------------------------
# 4b. Encode chart for embedding
# --------------------------------------------------------------------
def encode_chart_to_base64(path: Path) -> str:
    if not path.is_file():
        return ""
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("ascii")


# --------------------------------------------------------------------
# 5. Mini Backtest
# --------------------------------------------------------------------
def mini_backtest(df: pd.DataFrame, instrument: str, label: str) -> dict:
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    plot_path = out_dir / f"{instrument}_mini_backtest.png"

    cols_needed = ["ret", "slope_5", "rsi_14"]
    if not set(cols_needed).issubset(df.columns):
        return {
            "slope_only": {"sharpe": np.nan, "win_rate": np.nan, "max_drawdown": np.nan},
            "combined_slope_rsi": {"sharpe": np.nan, "win_rate": np.nan, "max_drawdown": np.nan},
            "plot_path": str(plot_path),
        }

    df_bt = df.dropna(subset=cols_needed).copy()
    if df_bt.empty:
        return {
            "slope_only": {"sharpe": np.nan, "win_rate": np.nan, "max_drawdown": np.nan},
            "combined_slope_rsi": {"sharpe": np.nan, "win_rate": np.nan, "max_drawdown": np.nan},
            "plot_path": str(plot_path),
        }

    df_bt["signal_slope"] = np.sign(df_bt["slope_5"])

    df_bt["signal_combined"] = 0
    long_cond = (df_bt["slope_5"] > 0) & (df_bt["rsi_14"] < 60)
    short_cond = (df_bt["slope_5"] < 0) & (df_bt["rsi_14"] > 40)
    df_bt.loc[long_cond, "signal_combined"] = 1
    df_bt.loc[short_cond, "signal_combined"] = -1

    for col in ["signal_slope", "signal_combined"]:
        df_bt[col] = df_bt[col].replace(0, np.nan)
        df_bt[col] = df_bt[col].ffill().fillna(0)

    df_bt["ret_slope"] = df_bt["signal_slope"].shift(1) * df_bt["ret"]
    df_bt["ret_combined"] = df_bt["signal_combined"].shift(1) * df_bt["ret"]

    df_bt = df_bt.dropna(subset=["ret_slope", "ret_combined"])

    def stats_from_series(r: pd.Series) -> dict:
        if r.std() == 0 or r.empty:
            sharpe = np.nan
        else:
            sharpe = (r.mean() / r.std()) * np.sqrt(252)

        win_rate = (r > 0).mean() if not r.empty else np.nan

        equity = (1 + r).cumprod()
        roll_max = equity.cummax()
        dd = equity / roll_max - 1
        max_dd = dd.min() if not dd.empty else np.nan

        return {
            "sharpe": round(float(sharpe), 3) if not np.isnan(sharpe) else np.nan,
            "win_rate": round(float(win_rate), 3) if not np.isnan(win_rate) else np.nan,
            "max_drawdown": round(float(max_dd), 3) if not np.isnan(max_dd) else np.nan,
        }

    stats_slope = stats_from_series(df_bt["ret_slope"])
    stats_combined = stats_from_series(df_bt["ret_combined"])

    eq_slope = (1 + df_bt["ret_slope"]).cumprod()
    eq_combined = (1 + df_bt["ret_combined"]).cumprod()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(eq_slope.index, eq_slope, label="Slope-only")
    ax.plot(eq_combined.index, eq_combined, label="Slope + RSI")
    ax.set_title(f"{label} – Mini Backtest (diagnostic)")
    ax.set_ylabel("Equity (gross)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    results = {
        "slope_only": stats_slope,
        "combined_slope_rsi": stats_combined,
        "plot_path": str(plot_path),
    }
    return results


# --------------------------------------------------------------------
# 6. Enhanced Tavily News
# --------------------------------------------------------------------
def get_tavily_news(query: str, max_results: int = 15):
    """
    Upgraded Tavily search:
    - Multi-query expansion
    - Domain filtering
    - Recency window
    - Deduplication of results
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key is None:
        raise ValueError("TAVILY_API_KEY missing.")
    client = TavilyClient(api_key=api_key)

    queries = [
        query,
        "macro outlook and risk sentiment",
        "Fed ECB policy impact FX and equities",
        "USD strength and global macro drivers",
        "inflation, yields, and risk sentiment across markets",
    ]

    include_domains = [
        "reuters.com",
        "bloomberg.com",
        "fxstreet.com",
        "investing.com",
        "forexlive.com",
        "marketwatch.com",
        "ft.com",
    ]

    all_results = []

    for q in queries:
        response = client.search(
            query=q,
            max_results=max_results,
            include_domains=include_domains,
            recency_days=7,
        )
        hits = response.get("results", [])
        all_results.extend(hits)

    deduped = {}
    for r in all_results:
        url = r.get("url")
        if url and url not in deduped:
            deduped[url] = r

    results_list = list(deduped.values())
    results_list = results_list[:max_results]

    return results_list


# --------------------------------------------------------------------
# 7. Prompt Builder
# --------------------------------------------------------------------
def build_research_prompt(
    label: str,
    instrument: str,
    df_quant: pd.DataFrame,
    quant_summary: str,
    news_results,
    chart_b64: str | None = None,
) -> str:

    cols = [c for c in ["open", "high", "low", "close"] if c in df_quant.columns]
    price_block = (
        df_quant[cols].tail(10).rename(columns=str.capitalize).to_string()
        if cols else "Price panel unavailable."
    )

    news_lines = []
    for i, r in enumerate(news_results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = (r.get("content", "") or "")[:300].replace("\n", " ")
        news_lines.append(f"{i}. {title}\n   {url}\n   Snippet: {snippet}")
    news_block = "\n\n".join(news_lines) if news_lines else "No recent news snippets available."

    if chart_b64:
        chart_block = f"""
4) A **Base64-encoded PNG chart** showing:
   - Top panel: daily price, moving averages, Bollinger bands, and regime-colored backgrounds.
   - Middle panel: RSI(14).
   - Bottom panel: 10-day annualised realized volatility.

<CHART_BASE64_PNG_START>
{chart_b64}
<CHART_BASE64_PNG_END>
"""
    else:
        chart_block = "4) No chart image was provided for this run."

    prompt = f"""
You are a hedge-fund style macro/FX strategist.

You are given data for **{label} ({instrument})**:

1) **Recent Oanda daily history** (OHLC, last 10 rows):
{price_block}

2) A **quant snapshot** including RSI, slopes, ATR, realized vols,
   moving averages, Bollinger bands, regime classification,
   and a composite **Signal Strength Score (0–100)**:
{quant_summary}

3) Recent **news & macro headlines**:
{news_block}

{chart_block}

TASK:
- Produce a structured **macro/strategy memo** in markdown.
- Include:
  - Market context & recent price action
  - Quant/technical interpretation, including explicit regime analysis
    and how the **Signal Strength Score** aligns (bullish / neutral / bearish)
  - Short-term (0–2 weeks) and medium-term (1–3 months) views
  - 2–3 trade ideas with entry zones, stops, targets, and risk-reward
  - Key risks & clear invalidation levels

Important:
- Stay grounded in the provided data (no hallucinated macro facts).
- If key information (rates, positioning, options vols) is missing,
  acknowledge that explicitly and keep conclusions appropriately conservative.
"""
    return dedent(prompt)


# --------------------------------------------------------------------
# 8. Per-asset runner
# --------------------------------------------------------------------
def run_for_asset(asset_key: str, lookback_days: int = 120):
    cfg = ASSETS[asset_key]
    instrument = cfg["instrument"]
    label = cfg["label"]
    tavily_query = cfg["tavily_query"]

    print("\n" + "=" * 80)
    print(f"{label}  |  Instrument: {instrument}")
    print("=" * 80 + "\n")

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=lookback_days)

    print(f"Fetching Oanda history for {instrument} from {start_date} to {end_date} ...")
    df_raw = get_oanda_history(
        instrument=instrument,
        start=str(start_date),
        end=str(end_date),
        granularity="D",
        price="M",
    )
    print("Oanda data shape:", df_raw.shape)

    df_q = add_quant_features(df_raw)

    chart_path = plot_quant(df_q, instrument, label)
    chart_b64 = encode_chart_to_base64(chart_path)

    backtest_results = mini_backtest(df_q, instrument, label)
    print("\nMini Backtest Results:")
    print(backtest_results)

    quant_summary = summarize_quant(df_q)

    print("\nFetching Tavily news ...")
    news_results = get_tavily_news(tavily_query, max_results=15)
    print(f"Tavily returned {len(news_results)} articles.\n")

    prompt = build_research_prompt(
        label=label,
        instrument=instrument,
        df_quant=df_q,
        quant_summary=quant_summary,
        news_results=news_results,
        chart_b64=chart_b64,
    )

    research_agent = Agent(
        name=f"{label} Oanda+Tavily Research Agent",
        model=OpenAIResponses(id="gpt-5.1"),
        instructions=dedent("""
            You are a hedge-fund macro/FX strategist.
            Produce a clean, well-structured strategy memo in markdown.
        """),
        markdown=True,
    )

    print("Running research agent...\n")
    response = research_agent.run(prompt)
    print(response.content)
    print("\n" + "-" * 80 + "\n")


# --------------------------------------------------------------------
# 9. Main: Hybrid CLI (single asset / all assets)
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Asset Oanda + Tavily Research Engine"
    )
    parser.add_argument(
        "--asset",
        type=str,
        choices=ASSETS.keys(),
        help=f"Asset key to run individually. Choices: {', '.join(ASSETS.keys())}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the engine for all configured assets.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=120,
        help="Lookback window in days for Oanda history (default: 120).",
    )

    args = parser.parse_args()

    if args.all:
        keys = list(ASSETS.keys())
    elif args.asset:
        keys = [args.asset]
    else:
        print("No --asset specified and --all not set. Defaulting to EURUSD.\n")
        keys = ["EURUSD"]

    for k in keys:
        run_for_asset(k, lookback_days=args.lookback)


if __name__ == "__main__":
    main()
