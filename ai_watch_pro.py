#!/usr/bin/env python3
"""
AI Watch Pro v3.2
-----------------
Long-term investor AI radar

Features:
- Analysis of key AI-related stocks (MSFT, GOOGL, AMZN, META, NVDA, AMD, ASML, AVGO)
- Comparison vs S&P 500
- Global and per-ticker news sentiment (NewsAPI + VADER NLP)
- Global AI score (raw + smoothed) + group scores
- Bubble risk estimate
- Allocation engine: weights per group & per ticker
- CSV history tracking with smoothing
- Visual flags (🟢🟡🔴) and arrows (▲▼▶)
- Automatic alerts (console + optional Telegram)
- Simple dashboard generation (Matplotlib PNG + HTML)

History file: ai_watch_history.csv
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ========= GENERAL CONFIG =========

BENCHMARK = "^GSPC"  # S&P 500
HORIZON_DAYS_1Y = 365
HORIZON_DAYS_3M = 90
HORIZON_DAYS_1M = 30

HISTORY_FILE = "ai_watch_history.csv"

# Alert thresholds
ALERT_SCORE_LOW = 55.0     # Global AI score < 55 -> stress
ALERT_SCORE_DROP = 15.0    # Drop > 15 pts vs last snapshot
ALERT_SCORE_HIGH = 85.0    # Global AI score > 85 -> possible euphoria

# Smoothing (Exponential Moving Average) for global score
SMOOTHING_ALPHA = 0.4      # 0.4 = 40% weight on today's raw score

# NewsAPI key (https://newsapi.org) – taken from env if set (for GitHub Actions)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_LANGUAGE = "en"
NEWS_PAGE_SIZE = 40

# Keywords for global AI sentiment query
GLOBAL_NEWS_QUERY = "artificial intelligence data center GPU investment"

# VADER sentiment analyzer (for news)
ANALYZER = SentimentIntensityAnalyzer()


# ========= AI UNIVERSE =========

@dataclass
class TickerInfo:
    symbol: str
    name: str
    role: str          # ex: "hyperscaler", "gpu", "semi", "consumer"
    weight: float = 1  # relative weight in group scores


AI_UNIVERSE: Dict[str, TickerInfo] = {
    "MSFT": TickerInfo("MSFT", "Microsoft", "hyperscaler", 1.2),
    "GOOGL": TickerInfo("GOOGL", "Alphabet/Google", "hyperscaler", 1.1),
    "AMZN": TickerInfo("AMZN", "Amazon", "hyperscaler", 1.1),
    "META": TickerInfo("META", "Meta Platforms", "consumer", 1.0),
    "NVDA": TickerInfo("NVDA", "NVIDIA", "gpu", 1.4),
    "AMD":  TickerInfo("AMD", "AMD", "gpu", 1.0),
    "ASML": TickerInfo("ASML", "ASML Holding", "semi", 1.2),
    "AVGO": TickerInfo("AVGO", "Broadcom", "semi", 1.0),
}


# ========= DATA CLASSES =========

@dataclass
class PerfWindow:
    perf_1y: float
    perf_3m: float
    perf_1m: float


@dataclass
class TickerSnapshot:
    info: TickerInfo
    perf: PerfWindow
    score_perf: float = 0.0
    score_news: Optional[float] = None
    score_total: float = 0.0


@dataclass
class GroupSnapshot:
    name: str
    tickers: List[TickerSnapshot] = field(default_factory=list)
    score: float = 0.0


@dataclass
class AIHealthSnapshot:
    as_of: dt.date
    benchmark_perf_1y: float
    avg_ai_1y: float
    ai_vs_bench_spread: float
    ticker_snaps: Dict[str, TickerSnapshot]
    groups: Dict[str, GroupSnapshot]
    news_global_ratio: Optional[float]
    score_global_raw: float        # one-shot score for today
    score_global: float            # smoothed score (EMA)
    status: str
    comment: str


# ========= VISUAL HELPERS =========

def flag_for_score(score: float) -> str:
    """Return a colored flag based on score level."""
    if score >= 75:
        return "🟢"
    if score >= 55:
        return "🟡"
    return "🔴"


def arrow_for_delta(delta: Optional[float]) -> str:
    """Arrow depending on the evolution."""
    if delta is None:
        return " "
    if delta > 3:
        return "▲"
    if delta < -3:
        return "▼"
    return "▶"


def format_delta(delta: Optional[float]) -> str:
    if delta is None:
        return ""
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}"


# ========= PRICE HELPERS =========

def _download_prices(tickers: List[str], days_back: int) -> Dict[str, pd.DataFrame]:
    """Download adjusted prices over `days_back` days for all tickers."""
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back + 5)
    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        group_by="ticker",
        auto_adjust=True,
    )
    out: Dict[str, pd.DataFrame] = {}

    # Single ticker case
    if isinstance(data, pd.DataFrame) and not isinstance(data.columns, pd.MultiIndex):
        out[tickers[0]] = data.dropna()
        return out

    # Multi-ticker case
    for t in tickers:
        if t in data:
            df = data[t].dropna()
            if not df.empty:
                out[t] = df

    return out


def _compute_period_return(
    df: pd.DataFrame,
    days_back: int,
    today: Optional[dt.date] = None
) -> float:
    """
    Simple return over `days_back` days, in %.
    If not enough data, returns 0.
    """
    if df.empty:
        return 0.0

    if today is None:
        today = dt.date.today()

    start_date = today - dt.timedelta(days=days_back)
    df_period = df[df.index.date >= start_date]

    if df_period.empty:
        return 0.0

    start_price = df_period["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]

    if start_price <= 0:
        return 0.0

    return (end_price / start_price - 1.0) * 100.0


# ========= NEWS / SENTIMENT (VADER) =========

def _fetch_news(query: str, api_key: str) -> List[Dict]:
    if not api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": NEWS_LANGUAGE,
        "pageSize": NEWS_PAGE_SIZE,
        "sortBy": "publishedAt",
        "apiKey": api_key,
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()

    data = resp.json()
    return data.get("articles", [])


def _score_text_sentiment(text: str) -> float:
    """
    Use VADER compound score in [-1, 1].
    """
    text = (text or "").strip()
    if not text:
        return 0.0
    scores = ANALYZER.polarity_scores(text)
    return scores["compound"]  # -1 (very negative) to +1 (very positive)


def _compute_news_ratio(articles: List[Dict]) -> Optional[float]:
    """
    Returns the fraction of 'positive' articles (0–1),
    or None if no usable articles.
    Criteria: VADER compound > 0.05 considered positive.
    """
    if not articles:
        return None

    scores: List[float] = []
    for art in articles:
        title = art.get("title") or ""
        description = art.get("description") or ""
        text = f"{title}. {description}".strip()
        s = _score_text_sentiment(text)
        scores.append(s)

    if not scores:
        return None

    positives = sum(1 for s in scores if s > 0.05)
    return positives / len(scores)


def compute_ticker_news_score(symbol: str) -> Optional[float]:
    """Returns a 0–100 sentiment score based on news articles for a ticker."""
    if not NEWS_API_KEY:
        return None

    query = f"{symbol} artificial intelligence data center GPU"
    try:
        articles = _fetch_news(query, NEWS_API_KEY)
        print(f"[DEBUG] {symbol} news: {len(articles)} articles fetched")
    except Exception as e:
        print(f"[WARN] NewsAPI error for {symbol}: {e}")
        return None

    ratio = _compute_news_ratio(articles)
    if ratio is None:
        return None

    return ratio * 100.0


def compute_global_news_ratio() -> Optional[float]:
    """Returns global AI news sentiment ratio (0–1), or None if unavailable."""
    if not NEWS_API_KEY:
        return None

    try:
        articles = _fetch_news(GLOBAL_NEWS_QUERY, NEWS_API_KEY)
        print(f"[DEBUG] Global AI news: {len(articles)} articles fetched")
    except Exception as e:
        print(f"[WARN] NewsAPI error (global): {e}")
        return None

    return _compute_news_ratio(articles)


# ========= BUBBLE RISK DETECTOR =========

def detect_bubble_risk(snap: AIHealthSnapshot) -> str:
    """
    Very simple bubble risk heuristic:
    - high group scores
    - strong spread vs benchmark
    - very positive news sentiment
    Returns: "Very Low", "Low", "Moderate", "High".
    """
    avg_group_score = statistics.mean(g.score for g in snap.groups.values())
    news_score = (snap.news_global_ratio * 100.0) if snap.news_global_ratio is not None else 60.0
    spread = snap.ai_vs_bench_spread

    risk_points = 0
    if avg_group_score > 80:
        risk_points += 1
    if spread > 40:
        risk_points += 1
    if news_score > 70:
        risk_points += 1

    if risk_points >= 3:
        return "High"
    if risk_points == 2:
        return "Moderate"
    if risk_points == 1:
        return "Low"
    return "Very Low"


# ========= SNAPSHOT CONSTRUCTION =========

def build_ai_snapshot() -> AIHealthSnapshot:
    today = dt.date.today()
    tickers = list(AI_UNIVERSE.keys()) + [BENCHMARK]

    prices = _download_prices(tickers, HORIZON_DAYS_1Y)
    if BENCHMARK not in prices:
        raise RuntimeError(f"No price data for benchmark {BENCHMARK}")

    bench_df = prices[BENCHMARK]
    benchmark_perf_1y = _compute_period_return(bench_df, HORIZON_DAYS_1Y, today)

    ticker_snaps: Dict[str, TickerSnapshot] = {}
    perfs_1y: List[float] = []

    # Per-ticker performance
    for symbol, info in AI_UNIVERSE.items():
        df = prices.get(symbol)
        if df is None:
            continue

        p1y = _compute_period_return(df, HORIZON_DAYS_1Y, today)
        p3m = _compute_period_return(df, HORIZON_DAYS_3M, today)
        p1m = _compute_period_return(df, HORIZON_DAYS_1M, today)

        perf = PerfWindow(p1y, p3m, p1m)
        snap = TickerSnapshot(info=info, perf=perf)
        ticker_snaps[symbol] = snap
        perfs_1y.append(p1y)

    if not perfs_1y:
        raise RuntimeError("Unable to compute AI performance: no data available.")

    avg_ai_1y = statistics.mean(perfs_1y)
    ai_vs_bench_spread = avg_ai_1y - benchmark_perf_1y

    # Performance score (based on spread vs benchmark)
    for snap in ticker_snaps.values():
        spread = snap.perf.perf_1y - benchmark_perf_1y
        # Map spread in [-30, +30] to [0, 100]
        perf_score = max(0.0, min(100.0, (spread + 30) * (100.0 / 60.0)))
        snap.score_perf = perf_score

    # News score + total score
    for symbol, snap in ticker_snaps.items():
        news_score = compute_ticker_news_score(symbol)
        snap.score_news = news_score

        if news_score is None:
            snap.score_total = snap.score_perf
        else:
            snap.score_total = 0.6 * snap.score_perf + 0.4 * news_score

    # Group aggregation
    groups: Dict[str, GroupSnapshot] = {}
    for snap in ticker_snaps.values():
        gname = snap.info.role
        if gname not in groups:
            groups[gname] = GroupSnapshot(name=gname, tickers=[])
        groups[gname].tickers.append(snap)

    for grp in groups.values():
        if not grp.tickers:
            grp.score = 0.0
        else:
            total_w = sum(t.info.weight for t in grp.tickers)
            if total_w <= 0:
                grp.score = statistics.mean(t.score_total for t in grp.tickers)
            else:
                grp.score = sum(
                    t.score_total * t.info.weight for t in grp.tickers
                ) / total_w

    global_news_ratio = compute_global_news_ratio()

    # Global AI score (raw)
    group_scores = [g.score for g in groups.values()] or [0.0]
    avg_group_score = statistics.mean(group_scores)
    spread_score = max(0.0, min(100.0, (ai_vs_bench_spread + 30) * (100.0 / 60.0)))

    if global_news_ratio is None:
        news_global_score = 60.0  # neutral default
    else:
        news_global_score = global_news_ratio * 100.0

    score_global_raw = (
        0.5 * avg_group_score +
        0.3 * spread_score +
        0.2 * news_global_score
    )

    # Regime status and commentary based on raw score for today
    if score_global_raw >= 80:
        status = "Strong / Bullish AI Cycle"
        comment = (
            "AI leaders are strongly outperforming the market. "
            "Infrastructure, GPUs, and semiconductors remain key growth drivers."
        )
    elif score_global_raw >= 60:
        status = "Positive / Moderate AI Cycle"
        comment = (
            "Overall AI signals remain positive, although some segments are digesting gains "
            "or showing mixed momentum."
        )
    elif score_global_raw >= 50:
        status = "AI Plateau / Normalization"
        comment = (
            "AI performance is broadly in line with the market, with a more neutral environment."
        )
    else:
        status = "AI Stress / Risk Zone"
        comment = (
            "AI leaders are no longer outperforming, some segments are under pressure, "
            "and news flow is turning more negative."
        )

    score_global_raw_rounded = round(score_global_raw, 1)

    return AIHealthSnapshot(
        as_of=today,
        benchmark_perf_1y=benchmark_perf_1y,
        avg_ai_1y=avg_ai_1y,
        ai_vs_bench_spread=ai_vs_bench_spread,
        ticker_snaps=ticker_snaps,
        groups=groups,
        news_global_ratio=global_news_ratio,
        score_global_raw=score_global_raw_rounded,
        score_global=score_global_raw_rounded,  # will be smoothed later
        status=status,
        comment=comment,
    )


# ========= HISTORY =========

def load_last_history_row() -> Optional[Dict[str, str]]:
    if not os.path.exists(HISTORY_FILE):
        return None

    last_row = None
    with open(HISTORY_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row

    return last_row


def save_snapshot_to_history(snap: AIHealthSnapshot) -> None:
    fieldnames = [
        "date",
        "score_global_raw",
        "score_global",          # smoothed
        "benchmark_perf_1y",
        "avg_ai_1y",
        "ai_vs_bench_spread",
    ]
    for gname in sorted(snap.groups.keys()):
        fieldnames.append(f"group_{gname}_score")

    file_exists = os.path.exists(HISTORY_FILE)

    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        row = {
            "date": snap.as_of.isoformat(),
            "score_global_raw": f"{snap.score_global_raw:.2f}",
            "score_global": f"{snap.score_global:.2f}",
            "benchmark_perf_1y": f"{snap.benchmark_perf_1y:.2f}",
            "avg_ai_1y": f"{snap.avg_ai_1y:.2f}",
            "ai_vs_bench_spread": f"{snap.ai_vs_bench_spread:.2f}",
        }

        for gname, grp in snap.groups.items():
            row[f"group_{gname}_score"] = f"{grp.score:.2f}"

        writer.writerow({k: row.get(k, "") for k in fieldnames})


# ========= ALERTS, MACRO, ALLOCATION =========

def generate_alerts(
    snap: AIHealthSnapshot,
    last_row: Optional[Dict[str, str]],
    bubble_risk: str
) -> List[str]:
    alerts: List[str] = []

    if snap.score_global < ALERT_SCORE_LOW:
        alerts.append(
            f"ALERT: Global AI score is low ({snap.score_global:.1f} < {ALERT_SCORE_LOW}). "
            "The AI market appears under pressure."
        )

    if snap.score_global > ALERT_SCORE_HIGH:
        alerts.append(
            f"NOTE: Global AI score is very high ({snap.score_global:.1f} > {ALERT_SCORE_HIGH}). "
            "Potential overheating / euphoria phase."
        )

    if bubble_risk in ("Moderate", "High"):
        alerts.append(
            f"NOTE: Bubble risk estimated as {bubble_risk}. Consider avoiding aggressive leverage "
            "or speculative short-term bets."
        )

    if last_row is not None and "score_global" in last_row:
        try:
            last_score = float(last_row["score_global"])
            delta = snap.score_global - last_score

            if delta <= -ALERT_SCORE_DROP:
                alerts.append(
                    f"ALERT: Rapid deterioration in AI score ({delta:.1f} pts vs last run). "
                    "Watch for a potential break or shock in AI-related assets."
                )
            elif delta >= ALERT_SCORE_DROP:
                alerts.append(
                    f"NOTE: Rapid improvement in AI score (+{delta:.1f} pts vs last run). "
                    "Significant positive shift in sentiment or performance."
                )
        except ValueError:
            pass

    return alerts


def compute_macro_reco(snap: AIHealthSnapshot, delta_global: Optional[float]) -> str:
    """
    Macro recommendation: do nothing / caution / strong cycle.
    Designed for long-term investors, not short-term trading.
    Uses the smoothed global score.
    """
    sg = snap.score_global

    if sg >= 80:
        if delta_global is not None and delta_global > 5:
            return (
                "Strong and accelerating AI cycle: continue your DCA strategy, "
                "do not increase risk further, accept volatility."
            )
        return (
            "Strong but stable AI cycle: keep your strategy unchanged, "
            "avoid emotional overexposure or chasing recent winners."
        )

    if sg >= 60:
        if delta_global is not None and delta_global < -5:
            return (
                "Positive but slowing AI cycle: remain cautious, "
                "avoid exceptional capital additions and monitor for a regime change."
            )
        return (
            "Moderately bullish AI cycle: do nothing special, "
            "let your long-term automatic plan run."
        )

    if sg >= 50:
        return (
            "AI consolidation / plateau: stay invested but be more selective, "
            "reserve major new capital for clearer opportunities."
        )

    return (
        "AI stress zone: only add gradually, if at all. "
        "Stay focused on long-term fundamentals and avoid emotional decisions."
    )


def compute_allocation_weights(snap: AIHealthSnapshot) -> Dict[str, Dict[str, float]]:
    """
    Compute simple allocation weights:
    - group weights based on group scores (clipped at 0)
    - ticker weights within each group based on score_total * weight
    Returns dict with "groups" and "tickers" percentage weights (sum to 100 each).
    """
    group_scores = {name: max(0.0, grp.score) for name, grp in snap.groups.items()}
    total_group_score = sum(group_scores.values()) or 1.0

    group_weights = {name: (score / total_group_score) * 100.0 for name, score in group_scores.items()}

    ticker_weights: Dict[str, float] = {}
    for name, grp in snap.groups.items():
        gw = group_weights[name] / 100.0  # fraction of portfolio in this group
        scores = [
            max(0.0, t.score_total * t.info.weight)
            for t in grp.tickers
        ]
        total = sum(scores) or 1.0
        for t, s in zip(grp.tickers, scores):
            ticker_weights[t.info.symbol] = gw * (s / total) * 100.0

    return {"groups": group_weights, "tickers": ticker_weights}


# ========= TELEGRAM ALERTS =========

def send_telegram_alert(message: str) -> None:
    """
    Send alert message via Telegram bot if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
    are present in environment variables. Silent fail otherwise.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url,
            data={"chat_id": chat_id, "text": message},
            timeout=10,
        )
        resp.raise_for_status()
        print("[INFO] Telegram alert sent.")
    except Exception as e:
        print(f"[WARN] Failed to send Telegram alert: {e}")


# ========= DASHBOARD (MATPLOTLIB) =========

def generate_dashboard(
    history_file: str,
    output_dir: str = "dashboard",
    snapshot: Optional[AIHealthSnapshot] = None,
    allocation_weights: Optional[Dict[str, Dict[str, float]]] = None,
    bubble_risk: Optional[str] = None,
) -> None:
    """
    Generate an investor-friendly dashboard:
    - Global score (raw + smoothed) over time
    - Group scores over time
    - Latest snapshot summary (regime, score, delta)
    - Latest group scores table
    - Allocation engine (groups + tickers) tables
    - Interpretation & actions
    """
    if not os.path.exists(history_file):
        print("[INFO] No history file yet, skipping dashboard generation.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Robust CSV loading
    df = pd.read_csv(
        history_file,
        on_bad_lines="skip",
        engine="python",
    )
    if df.empty:
        print("[INFO] Empty history file, skipping dashboard generation.")
        return

    df["date"] = pd.to_datetime(df["date"])

    # ---------- TIME-SERIES CHARTS ----------

    # Global scores (raw vs smoothed)
    plt.figure(figsize=(7, 4))
    plt.plot(df["date"], df["score_global_raw"], label="Raw score")
    plt.plot(df["date"], df["score_global"], label="Smoothed score")
    plt.axhline(50, linestyle="--", linewidth=0.8)
    plt.axhline(65, linestyle="--", linewidth=0.8)
    plt.axhline(80, linestyle="--", linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel("Global AI score")
    plt.title("Global AI Score (Raw vs Smoothed)")
    plt.legend()
    plt.tight_layout()
    global_png = os.path.join(output_dir, "global_scores.png")
    plt.savefig(global_png)
    plt.close()

    # Group scores
    group_cols = [c for c in df.columns if c.startswith("group_")]
    groups_png = None
    if group_cols:
        plt.figure(figsize=(7, 4))
        for col in group_cols:
            label = col.replace("group_", "")
            plt.plot(df["date"], df[col], label=label)
        plt.xlabel("Date")
        plt.ylabel("Group score")
        plt.title("AI Group Scores Over Time")
        plt.legend()
        plt.tight_layout()
        groups_png = os.path.join(output_dir, "group_scores.png")
        plt.savefig(groups_png)
        plt.close()

    # ---------- LATEST SNAPSHOT & REGIME ----------

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    latest_date = latest["date"].date()
    latest_score = float(latest["score_global"])
    latest_raw = float(latest["score_global_raw"])

    delta_val: Optional[float] = None
    delta_str = "N/A"
    if prev is not None:
        try:
            delta_val = latest_score - float(prev["score_global"])
            sign = "+" if delta_val >= 0 else ""
            delta_str = f"{sign}{delta_val:.1f} pts vs previous"
        except Exception:
            delta_val = None

    def regime_from_score(s: float) -> str:
        if s >= 80:
            return "Strong / Bullish AI Cycle"
        if s >= 60:
            return "Positive / Moderate AI Cycle"
        if s >= 50:
            return "AI Plateau / Normalization"
        return "AI Stress / Risk Zone"

    regime = snapshot.status if snapshot is not None else regime_from_score(latest_raw)
    bubble = bubble_risk or "N/A"

    # ---------- LATEST GROUP SCORES ----------

    latest_groups: List[Tuple[str, float]] = []
    for col in group_cols:
        gname = col.replace("group_", "")
        try:
            val = float(latest[col])
        except Exception:
            val = float("nan")
        latest_groups.append((gname, val))

    latest_groups_sorted = sorted(latest_groups, key=lambda x: -x[1]) if latest_groups else []

    # ---------- ALLOCATION DATA ----------

    group_alloc_rows: List[Tuple[str, float]] = []
    ticker_alloc_rows: List[Tuple[str, str, float]] = []

    if allocation_weights is not None:
        for gname, w in sorted(allocation_weights["groups"].items(), key=lambda x: -x[1]):
            group_alloc_rows.append((gname, w))

        for symbol, w in sorted(allocation_weights["tickers"].items(), key=lambda x: -x[1]):
            name = AI_UNIVERSE[symbol].name
            ticker_alloc_rows.append((symbol, name, w))

    # ---------- INTERPRETATION & ACTIONS ----------

    # Top / bottom groups
    top_group = latest_groups_sorted[0][0] if latest_groups_sorted else "N/A"
    bottom_group = latest_groups_sorted[-1][0] if len(latest_groups_sorted) >= 1 else "N/A"

    actions: List[str] = []

    if latest_score >= 80:
        actions.append(
            "Strong AI regime: keep your long-term allocation, avoid adding large new risk driven by euphoria."
        )
    elif latest_score >= 60:
        actions.append(
            "Moderately bullish AI regime: let your automatic long-term plan run, no need to time the market."
        )
    elif latest_score >= 50:
        actions.append(
            "Neutral / plateau regime: be more selective, reserve large new capital for clearer opportunities."
        )
    else:
        actions.append(
            "AI stress regime: only add gradually, if at all. Focus on quality names and long-term horizon."
        )

    if top_group != "N/A":
        actions.append(f"Overweight high-scoring AI groups (e.g. {top_group}).")
    if bottom_group != "N/A":
        actions.append(f"Underweight or avoid adding to weaker groups (e.g. {bottom_group}).")

    if bubble == "High":
        actions.append(
            "Bubble risk HIGH: avoid leverage, trim speculative positions, and prioritize balance sheet strength."
        )
    elif bubble == "Moderate":
        actions.append(
            "Bubble risk MODERATE: avoid aggressive new capital, focus on core holdings rather than fringe plays."
        )
    elif bubble == "Low":
        actions.append(
            "Bubble risk LOW: current optimism is supported by fundamentals and sentiment; no obvious excess."
        )

    # ---------- HTML LAYOUT (2-COLUMN) ----------

    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><title>AI Watch Dashboard</title>\n")
        f.write(
            "<style>"
            "body{font-family:Arial,Helvetica,sans-serif;margin:20px;}"
            "h1{margin-bottom:0.2rem;}"
            ".small{font-size:0.85rem;color:#6b7280;}"
            ".section{margin-bottom:2rem;}"
            ".tag{display:inline-block;padding:4px 8px;border-radius:6px;font-size:0.85rem;margin-left:4px;}"
            ".tag-green{background:#d1fae5;color:#065f46;}"
            ".tag-yellow{background:#fef9c3;color:#854d0e;}"
            ".tag-red{background:#fee2e2;color:#b91c1c;}"
            "table{border-collapse:collapse;margin-top:0.5rem;}"
            "th,td{border:1px solid #ddd;padding:6px 10px;font-size:0.9rem;}"
            "th{background:#f3f4f6;text-align:left;}"
            ".grid{display:grid;grid-template-columns:2fr 1.5fr;grid-gap:24px;align-items:flex-start;}"
            ".grid-full{display:grid;grid-template-columns:1.4fr 1.6fr;grid-gap:24px;align-items:flex-start;}"
            "ul{margin-top:0.4rem;}"
            "@media(max-width:900px){.grid,.grid-full{grid-template-columns:1fr;}}"
            "</style>\n"
        )
        f.write("</head><body>\n")

        # HEADER
        f.write("<h1>AI Watch Dashboard</h1>\n")
        f.write(f"<p class='small'>Last update: {latest_date}</p>\n")

        # SUMMARY BLOCK
        f.write("<div class='section'>\n")
        f.write("<h2>Market Regime & Summary</h2>\n")

        if latest_score >= 80:
            tag_cls = "tag tag-green"
            tag_txt = "Bullish"
        elif latest_score >= 60:
            tag_cls = "tag tag-yellow"
            tag_txt = "Moderate"
        elif latest_score >= 50:
            tag_cls = "tag tag-yellow"
            tag_txt = "Plateau"
        else:
            tag_cls = "tag tag-red"
            tag_txt = "Stress"

        f.write(
            f"<p><strong>Global AI Score (smoothed):</strong> {latest_score:.1f} / 100 "
            f"<span class='{tag_cls}'>{tag_txt}</span></p>\n"
        )
        f.write(f"<p><strong>Raw AI Score (today):</strong> {latest_raw:.1f} / 100</p>\n")
        if delta_val is not None:
            f.write(f"<p><strong>Change vs previous:</strong> {delta_str}</p>\n")
        f.write(f"<p><strong>Regime:</strong> {regime}</p>\n")
        f.write(f"<p><strong>Bubble risk (heuristic):</strong> {bubble}</p>\n")
        f.write("</div>\n")

        # GRID 1: Global score chart (left) + Actions (right)
        f.write("<div class='section grid'>\n")

        # LEFT: Global AI score chart
        f.write("<div>\n")
        f.write("<h2>Global AI Score</h2>\n")
        f.write("<p class='small'>Raw vs smoothed score over time. Dashed lines mark key regime thresholds (50 / 65 / 80).</p>\n")
        f.write('<img src="global_scores.png" alt="Global AI scores">\n')
        f.write("</div>\n")

        # RIGHT: Interpretation & Actions
        f.write("<div>\n")
        f.write("<h2>Interpretation & Actions</h2>\n")
        f.write("<ul>\n")
        for a in actions:
            f.write(f"<li>{a}</li>\n")
        f.write("</ul>\n")
        f.write("</div>\n")

        f.write("</div>\n")  # end grid

        # GRID 2: Group chart (left) + tables / allocation (right)
        f.write("<div class='section grid-full'>\n")

        # LEFT: group scores chart
        f.write("<div>\n")
        f.write("<h2>AI Group Scores Over Time</h2>\n")
        if groups_png:
            f.write("<p class='small'>Evolution of scores by AI segment (hyperscalers, GPU, semiconductors, consumer).</p>\n")
            f.write('<img src="group_scores.png" alt="AI group scores over time">\n')
        else:
            f.write("<p class='small'>No group history available yet.</p>\n")
        f.write("</div>\n")

        # RIGHT: latest group scores + allocation engine
        f.write("<div>\n")
        f.write("<h2>Latest Group Scores</h2>\n")
        if latest_groups_sorted:
            f.write("<table>\n")
            f.write("<tr><th>Group</th><th>Latest score</th></tr>\n")
            for gname, val in latest_groups_sorted:
                f.write(f"<tr><td>{gname}</td><td>{val:.1f}</td></tr>\n")
            f.write("</table>\n")
        else:
            f.write("<p class='small'>No group scores available.</p>\n")

        # Allocation engine tables
        f.write("<h2>AI Allocation Engine</h2>\n")
        if group_alloc_rows:
            f.write("<h3>By group</h3>\n")
            f.write("<table>\n")
            f.write("<tr><th>Group</th><th>Suggested weight (of AI sleeve)</th></tr>\n")
            for gname, w in group_alloc_rows:
                f.write(f"<tr><td>{gname}</td><td>{w:.1f}%</td></tr>\n")
            f.write("</table>\n")

        if ticker_alloc_rows:
            f.write("<h3>By ticker</h3>\n")
            f.write("<table>\n")
            f.write("<tr><th>Ticker</th><th>Name</th><th>Suggested weight (of AI sleeve)</th></tr>\n")
            for symbol, name, w in ticker_alloc_rows:
                f.write(f"<tr><td>{symbol}</td><td>{name}</td><td>{w:.1f}%</td></tr>\n")
            f.write("</table>\n")

        if not group_alloc_rows and not ticker_alloc_rows:
            f.write("<p class='small'>No allocation data available for this run.</p>\n")

        f.write("</div>\n")  # right side

        f.write("</div>\n")  # end grid-full

        f.write("</body></html>\n")

    print(f"[INFO] Dashboard generated in: {output_dir}")

# ========= DISPLAY =========

def print_ai_snapshot(
    snap: AIHealthSnapshot,
    last_row: Optional[Dict[str, str]],
    delta_global: Optional[float],
    group_deltas: Dict[str, Optional[float]],
    bubble_risk: str,
    allocation_weights: Dict[str, Dict[str, float]],
) -> None:
    flag = flag_for_score(snap.score_global)
    arr = arrow_for_delta(delta_global)
    delta_str = format_delta(delta_global)

    print(f"\n=== AI RADAR — {snap.as_of} ===\n")
    print(f"Global AI Score (smoothed)   : {snap.score_global:5.1f} / 100 {flag} {arr} {delta_str}")
    print(f"Raw AI Score (today only)    : {snap.score_global_raw:5.1f} / 100")
    print(f"Benchmark 1Y ({BENCHMARK})   : {snap.benchmark_perf_1y:6.2f}%")
    print(f"Avg AI Performance (1Y)      : {snap.avg_ai_1y:6.2f}%")
    print(f"AI vs Benchmark Spread (1Y)  : {snap.ai_vs_bench_spread:6.2f} pts")

    if snap.news_global_ratio is not None:
        print(
            f"Global AI news sentiment    : {snap.news_global_ratio*100:5.1f}% "
            f"positive articles (approx.)"
        )
    else:
        if NEWS_API_KEY:
            print("Global AI news sentiment    : unavailable (API error or no data)")
        else:
            print("Global AI news sentiment    : unavailable (API key missing)")

    print(f"Bubble risk estimate         : {bubble_risk}")
    print(f"Market Regime (raw score)    : {snap.status}")
    print(f"Commentary                   : {snap.comment}\n")

    print("=== AI GROUPS ===")
    for gname, grp in snap.groups.items():
        g_flag = flag_for_score(grp.score)
        d = group_deltas.get(gname)
        g_arr = arrow_for_delta(d)
        g_delta = format_delta(d)
        tickers_str = ", ".join(t.info.symbol for t in grp.tickers)
        print(
            f"- {gname:11s} : {grp.score:5.1f} / 100 {g_flag} {g_arr} {g_delta} "
            f"({tickers_str})"
        )
    print()

    print("=== PER TICKER DETAILS ===")
    for symbol, ts in snap.ticker_snaps.items():
        p = ts.perf
        news_str = f"{ts.score_news:5.1f}" if ts.score_news is not None else "  NA "
        t_flag = flag_for_score(ts.score_total)
        print(
            f"{symbol:5s} ({ts.info.name:15s}) "
            f"| 1Y: {p.perf_1y:6.2f}%  3M: {p.perf_3m:6.2f}%  "
            f"1M: {p.perf_1m:6.2f}%  | TotalScore: {ts.score_total:6.1f} {t_flag}  "
            f"(Perf: {ts.score_perf:5.1f}, News: {news_str})"
        )
    print()

    print("=== AI ALLOCATION ENGINE (GROUPS) ===")
    for gname, w in sorted(allocation_weights["groups"].items(), key=lambda x: -x[1]):
        print(f"- {gname:11s}: {w:5.1f}% of AI sleeve")
    print()

    print("=== AI ALLOCATION ENGINE (TICKERS) ===")
    for symbol, w in sorted(allocation_weights["tickers"].items(), key=lambda x: -x[1]):
        name = AI_UNIVERSE[symbol].name
        print(f"- {symbol:5s} ({name:15s}): {w:5.1f}% of AI sleeve")
    print()


# ========= MAIN =========

def main() -> None:
    last_row = load_last_history_row()
    snap = build_ai_snapshot()

    # Apply smoothing to global score (EMA) based on last history row
    if last_row is not None and "score_global" in last_row:
        try:
            last_smooth = float(last_row["score_global"])
            smooth = (
                SMOOTHING_ALPHA * snap.score_global_raw
                + (1.0 - SMOOTHING_ALPHA) * last_smooth
            )
            snap.score_global = round(smooth, 1)
        except ValueError:
            snap.score_global = snap.score_global_raw
    else:
        snap.score_global = snap.score_global_raw

    # Deltas vs previous run (using smoothed scores)
    delta_global: Optional[float] = None
    group_deltas: Dict[str, Optional[float]] = {g: None for g in snap.groups.keys()}

    if last_row is not None and "score_global" in last_row:
        try:
            last_score = float(last_row["score_global"])
            delta_global = snap.score_global - last_score
        except ValueError:
            delta_global = None

        for gname, grp in snap.groups.items():
            key = f"group_{gname}_score"
            if key in last_row:
                try:
                    last_g = float(last_row[key])
                    group_deltas[gname] = grp.score - last_g
                except ValueError:
                    group_deltas[gname] = None

    # Bubble risk, allocation, recommendation, alerts
    bubble_risk = detect_bubble_risk(snap)
    allocation_weights = compute_allocation_weights(snap)

    # Print core snapshot + allocation
    print_ai_snapshot(snap, last_row, delta_global, group_deltas, bubble_risk, allocation_weights)

    # Macro recommendation
    reco = compute_macro_reco(snap, delta_global)
    print("=== LONG-TERM INVESTOR MACRO RECOMMENDATION ===")
    print(reco)
    print()

    # Alerts
    alerts = generate_alerts(snap, last_row, bubble_risk)
    if alerts:
        print("=== ALERTS / NOTES ===")
        for a in alerts:
            print("- " + a)
        print()

        # Send Telegram alert (short summary)
        alert_text = (
            f"AI Watch Alert — {snap.as_of}\n"
            f"Score: {snap.score_global:.1f} (raw {snap.score_global_raw:.1f})\n"
            f"Bubble risk: {bubble_risk}\n"
            + "\n".join(f"- {a}" for a in alerts)
        )
        send_telegram_alert(alert_text)

    # Persist history
    save_snapshot_to_history(snap)
    print(f"History updated in: {HISTORY_FILE}\n")

    # Dashboard generation (controlled via env var for GitHub Actions)
    if os.getenv("GENERATE_DASHBOARD", "0") == "1":
        generate_dashboard(
            HISTORY_FILE,
            output_dir="dashboard",
            snapshot=snap,
            allocation_weights=allocation_weights,
            bubble_risk=bubble_risk,
        )


if __name__ == "__main__":
    main()
