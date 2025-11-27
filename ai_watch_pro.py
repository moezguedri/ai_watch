#!/usr/bin/env python3
"""
AI Watch Pro
------------
Long-term "AI radar" for a small universe of AI-related leaders.

Features:
- Universe: MSFT, GOOGL, AMZN, META, NVDA, AMD, ASML, AVGO
- Benchmark: S&P 500 (^GSPC)
- 1Y / 3M / 1M performance windows
- Per-ticker performance score vs benchmark (0–100)
- Per-ticker news sentiment score using VADER + NewsAPI
- Per-group AI scores (hyperscaler / consumer / gpu / semi)
- Global AI score (raw + smoothed EMA)
- Simple bubble-risk heuristic
- Allocation engine for groups & tickers
- CSV history (ai_watch_history.csv)
- Static HTML dashboard (dashboard/index.html) for GitHub Pages
- Optional Telegram alerts

Environment variables (recommended):
- NEWS_API_KEY            : NewsAPI key
- TELEGRAM_BOT_TOKEN      : Telegram bot token (optional)
- TELEGRAM_CHAT_ID        : Telegram chat id (optional)
- GENERATE_DASHBOARD=1    : to enable dashboard generation
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# yfinance configuration
# ---------------------------------------------------------------------------
# Older yfinance versions don't support .set_config() with these arguments.
# To stay compatible, we do NOT touch its internal cache settings here.
# Robustness comes from:
#   - threads=False in yf.download()
#   - multiple retries in _download_prices()
#
# This avoids the "database is locked" issue in most CI cases without
# relying on yfinance's newer config API.

# ---------------------------------------------------------------------------
# GENERAL CONFIG
# ---------------------------------------------------------------------------

BENCHMARK = "^GSPC"  # S&P 500

HORIZON_DAYS_1Y = 365
HORIZON_DAYS_3M = 90
HORIZON_DAYS_1M = 30

HISTORY_FILE = "ai_watch_history.csv"

# Score alert thresholds
ALERT_SCORE_LOW = 55.0     # global score < 55 -> stress
ALERT_SCORE_DROP = 15.0    # drop > 15 pts vs last snapshot
ALERT_SCORE_HIGH = 85.0    # global score > 85 -> euphoria

# News API
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
NEWS_LANGUAGE = "en"
NEWS_PAGE_SIZE = 40

GLOBAL_NEWS_QUERY = "artificial intelligence data center GPU investment"

# VADER analyzer
SENT_ANALYZER = SentimentIntensityAnalyzer()

# ---------------------------------------------------------------------------
# AI UNIVERSE
# ---------------------------------------------------------------------------


@dataclass
class TickerInfo:
    symbol: str
    name: str
    role: str          # "hyperscaler", "gpu", "semi", "consumer"
    weight: float = 1  # relative weight inside its group


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

# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------


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
    score_global_raw: float
    score_global: float
    status: str
    comment: str
    bubble_risk: str


# ---------------------------------------------------------------------------
# METHODOLOGY HTML (OPTIONAL IN DASHBOARD)
# ---------------------------------------------------------------------------

METHOD_EXPLANATION_HTML = """
<h2>Methodology: AI Watch Engine</h2>
<p>
This dashboard summarizes a rules-based process designed for long-term investors
who want a simple, repeatable way to monitor the state of the AI equity cycle.
It is not a trading system and it does not attempt to time short-term moves.
</p>

<ol>
  <li><strong>Universe & Benchmark</strong>
    <ul>
      <li><em>Universe:</em> a small set of liquid, large-cap AI-related leaders:
        MSFT, GOOGL, AMZN, META (hyperscalers / consumer), NVDA, AMD (GPU),
        ASML, AVGO (semiconductors).</li>
      <li><em>Benchmark:</em> the S&amp;P 500 (^GSPC) is used as the broad equity
        reference.</li>
    </ul>
  </li>

  <li><strong>Price data & performance windows</strong>
    <ul>
      <li>Daily adjusted prices are downloaded with yfinance.</li>
      <li>For each stock and for the benchmark the script computes:
        <ul>
          <li>1-year performance (1Y)</li>
          <li>3-month performance (3M)</li>
          <li>1-month performance (1M)</li>
        </ul>
      </li>
    </ul>
  </li>

  <li><strong>Performance scoring vs benchmark</strong>
    <ul>
      <li>For each stock, its 1Y performance is compared to the 1Y benchmark performance.</li>
      <li>The spread = stock_1Y_return − benchmark_1Y_return.</li>
      <li>This spread is mapped from roughly [-30%, +30%] to a 0–100 performance score:
        values below -30% are floored near 0, values above +30% are capped near 100.</li>
      <li>This produces a normalized performance score that says:
        “how much this AI name has out- or underperformed the broad market over 1 year.”</li>
    </ul>
  </li>

  <li><strong>News sentiment using VADER + NewsAPI</strong>
    <ul>
      <li>For each ticker, the script queries NewsAPI with an AI-related query
        (ticker + terms like "artificial intelligence", "data center", "GPU").</li>
      <li>Titles + descriptions of recent articles are fed into the
        VADER sentiment analyzer (compound score in [-1, 1]).</li>
      <li>An article is considered positive if compound &gt; 0.05.</li>
      <li>The ticker’s news sentiment score is the fraction of positive articles,
        scaled to 0–100.</li>
      <li>A similar global query is run for “AI / data center / GPU investment”,
        giving a global AI news sentiment ratio.</li>
      <li>If the API fails or returns nothing, sentiment falls back to a neutral value.</li>
    </ul>
  </li>

  <li><strong>Per-ticker total score</strong>
    <ul>
      <li>Each stock gets:
        <ul>
          <li>a performance score (vs benchmark), and</li>
          <li>a news sentiment score (0–100, or None if unavailable).</li>
        </ul>
      </li>
      <li>If news sentiment is available, total_score = 0.6 × perf_score + 0.4 × news_score.</li>
      <li>If news sentiment is missing, total_score = perf_score.</li>
    </ul>
  </li>

  <li><strong>Groups & group scores</strong>
    <ul>
      <li>Each stock belongs to an AI segment: hyperscaler, consumer, gpu, semi.</li>
      <li>Within each group, total scores are aggregated using simple weights
        (e.g. NVDA has a slightly higher weight inside the GPU group).</li>
      <li>Result: a group-level score (0–100) for each AI segment.</li>
    </ul>
  </li>

  <li><strong>Global AI score (raw) & smoothing</strong>
    <ul>
      <li>Several ingredients are combined into a single raw global AI score:
        <ul>
          <li>average group score (core of the signal)</li>
          <li>spread of average AI 1Y performance vs benchmark</li>
          <li>global AI news sentiment score</li>
        </ul>
      </li>
      <li>The raw global score is then smoothed over time using an
        Exponential Moving Average (EMA) with a configurable alpha.</li>
      <li>The smoothed score is what you see as “Global AI Score (smoothed)”.</li>
      <li>The raw score is the “snapshot of today”, more volatile and sensitive to new data.</li>
    </ul>
  </li>

  <li><strong>Bubble risk heuristic</strong>
    <ul>
      <li>The script uses a simple 3-point heuristic based on:
        <ul>
          <li>average group score</li>
          <li>AI vs benchmark performance spread</li>
          <li>global news sentiment level</li>
        </ul>
      </li>
      <li>If all three are elevated at the same time, bubble risk is flagged as <em>High</em>.</li>
      <li>Two out of three elevated → <em>Moderate</em> risk.</li>
      <li>One out of three → <em>Low</em> risk.</li>
      <li>None elevated → <em>Very Low</em> risk.</li>
      <li>This is intentionally coarse and should be read as “risk of overheating”, not a precise timing signal.</li>
    </ul>
  </li>

  <li><strong>Allocation engine (groups & tickers)</strong>
    <ul>
      <li>The allocation engine is meant to suggest how to spread the “AI sleeve”
        of a portfolio, not the whole portfolio.</li>
      <li>First, each group gets a base weight proportional to its score, with
        mild compression so that strong groups do not become unrealistically dominant.</li>
      <li>Then, inside each group, tickers are weighted by a combination of
        total_score and their internal weight (e.g. NVDA slightly more than AMD).</li>
      <li>The result is:
        <ul>
          <li>Suggested % by group (of the AI sleeve)</li>
          <li>Suggested % by ticker (of the AI sleeve)</li>
        </ul>
      </li>
      <li>The allocation engine does <strong>not</strong> tell you to constantly add to winners.
        Instead, it treats high-scoring groups as core exposures to maintain, and lower-scoring or
        deteriorating groups as segments where you should avoid adding aggressively.</li>
    </ul>
  </li>

  <li><strong>Regime & interpretation logic</strong>
    <ul>
      <li>The global smoothed score defines the regime:
        <ul>
          <li>&gt;= 80 → strong / bullish AI cycle</li>
          <li>60–79 → positive / moderate AI cycle</li>
          <li>50–59 → plateau / normalization</li>
          <li>&lt; 50 → stress / risk zone</li>
        </ul>
      </li>
      <li>The dashboard then combines:
        <ul>
          <li>score level (where we are in the cycle)</li>
          <li>change vs previous run (are we accelerating, stabilizing, or deteriorating?)</li>
          <li>group score rankings (which segments are leading or lagging)</li>
          <li>bubble risk estimation</li>
        </ul>
      </li>
      <li>From these it generates a qualitative interpretation such as:
        “moderately bullish AI regime: keep your long-term plan running, do not try to time the top”,
        and group-level guidance (“leaders to maintain”, “laggards to avoid adding to for now”).</li>
    </ul>
  </li>

  <li><strong>Alerts & long-term focus</strong>
    <ul>
      <li>Alerts are triggered if:
        <ul>
          <li>the global score drops below a stress threshold,</li>
          <li>the global score exceeds a high euphoria threshold,</li>
          <li>the score moves very quickly (large positive or negative change vs last run).</li>
        </ul>
      </li>
      <li>The entire system is intentionally slow-moving:
        it is designed for long-term investors doing periodic allocation checks,
        not intraday trading or leveraged speculation.</li>
      <li>All outputs are <strong>descriptive and educational</strong>, not financial advice.</li>
    </ul>
  </li>
</ol>
"""

# ---------------------------------------------------------------------------
# VISUAL UTILITIES
# ---------------------------------------------------------------------------


def flag_for_score(score: float) -> str:
    if score >= 75:
        return "🟢"
    if score >= 55:
        return "🟡"
    return "🔴"


def arrow_for_delta(delta: Optional[float]) -> str:
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


# ---------------------------------------------------------------------------
# PRICE UTILITIES (with yfinance lock protection)
# ---------------------------------------------------------------------------


def _download_prices(tickers: List[str], days_back: int) -> Dict[str, pd.DataFrame]:
    """
    Download adjusted prices for tickers over the last `days_back` days.
    Uses retries and threads=False to avoid yfinance locking issues.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back + 5)

    data = None
    for attempt in range(3):
        try:
            data = yf.download(
                tickers,
                start=start,
                end=end,
                progress=False,
                group_by="ticker",
                auto_adjust=True,
                threads=False,  # important for CI stability
            )
            if data is not None and not data.empty:
                break
        except Exception as e:
            print(f"[WARN] yfinance download attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    if data is None or data.empty:
        raise RuntimeError(f"Failed to download price data for tickers: {tickers}")

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
    today: Optional[dt.date] = None,
) -> float:
    """Simple % return over days_back window."""
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


# ---------------------------------------------------------------------------
# NEWS / SENTIMENT (VADER + NewsAPI)
# ---------------------------------------------------------------------------


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
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("articles", [])


def _compute_news_score_from_articles(articles: List[Dict]) -> Optional[float]:
    if not articles:
        return None
    positives = 0
    total = 0
    for art in articles:
        title = art.get("title") or ""
        desc = art.get("description") or ""
        text = f"{title} {desc}"
        if not text.strip():
            continue
        scores = SENT_ANALYZER.polarity_scores(text)
        compound = scores.get("compound", 0.0)
        if compound > 0.05:
            positives += 1
        total += 1

    if total == 0:
        return None
    return (positives / total) * 100.0


def compute_ticker_news_score(symbol: str) -> Optional[float]:
    if not NEWS_API_KEY:
        return None
    query = f"{symbol} artificial intelligence data center GPU"
    try:
        arts = _fetch_news(query, NEWS_API_KEY)
    except Exception as e:
        print(f"[WARN] News fetch failed for {symbol}: {e}")
        return None
    return _compute_news_score_from_articles(arts)


def compute_global_news_ratio() -> Optional[float]:
    if not NEWS_API_KEY:
        return None
    try:
        arts = _fetch_news(GLOBAL_NEWS_QUERY, NEWS_API_KEY)
    except Exception as e:
        print(f"[WARN] Global news fetch failed: {e}")
        return None
    score = _compute_news_score_from_articles(arts)
    if score is None:
        return None
    return score / 100.0


# ---------------------------------------------------------------------------
# BUBBLE RISK HEURISTIC
# ---------------------------------------------------------------------------


def estimate_bubble_risk(
    avg_group_score: float,
    ai_vs_bench_spread: float,
    news_ratio: Optional[float],
) -> str:
    flags = 0
    if avg_group_score >= 80:
        flags += 1
    if ai_vs_bench_spread >= 25:
        flags += 1
    if news_ratio is not None and news_ratio >= 0.7:
        flags += 1

    if flags >= 3:
        return "High"
    if flags == 2:
        return "Moderate"
    if flags == 1:
        return "Low"
    return "Very Low"


# ---------------------------------------------------------------------------
# SNAPSHOT BUILDING
# ---------------------------------------------------------------------------


def build_ai_snapshot(last_row: Optional[Dict[str, str]] = None) -> AIHealthSnapshot:
    today = dt.date.today()
    tickers = list(AI_UNIVERSE.keys()) + [BENCHMARK]

    prices = _download_prices(tickers, HORIZON_DAYS_1Y)
    if BENCHMARK not in prices:
        raise RuntimeError(f"No price data for benchmark {BENCHMARK}")

    bench_df = prices[BENCHMARK]
    benchmark_perf_1y = _compute_period_return(bench_df, HORIZON_DAYS_1Y, today)

    ticker_snaps: Dict[str, TickerSnapshot] = {}
    perfs_1y: List[float] = []

    for symbol, info in AI_UNIVERSE.items():
        df = prices.get(symbol)
        if df is None or df.empty:
            continue
        p1y = _compute_period_return(df, HORIZON_DAYS_1Y, today)
        p3m = _compute_period_return(df, HORIZON_DAYS_3M, today)
        p1m = _compute_period_return(df, HORIZON_DAYS_1M, today)
        perf = PerfWindow(p1y, p3m, p1m)
        snap = TickerSnapshot(info=info, perf=perf)
        ticker_snaps[symbol] = snap
        perfs_1y.append(p1y)

    if not perfs_1y:
        raise RuntimeError("No AI performance computed.")

    avg_ai_1y = statistics.mean(perfs_1y)
    ai_vs_bench_spread = avg_ai_1y - benchmark_perf_1y

    # Performance score vs benchmark
    for snap in ticker_snaps.values():
        spread = snap.perf.perf_1y - benchmark_perf_1y
        perf_score = (spread + 30.0) * (100.0 / 60.0)
        perf_score = max(0.0, min(100.0, perf_score))
        snap.score_perf = perf_score

    # News + total score
    for symbol, snap in ticker_snaps.items():
        news_score = compute_ticker_news_score(symbol)
        snap.score_news = news_score
        if news_score is None:
            snap.score_total = snap.score_perf
        else:
            snap.score_total = 0.6 * snap.score_perf + 0.4 * news_score

    # Groups
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

    # Global score raw
    group_scores = [g.score for g in groups.values()] or [0.0]
    avg_group_score = statistics.mean(group_scores)
    spread_score = (ai_vs_bench_spread + 30.0) * (100.0 / 60.0)
    spread_score = max(0.0, min(100.0, spread_score))
    if global_news_ratio is None:
        news_global_score = 60.0
    else:
        news_global_score = global_news_ratio * 100.0

    score_global_raw = (
        0.5 * avg_group_score +
        0.3 * spread_score +
        0.2 * news_global_score
    )

    # Smoothed score (EMA on previous smoothed value)
    last_smoothed = None
    if last_row is not None and "score_global" in last_row:
        try:
            last_smoothed = float(last_row["score_global"])
        except Exception:
            last_smoothed = None

    if last_smoothed is None:
        score_global = score_global_raw
    else:
        alpha = 0.3
        score_global = alpha * score_global_raw + (1.0 - alpha) * last_smoothed

    score_global_raw = round(score_global_raw, 1)
    score_global = round(score_global, 1)

    # Regime text
    if score_global_raw >= 80:
        status = "Strong / Bullish AI Cycle"
        comment = (
            "AI leaders are strongly outperforming the market. Infrastructure, GPUs and "
            "semiconductors remain key growth drivers."
        )
    elif score_global_raw >= 60:
        status = "Positive / Moderate AI Cycle"
        comment = (
            "Overall AI signals are constructive, although some segments are digesting gains "
            "or showing more mixed momentum."
        )
    elif score_global_raw >= 50:
        status = "AI Plateau / Normalization"
        comment = (
            "The AI complex trades more in line with the broader market; the explosive phase "
            "of outperformance may be cooling."
        )
    else:
        status = "AI Stress / Risk Zone"
        comment = (
            "AI leaders are no longer clearly outperforming, some segments are under pressure, "
            "and news flow is more challenging."
        )

    bubble_risk = estimate_bubble_risk(avg_group_score, ai_vs_bench_spread, global_news_ratio)

    return AIHealthSnapshot(
        as_of=today,
        benchmark_perf_1y=benchmark_perf_1y,
        avg_ai_1y=avg_ai_1y,
        ai_vs_bench_spread=ai_vs_bench_spread,
        ticker_snaps=ticker_snaps,
        groups=groups,
        news_global_ratio=global_news_ratio,
        score_global_raw=score_global_raw,
        score_global=score_global,
        status=status,
        comment=comment,
        bubble_risk=bubble_risk,
    )


# ---------------------------------------------------------------------------
# HISTORY & ALERTS
# ---------------------------------------------------------------------------


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
        "score_global",
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

        # ensure all columns are present
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def generate_alerts(
    snap: AIHealthSnapshot,
    last_row: Optional[Dict[str, str]],
) -> List[str]:
    alerts: List[str] = []

    if snap.score_global < ALERT_SCORE_LOW:
        alerts.append(
            f"ALERT: Global AI score is low ({snap.score_global:.1f} < {ALERT_SCORE_LOW}). "
            "The AI market seems under pressure."
        )

    if snap.score_global > ALERT_SCORE_HIGH:
        alerts.append(
            f"NOTE: Global AI score is very high ({snap.score_global:.1f} > {ALERT_SCORE_HIGH}). "
            "Potential overheating / euphoria phase."
        )

    if last_row is not None and "score_global" in last_row:
        try:
            last_score = float(last_row["score_global"])
            delta = snap.score_global - last_score
            if delta <= -ALERT_SCORE_DROP:
                alerts.append(
                    f"ALERT: Rapid deterioration in AI score ({delta:.1f} pts vs last run). "
                    "Watch for a potential regime shift or shock."
                )
            elif delta >= ALERT_SCORE_DROP:
                alerts.append(
                    f"NOTE: Rapid improvement in AI score (+{delta:.1f} pts vs last run). "
                    "Strong positive shift in sentiment or fundamentals."
                )
        except Exception:
            pass

    return alerts


def compute_macro_reco(
    snap: AIHealthSnapshot,
    delta_global: Optional[float],
) -> str:
    sg = snap.score_global

    if sg >= 80:
        if delta_global is not None and delta_global > 5:
            return (
                "AI cycle is strong and accelerating: keep your DCA running, "
                "do not increase risk aggressively, and accept volatility as part of the process."
            )
        return (
            "AI cycle is strong but more stable: keep your long-term strategy unchanged, "
            "avoid 'over-investing' out of euphoria."
        )

    if sg >= 60:
        if delta_global is not None and delta_global < -5:
            return (
                "AI cycle is still positive but slowing: be cautious with new capital, "
                "keep your plan but avoid exceptional additions."
            )
        return (
            "Moderately bullish AI cycle: do nothing special, "
            "let your long-term automatic plan run."
        )

    if sg >= 50:
        return (
            "AI plateau: reinforce prudence, stay long-term oriented, and keep dry powder "
            "for clearer opportunities or stress periods."
        )

    return (
        "AI stress zone: if you add at all, do it slowly and focus on quality names. "
        "Avoid emotional decisions and respect your risk budget."
    )


# ---------------------------------------------------------------------------
# ALLOCATION ENGINE
# ---------------------------------------------------------------------------


def compute_ai_allocation(snap: AIHealthSnapshot) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      {
        "groups": { group_name: weight_percent_of_AI_sleeve },
        "tickers": { symbol: weight_percent_of_AI_sleeve },
      }
    """
    # Group weights (mild compression to avoid concentration)
    raw_group_weights: Dict[str, float] = {}
    for gname, grp in snap.groups.items():
        score = grp.score
        # Normalize score relative to 50-90 range
        x = max(0.0, score - 50.0) / 40.0  # 0 at 50, 1 at 90
        x = max(0.2, min(1.0, x))  # keep a minimum weight
        raw_group_weights[gname] = x

    total = sum(raw_group_weights.values())
    if total <= 0:
        group_alloc = {g: 100.0 / len(raw_group_weights) for g in raw_group_weights}
    else:
        group_alloc = {g: 100.0 * w / total for g, w in raw_group_weights.items()}

    # Ticker weights
    ticker_alloc: Dict[str, float] = {}
    for gname, grp in snap.groups.items():
        g_weight = group_alloc.get(gname, 0.0)
        if not grp.tickers or g_weight <= 0:
            continue
        raw_ticker_w: Dict[str, float] = {}
        for ts in grp.tickers:
            base = max(0.0, ts.score_total) / 100.0
            base *= max(0.2, ts.info.weight)
            raw_ticker_w[ts.info.symbol] = base
        subtotal = sum(raw_ticker_w.values())
        if subtotal <= 0:
            share_each = g_weight / len(raw_ticker_w)
            for sym in raw_ticker_w.keys():
                ticker_alloc[sym] = share_each
        else:
            for sym, w in raw_ticker_w.items():
                ticker_alloc[sym] = g_weight * (w / subtotal)

    return {"groups": group_alloc, "tickers": ticker_alloc}


# ---------------------------------------------------------------------------
# TELEGRAM ALERTS
# ---------------------------------------------------------------------------


def send_telegram_alert(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
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


# ---------------------------------------------------------------------------
# DASHBOARD GENERATION (HTML + PNG)
# ---------------------------------------------------------------------------


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

    df = pd.read_csv(
        history_file,
        on_bad_lines="skip",
        engine="python",
    )
    if df.empty:
        print("[INFO] Empty history file, skipping dashboard generation.")
        return

    df["date"] = pd.to_datetime(df["date"])

    # --- Charts ---

    # Global scores
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

    # --- Latest snapshot & dynamics ---

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
    bubble = bubble_risk or (snapshot.bubble_risk if snapshot else "N/A")

    # Latest group scores & deltas
    latest_groups: List[Tuple[str, float]] = []
    group_deltas: Dict[str, float] = {}

    for col in group_cols:
        gname = col.replace("group_", "")
        try:
            val = float(latest[col])
        except Exception:
            val = float("nan")
        latest_groups.append((gname, val))

        if prev is not None:
            try:
                prev_val = float(prev[col])
                group_deltas[gname] = val - prev_val
            except Exception:
                group_deltas[gname] = 0.0
        else:
            group_deltas[gname] = 0.0

    latest_groups_sorted = sorted(latest_groups, key=lambda x: -x[1]) if latest_groups else []

    # Allocation data
    group_alloc_rows: List[Tuple[str, float]] = []
    ticker_alloc_rows: List[Tuple[str, str, float]] = []

    if allocation_weights is not None:
        for gname, w in sorted(allocation_weights["groups"].items(), key=lambda x: -x[1]):
            group_alloc_rows.append((gname, w))
        for symbol, w in sorted(allocation_weights["tickers"].items(), key=lambda x: -x[1]):
            name = AI_UNIVERSE[symbol].name
            ticker_alloc_rows.append((symbol, name, w))

    # --- Group classification for text guidance ---

    def classify_group(score: float, delta: float) -> str:
        if score >= 80 and delta > 4:
            return "hot"
        if score >= 75:
            return "core_maintain"
        if 60 <= score < 75 and delta > 2:
            return "improving"
        if score < 55 and delta <= 0:
            return "weak"
        return "neutral"

    group_states: Dict[str, str] = {}
    for gname, val in latest_groups:
        d = group_deltas.get(gname, 0.0)
        group_states[gname] = classify_group(val, d)

    # --- Interpretation & actions ---

    actions: List[str] = []

    if latest_score >= 80:
        actions.append(
            "AI regime is strong and mature: keep your core allocation, avoid chasing short-term strength "
            "or adding aggressive new risk."
        )
    elif latest_score >= 60:
        actions.append(
            "Moderately bullish AI regime: let your automatic long-term plan run and avoid overreacting "
            "to short-term noise."
        )
    elif latest_score >= 50:
        actions.append(
            "Plateau / normalization regime: stay invested but be more selective with new capital."
        )
    else:
        actions.append(
            "Stress regime: if you add at all, do it gradually and focus on quality names, not speculative plays."
        )

    core_groups = [g for g, s in group_states.items() if s == "core_maintain"]
    improving_groups = [g for g, s in group_states.items() if s == "improving"]
    weak_groups = [g for g, s in group_states.items() if s == "weak"]
    hot_groups = [g for g, s in group_states.items() if s == "hot"]

    if core_groups:
        actions.append(
            "Treat high-scoring groups as core holdings to maintain (e.g. "
            + ", ".join(core_groups)
            + "), rather than segments to aggressively increase at any price."
        )

    if improving_groups:
        actions.append(
            "If you decide to increase AI exposure, prioritize groups that are improving from mid levels (e.g. "
            + ", ".join(improving_groups)
            + ") instead of chasing already extended leaders."
        )

    if weak_groups:
        actions.append(
            "Avoid adding to structurally weak or deteriorating groups for now (e.g. "
            + ", ".join(weak_groups)
            + "); wait for stabilization or genuine improvement."
        )

    if hot_groups:
        actions.append(
            "Very strong and accelerating groups (e.g. "
            + ", ".join(hot_groups)
            + ") should be monitored for potential overheating; consider holding rather than adding aggressively."
        )

    if bubble == "High":
        actions.append(
            "Bubble risk is HIGH: avoid leverage, reduce speculative positions, and favor balance-sheet strength "
            "and diversification."
        )
    elif bubble == "Moderate":
        actions.append(
            "Bubble risk is MODERATE: avoid aggressive new capital, focus on core holdings rather than fringe or "
            "highly speculative names."
        )
    elif bubble == "Low":
        actions.append(
            "Bubble risk is LOW: current optimism appears broadly supported by fundamentals and sentiment, "
            "but discipline is still required."
        )

    # --- HTML layout (two-column) ---

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

        # Header
        f.write("<h1>AI Watch Dashboard</h1>\n")
        f.write(f"<p class='small'>Last update: {latest_date}</p>\n")

        # Summary
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

        # Grid 1: global chart + actions
        f.write("<div class='section grid'>\n")

        f.write("<div>\n")
        f.write("<h2>Global AI Score</h2>\n")
        f.write("<p class='small'>Raw vs smoothed score over time. Dashed lines mark key regime thresholds (50 / 65 / 80).</p>\n")
        f.write('<img src="global_scores.png" alt="Global AI scores">\n')
        f.write("</div>\n")

        f.write("<div>\n")
        f.write("<h2>Interpretation & Actions</h2>\n")
        f.write("<ul>\n")
        for a in actions:
            f.write(f"<li>{a}</li>\n")
        f.write("</ul>\n")
        f.write("</div>\n")

        f.write("</div>\n")  # end first grid

        # Grid 2: group chart + tables/alloc
        f.write("<div class='section grid-full'>\n")

        f.write("<div>\n")
        f.write("<h2>AI Group Scores Over Time</h2>\n")
        if groups_png:
            f.write("<p class='small'>Evolution of scores by AI segment (hyperscalers, GPU, semiconductors, consumer).</p>\n")
            f.write('<img src="group_scores.png" alt="AI group scores over time">\n')
        else:
            f.write("<p class='small'>No group history available yet.</p>\n")
        f.write("</div>\n")

        f.write("<div>\n")
        f.write("<h2>Latest Group Scores</h2>\n")
        if latest_groups_sorted:
            f.write("<table>\n")
            f.write("<tr><th>Group</th><th>Latest score</th><th>1-step change</th></tr>\n")
            for gname, val in latest_groups_sorted:
                d = group_deltas.get(gname, 0.0)
                sign = "+" if d >= 0 else ""
                f.write(f"<tr><td>{gname}</td><td>{val:.1f}</td><td>{sign}{d:.1f}</td></tr>\n")
            f.write("</table>\n")
        else:
            f.write("<p class='small'>No group scores available.</p>\n")

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
        f.write("</div>\n")  # end second grid

        # Methodology section (comment out if you don't want it)
        f.write("<div class='section'>\n")
        f.write(METHOD_EXPLANATION_HTML)
        f.write("</div>\n")

        f.write("</body></html>\n")

    print(f"[INFO] Dashboard generated in: {output_dir}")


# ---------------------------------------------------------------------------
# PRINT SNAPSHOT (CLI)
# ---------------------------------------------------------------------------


def print_ai_snapshot(
    snap: AIHealthSnapshot,
    delta_global: Optional[float],
    group_deltas: Dict[str, Optional[float]],
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
            f"Global AI news sentiment    : {snap.news_global_ratio*100:5.1f}% positive articles (approx.)"
        )
    else:
        print("Global AI news sentiment    : not evaluated (NEWS_API_KEY missing or API error)")
    print(f"Bubble risk estimate         : {snap.bubble_risk}")
    print(f"Market Regime (raw score)    : {snap.status}")
    print(f"Commentary                   : {snap.comment}\n")

    print("=== AI GROUPS ===")
    for gname, grp in snap.groups.items():
        g_flag = flag_for_score(grp.score)
        d = group_deltas.get(gname)
        g_arr = arrow_for_delta(d)
        g_delta = format_delta(d)
        symbols = ", ".join(t.info.symbol for t in grp.tickers)
        print(
            f"- {gname:11s} : {grp.score:5.1f} / 100 {g_flag} {g_arr} {g_delta} "
            f"({symbols})"
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

    # Allocation engine printout
    groups_alloc = allocation_weights.get("groups", {})
    tickers_alloc = allocation_weights.get("tickers", {})

    if groups_alloc:
        print("=== AI ALLOCATION ENGINE (GROUPS) ===")
        for gname, w in sorted(groups_alloc.items(), key=lambda x: -x[1]):
            print(f"- {gname:11s}: {w:5.1f}% of AI sleeve")
        print()

    if tickers_alloc:
        print("=== AI ALLOCATION ENGINE (TICKERS) ===")
        for sym, w in sorted(tickers_alloc.items(), key=lambda x: -x[1]):
            info = AI_UNIVERSE[sym]
            print(f"- {sym:4s} ({info.name:14s}): {w:5.1f}% of AI sleeve")
        print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    last_row = load_last_history_row()
    snap = build_ai_snapshot(last_row)

    # Deltas vs last run
    delta_global: Optional[float] = None
    group_deltas: Dict[str, Optional[float]] = {g: None for g in snap.groups.keys()}

    if last_row is not None and "score_global" in last_row:
        try:
            last_score = float(last_row["score_global"])
            delta_global = snap.score_global - last_score
        except Exception:
            delta_global = None

        for gname, grp in snap.groups.items():
            key = f"group_{gname}_score"
            if key in last_row:
                try:
                    last_g = float(last_row[key])
                    group_deltas[gname] = grp.score - last_g
                except Exception:
                    group_deltas[gname] = None

    allocation_weights = compute_ai_allocation(snap)

    print_ai_snapshot(snap, delta_global, group_deltas, allocation_weights)

    # Macro recommendation
    reco = compute_macro_reco(snap, delta_global)
    print("=== LONG-TERM INVESTOR MACRO RECOMMENDATION ===")
    print(reco)
    print()

    # Alerts
    alerts = generate_alerts(snap, last_row)
    if alerts:
        print("=== ALERTS / NOTES ===")
        for a in alerts:
            print("- " + a)
        print()

        # Optional Telegram alerts
        send_telegram_alert("\n".join(alerts))

    # Save history
    save_snapshot_to_history(snap)
    print(f"History updated in: {HISTORY_FILE}\n")

    # Dashboard generation (for GitHub Pages)
    if os.getenv("GENERATE_DASHBOARD", "0") == "1":
        generate_dashboard(
            HISTORY_FILE,
            output_dir="dashboard",
            snapshot=snap,
            allocation_weights=allocation_weights,
            bubble_risk=snap.bubble_risk,
        )


if __name__ == "__main__":
    main()
