#!/usr/bin/env python3
"""
AI Watch Pro v3
---------------
Radar IA "investisseur long terme"

Fonctionnalités :
- Analyse des géants IA (MSFT, GOOGL, AMZN, META, NVDA, AMD, ASML, AVGO)
- Comparaison vs S&P 500
- Sentiment news global + par ticker (NewsAPI si clé fournie)
- Score global IA + scores par groupe
- Historique CSV
- Flèches ▲▼▶, drapeaux 🟢🟡🔴
- Recommandation macro : "ne rien faire", "prudence", "cycle fort"

Historique : ai_watch_history.csv
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf


# ========= CONFIG GÉNÉRALE =========

BENCHMARK = "^GSPC"  # S&P 500
HORIZON_DAYS_1Y = 365
HORIZON_DAYS_3M = 90
HORIZON_DAYS_1M = 30

HISTORY_FILE = "ai_watch_history.csv"

# Seuils d’alertes
ALERT_SCORE_LOW = 55.0     # score global IA < 55 -> stress
ALERT_SCORE_DROP = 15.0    # baisse > 15 pts vs dernier snapshot
ALERT_SCORE_HIGH = 85.0    # score global IA > 85 -> euphorie possible

# Clé NewsAPI (https://newsapi.org)
NEWS_API_KEY = "bb163b5b0a6844e4b5b66c1d649986e9"  # <-- À REMPLACER pour activer les news
NEWS_LANGUAGE = "en"
NEWS_PAGE_SIZE = 40

# Mots-clés pour le sentiment global IA
GLOBAL_NEWS_QUERY = "artificial intelligence data center GPU investment"

# Mots-clés sentiment
POSITIVE_KEYWORDS = [
    "growth", "record", "surge", "booming", "expansion",
    "strong", "profit", "beat", "outperform", "demand"
]
NEGATIVE_KEYWORDS = [
    "regulation", "ban", "slowdown", "collapse", "bubble",
    "risk", "loss", "probe", "fine", "investigation"
]


# ========= UNIVERS IA =========

@dataclass
class TickerInfo:
    symbol: str
    name: str
    role: str          # ex: "hyperscaler", "gpu", "semi", "consumer"
    weight: float = 1  # poids relatif dans les scores


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
    score_global: float
    status: str
    comment: str


# ========= UTILITAIRES VISUELS =========

def flag_for_score(score: float) -> str:
    """Renvoie un drapeau couleur selon le niveau de score."""
    if score >= 75:
        return "🟢"
    if score >= 55:
        return "🟡"
    return "🔴"


def arrow_for_delta(delta: Optional[float]) -> str:
    """Flèche en fonction de l’évolution."""
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


# ========= UTILITAIRES PRIX =========

def _download_prices(tickers: List[str], days_back: int) -> Dict[str, pd.DataFrame]:
    """Télécharge les prix ajustés sur `days_back` jours pour tous les tickers."""
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

    if isinstance(data, pd.DataFrame) and not isinstance(data.columns, pd.MultiIndex):
        out[tickers[0]] = data.dropna()
        return out

    for t in tickers:
        if t in data:
            df = data[t].dropna()
            if not df.empty:
                out[t] = df

    return out


def _compute_period_return(df: pd.DataFrame, days_back: int, today: Optional[dt.date] = None) -> float:
    """Perf simple sur `days_back` jours, en %. Si pas assez de données, renvoie 0."""
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


# ========= NEWS / SENTIMENT =========

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


def _score_text_sentiment(text: str) -> int:
    text = (text or "").lower()
    score = 0
    if any(k in text for k in POSITIVE_KEYWORDS):
        score += 1
    if any(k in text for k in NEGATIVE_KEYWORDS):
        score -= 1
    return score


def _compute_news_ratio(articles: List[Dict]) -> Optional[float]:
    if not articles:
        return None
    scores = []
    for art in articles:
        title = art.get("title") or ""
        desc = art.get("description") or ""
        text = f"{title} {desc}"
        s = _score_text_sentiment(text)
        scores.append(s)
    if not scores:
        return None
    positives = sum(1 for s in scores if s > 0)
    return positives / len(scores)


def compute_ticker_news_score(symbol: str) -> Optional[float]:
    """Renvoie un score 0–100 basé sur les news autour du ticker."""
    if not NEWS_API_KEY:
        return None
    query = f"{symbol} artificial intelligence data center GPU"
    try:
        arts = _fetch_news(query, NEWS_API_KEY)
    except Exception:
        return None
    ratio = _compute_news_ratio(arts)
    if ratio is None:
        return None
    return ratio * 100.0


def compute_global_news_ratio() -> Optional[float]:
    if not NEWS_API_KEY:
        return None
    try:
        arts = _fetch_news(GLOBAL_NEWS_QUERY, NEWS_API_KEY)
    except Exception:
        return None
    return _compute_news_ratio(arts)


# ========= CONSTRUCTION DU SNAPSHOT =========

def build_ai_snapshot() -> AIHealthSnapshot:
    today = dt.date.today()
    tickers = list(AI_UNIVERSE.keys()) + [BENCHMARK]

    prices = _download_prices(tickers, HORIZON_DAYS_1Y)
    if BENCHMARK not in prices:
        raise RuntimeError(f"Pas de données pour benchmark {BENCHMARK}")

    bench_df = prices[BENCHMARK]
    benchmark_perf_1y = _compute_period_return(bench_df, HORIZON_DAYS_1Y, today)

    ticker_snaps: Dict[str, TickerSnapshot] = {}
    perfs_1y: List[float] = []

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
        raise RuntimeError("Aucune performance IA calculée.")

    avg_ai_1y = statistics.mean(perfs_1y)
    ai_vs_bench_spread = avg_ai_1y - benchmark_perf_1y

    # Score perf (surperf vs benchmark)
    for snap in ticker_snaps.values():
        spread = snap.perf.perf_1y - benchmark_perf_1y
        perf_score = max(0.0, min(100.0, (spread + 30) * (100.0 / 60.0)))
        snap.score_perf = perf_score

    # Score news + total
    for symbol, snap in ticker_snaps.items():
        news_score = compute_ticker_news_score(symbol)
        snap.score_news = news_score
        if news_score is None:
            snap.score_total = snap.score_perf
        else:
            snap.score_total = 0.6 * snap.score_perf + 0.4 * news_score

    # Groupes
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

    # Score global IA
    group_scores = [g.score for g in groups.values()] or [0.0]
    avg_group_score = statistics.mean(group_scores)
    spread_score = max(0.0, min(100.0, (ai_vs_bench_spread + 30) * (100.0 / 60.0)))

    if global_news_ratio is None:
        news_global_score = 60.0
    else:
        news_global_score = global_news_ratio * 100.0

    score_global = (
        0.5 * avg_group_score +
        0.3 * spread_score +
        0.2 * news_global_score
    )

    # Statut
    if score_global >= 80:
        status = "Cycle IA fort / haussier"
        comment = (
            "Les géants IA surperforent nettement le marché, "
            "infra + GPU + semi sont bien orientés et le flux de news est globalement positif."
        )
    elif score_global >= 60:
        status = "Cycle IA positif / modéré"
        comment = (
            "Les signaux IA sont globalement bons, mais certaines poches sont en digestion "
            "ou plus contrastées."
        )
    elif score_global >= 50:
        status = "IA en plateau / normalisation"
        comment = (
            "L’IA évolue à un rythme proche du marché, avec un contexte plus neutre."
        )
    else:
        status = "IA en stress / risque"
        comment = (
            "Les leaders IA ne surperforment plus, certains segments sont sous pression "
            "et les news deviennent plus négatives."
        )

    return AIHealthSnapshot(
        as_of=today,
        benchmark_perf_1y=benchmark_perf_1y,
        avg_ai_1y=avg_ai_1y,
        ai_vs_bench_spread=ai_vs_bench_spread,
        ticker_snaps=ticker_snaps,
        groups=groups,
        news_global_ratio=global_news_ratio,
        score_global=round(score_global, 1),
        status=status,
        comment=comment,
    )


# ========= HISTORIQUE & ALERTES =========

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
            "score_global": f"{snap.score_global:.2f}",
            "benchmark_perf_1y": f"{snap.benchmark_perf_1y:.2f}",
            "avg_ai_1y": f"{snap.avg_ai_1y:.2f}",
            "ai_vs_bench_spread": f"{snap.ai_vs_bench_spread:.2f}",
        }
        for gname, grp in snap.groups.items():
            row[f"group_{gname}_score"] = f"{grp.score:.2f}"
        writer.writerow(row)


def generate_alerts(snap: AIHealthSnapshot, last_row: Optional[Dict[str, str]]) -> List[str]:
    alerts: List[str] = []

    if snap.score_global < ALERT_SCORE_LOW:
        alerts.append(
            f"ALERTE: Score global IA faible ({snap.score_global:.1f} < {ALERT_SCORE_LOW}). "
            "Le marché IA semble sous pression."
        )

    if snap.score_global > ALERT_SCORE_HIGH:
        alerts.append(
            f"NOTE: Score global IA très élevé ({snap.score_global:.1f} > {ALERT_SCORE_HIGH}). "
            "Phase possible de surchauffe / euphorie."
        )

    if last_row is not None and "score_global" in last_row:
        try:
            last_score = float(last_row["score_global"])
            delta = snap.score_global - last_score
            if delta <= -ALERT_SCORE_DROP:
                alerts.append(
                    f"ALERTE: Dégradation rapide du score IA ({delta:.1f} pts vs dernier run). "
                    "Surveille une possible rupture ou choc IA."
                )
            elif delta >= ALERT_SCORE_DROP:
                alerts.append(
                    f"NOTE: Amélioration rapide du score IA (+{delta:.1f} pts vs dernier run). "
                    "Regain d’optimisme ou bonnes nouvelles fortes."
                )
        except ValueError:
            pass

    return alerts


def compute_macro_reco(snap: AIHealthSnapshot, delta_global: Optional[float]) -> str:
    """Recommandation macro : ne rien faire / prudence / cycle fort."""
    sg = snap.score_global

    if sg >= 80:
        if delta_global is not None and delta_global > 5:
            return (
                "Cycle IA fort et en accélération : continuer ton DCA, "
                "ne pas augmenter le risque, accepter la volatilité."
            )
        return (
            "Cycle IA fort mais stable : ne rien changer à ta stratégie, "
            "éviter de 'surinvestir' par euphorie."
        )

    if sg >= 60:
        if delta_global is not None and delta_global < -5:
            return (
                "Cycle IA encore positif mais en ralentissement : prudence, "
                "garder le plan mais éviter d’ajouter des mises exceptionnelles."
            )
        return (
            "Cycle IA modérément haussier : ne rien faire de spécial, "
            "laisser tourner ton plan automatique."
        )

    if sg >= 50:
        return (
            "IA en plateau : prudence renforcée, garder le long terme, "
            "réserver les gros ajouts de capital pour plus tard."
        )

    return (
        "Zone de stress IA : n’ajouter que très progressivement, "
        "garder ton horizon long terme et éviter toute décision émotionnelle."
    )


# ========= AFFICHAGE =========

def print_ai_snapshot(
    snap: AIHealthSnapshot,
    last_row: Optional[Dict[str, str]],
    delta_global: Optional[float],
    group_deltas: Dict[str, Optional[float]],
) -> None:
    flag = flag_for_score(snap.score_global)
    arr = arrow_for_delta(delta_global)
    delta_str = format_delta(delta_global)

    print(f"\n=== RADAR IA AU {snap.as_of} ===\n")
    print(f"Score global IA             : {snap.score_global:5.1f} / 100 {flag} {arr} {delta_str}")
    print(f"Benchmark 1 an ({BENCHMARK}) : {snap.benchmark_perf_1y:6.2f}%")
    print(f"Perf moyenne IA 1 an        : {snap.avg_ai_1y:6.2f}%")
    print(f"Surperf IA vs bench (1 an)  : {snap.ai_vs_bench_spread:6.2f} pts")
    if snap.news_global_ratio is not None:
        print(
            f"Sentiment news IA global    : {snap.news_global_ratio*100:5.1f}% "
            f"d’articles positifs approx."
        )
    else:
        print("Sentiment news IA global    : non évalué (NEWS_API_KEY manquant)")
    print(f"Statut                      : {snap.status}")
    print(f"Commentaire                 : {snap.comment}\n")

    print("=== GROUPES IA ===")
    for gname, grp in snap.groups.items():
        g_flag = flag_for_score(grp.score)
        d = group_deltas.get(gname)
        g_arr = arrow_for_delta(d)
        g_delta = format_delta(d)
        print(f"- {gname:11s} : {grp.score:5.1f} / 100 {g_flag} {g_arr} {g_delta} "
              f"({', '.join(t.info.symbol for t in grp.tickers)})")
    print()

    print("=== DÉTAIL PAR TICKER ===")
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


# ========= MAIN =========

def main() -> None:
    last_row = load_last_history_row()
    snap = build_ai_snapshot()

    # Deltas vs dernier run
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

    print_ai_snapshot(snap, last_row, delta_global, group_deltas)

    # Recommandation macro
    reco = compute_macro_reco(snap, delta_global)
    print("=== RECOMMANDATION MACRO INVESTISSEUR LONG TERME ===")
    print(reco)
    print()

    # Alertes
    alerts = generate_alerts(snap, last_row)
    if alerts:
        print("=== ALERTES / NOTES ===")
        for a in alerts:
            print("- " + a)
        print()

    save_snapshot_to_history(snap)
    print(f"Historique mis à jour dans : {HISTORY_FILE}\n")


if __name__ == "__main__":
    main()
