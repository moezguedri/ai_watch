# AI Watch Pro

AI Watch Pro is a small, opinionated tool for long-term investors who want a
simple way to monitor the state of the "AI trade" across a handful of liquid,
large-cap leaders.

It is **not** a trading system, and it does **not** attempt to time short-term
moves. Instead, it produces a daily (or periodic) "AI radar" snapshot, plus a
static web dashboard that can be published via GitHub Pages.

---

## Features

- Monitors a small AI universe:
  - Hyperscalers: MSFT, GOOGL, AMZN
  - Consumer AI: META
  - GPU: NVDA, AMD
  - Semiconductors: ASML, AVGO
- Compares AI performance vs the S&P 500 (^GSPC)
- Computes:
  - 1Y / 3M / 1M performance windows
  - Per-ticker performance scores vs benchmark
  - Per-ticker news sentiment scores (via NewsAPI + VADER)
  - Per-group AI scores (hyperscaler / consumer / GPU / semi)
  - Global AI score (raw and smoothed)
  - Simple bubble-risk heuristic
- Maintains a CSV history (`ai_watch_history.csv`)
- Generates a static HTML dashboard (`dashboard/index.html`) with:
  - Market regime & summary
  - Global AI score chart (raw vs smoothed)
  - AI group scores over time
  - AI allocation engine (by group and by ticker)
  - Interpretation & actions commentary
  - Optional methodology section explaining the full logic
- Can be run:
  - locally, from the command line, or
  - automatically via GitHub Actions + GitHub Pages

---

## Installation (local use)

### 1. Clone the repository

```bash
git clone https://github.com/<your-user>/ai_watch.git
cd ai_watch
