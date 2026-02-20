# Football Alpha Analysis

Data-driven football analytics platform analyzing player performance across Europe's Top 5 leagues using a dual-source data pipeline (FBref + Understat) and cloud infrastructure.

**[Live Dashboard](https://football-alpha-analysis-dorukyildiz.streamlit.app/)**

---

## Overview

This project calculates "Alpha" metrics — borrowed from quantitative finance — to identify overperforming and underperforming players in the 2025-26 season.

| Metric | Formula | Meaning |
|--------|---------|---------|
| Finishing Alpha | Goals - xG | Shot conversion efficiency |
| Playmaking Alpha | Assists - xAG | Chance creation efficiency |
| Alpha per 90 | Finishing Alpha / 90s | Rate-adjusted outperformance |

**Positive Alpha** = Overperforming expectations
**Negative Alpha** = Underperforming expectations

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Data Sources | FBref (76 columns), Understat (10 xG columns) |
| Cloud | AWS S3, AWS Athena |
| Backend | Python, Pandas, NumPy, Scikit-learn |
| Scraping | Playwright (Understat), undetected-chromedriver (FBref) |
| Visualization | Matplotlib, Seaborn, adjustText |
| Dashboard | Streamlit |
| PDF Generation | ReportLab |
| Data Query | PyAthena (SQL) |
| CI/CD | GitHub Actions (weekly automated updates) |

---

## Features

| Feature | Description |
|---------|-------------|
| **Overview** | Season stats, xG vs Goals scatter, league comparison charts |
| **Player Search** | Find similar players using cosine similarity |
| **Scout Report** | PDF reports with percentile rankings and radar charts |
| **Player Comparison** | Side-by-side comparison with category breakdowns |
| **Team Analysis** | Squad analysis, composition, Goals vs xG, league comparison |
| **Rankings** | Position-based player ratings |
| **Transfer Finder** | Transfer targets filtered by position, age, and team |

---

## Project Structure

```
football-alpha-analysis/
├── src/
│   ├── analysis.py                 # AWS Athena data pipeline
│   ├── visualization.py            # Chart types
│   ├── similarity.py               # Player similarity engine (cosine similarity)
│   ├── comparison.py               # Player comparison
│   ├── team_analysis.py            # Team analytics
│   ├── scout_report.py             # PDF report generation
│   ├── player_ranking.py           # Position-based rankings
│   ├── transfer_recommendation.py  # Transfer target finder
│   ├── generate_athena_sql.py      # Athena table DDL generator
│   └── dashboard.py                # Streamlit web app
├── scraper/
│   ├── fbref_scraper.py            # FBref scraper (undetected-chromedriver)
│   ├── understat_scraper.py        # Understat scraper (Playwright)
│   ├── merge_data.py               # 3-step merge (exact + name + last-name)
│   ├── upload_to_s3.py             # S3 upload (merged/ + raw/)
│   └── run_pipeline.py             # Full pipeline runner
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_Alpha_Analysis.ipynb     # Alpha metrics deep dive
│   ├── 03_Player_Similarity.ipynb  # ML similarity algorithm
│   └── 04_Visualizations.ipynb     # Chart gallery
├── .github/
│   └── workflows/
│       └── update_data.yml         # Weekly automated data refresh
├── data/                           # Local data (gitignored)
├── requirements.txt
└── README.md
```

---

## Data Pipeline

```
FBref ──(scrape)──┐
                   ├──(merge)──> players_data.csv ──> S3 (merged/) ──> Athena ──> Streamlit
Understat ─(scrape)┘
```

### Data Sources

| Source | Data | Columns |
|--------|------|---------|
| FBref | Standard, Shooting, Keeper, Playing Time, Misc | 76 |
| Understat | xG, xA, npxG, xGChain, xGBuildup, shots, key passes, NPG | 10 |
| Computed | Finishing Alpha, Playmaking Alpha, per-90 metrics | 7 |

### Merge Process

The merge pipeline matches players across FBref and Understat using a 3-step approach:

1. **Exact match** — normalized name + team
2. **Name-only match** — for players with team name mismatches
3. **Last-name + team match** — for transliteration differences

Result: **95%+ xG match rate** across 2,600+ players.

### S3 Structure

```
s3://football-alpha-analysis-doruk/
├── merged/          ← Athena reads from here
│   └── players_data.csv
├── raw/             ← Source files
│   ├── fbref_players.csv
│   ├── understat_players.csv
│   └── understat_raw.csv
└── archive/         ← Old versions
```

### Automated Updates

GitHub Actions runs weekly (Monday 17:00 UTC):
1. Downloads latest FBref data from S3
2. Scrapes fresh Understat xG data
3. Merges datasets
4. Uploads to S3
5. Athena picks up new data automatically

> FBref scraping requires `undetected-chromedriver` (Cloudflare bypass) and runs locally. CI only handles Understat + merge.

---

## Installation

```bash
# Clone repository
git clone https://github.com/dorukyildiz/football-alpha-analysis.git
cd football-alpha-analysis

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure

# Run dashboard locally
streamlit run src/dashboard.py
```

### Running Scrapers Locally

```bash
# Install scraper dependencies
pip install undetected-chromedriver selenium playwright
playwright install chromium

# Run full pipeline
cd scraper && python run_pipeline.py

# Or run individually
python fbref_scraper.py      # FBref (opens browser, Cloudflare bypass)
python understat_scraper.py  # Understat (Playwright)
python merge_data.py         # Merge datasets
python upload_to_s3.py       # Upload to S3
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_EDA.ipynb` | Data exploration, missing values, distributions, xG coverage |
| `02_Alpha_Analysis.ipynb` | Alpha metrics, xGChain/xGBuildup, league comparison |
| `03_Player_Similarity.ipynb` | Cosine similarity algorithm with dual-source metrics |
| `04_Visualizations.ipynb` | 11 chart types including xGChain involvement network |

```bash
pip install jupyter
jupyter notebook
```

---

## Screenshots

### Dashboard Overview
- Season statistics with key metrics
- Interactive xG vs Goals scatter plot
- League efficiency comparison

### Scout Report
- Professional PDF export
- Percentile rankings vs same position
- Radar chart visualization
- Strengths/weaknesses analysis

### Player Comparison
- Side-by-side metrics comparison
- Radar chart overlay
- Category breakdown (Attacking, Passing, Defensive)

---

## Author

**Doruk YILDIZ**

- GitHub: [@dorukyildiz](https://github.com/dorukyildiz)