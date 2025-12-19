# Football Alpha Analysis

Data-driven football analytics platform analyzing player performance across Europe's Top 5 leagues using expected goals (xG) metrics and cloud infrastructure.

**[Live Dashboard](https://dorukyildiz-football-alpha-analysis.streamlit.app)**

---

## Overview

This project calculates "Alpha" metrics - the difference between actual performance and expected performance - to identify overperforming and underperforming players in the 2025-26 season.

| Metric | Formula | Meaning |
|--------|---------|---------|
| Finishing Alpha | Goals - xG | Shot conversion efficiency |
| Playmaking Alpha | Assists - xAG | Chance creation efficiency |

**Positive Alpha** = Overperforming expectations  
**Negative Alpha** = Underperforming expectations

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Cloud | AWS S3, AWS Athena |
| Backend | Python, Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn, adjustText |
| Dashboard | Streamlit |
| PDF Generation | ReportLab |
| Data Query | PyAthena (SQL) |

---

## Features

| Feature | Description                                                      |
|---------|------------------------------------------------------------------|
| **Overview** | Season stats, xG vs Goals scatter, league comparison charts      |
| **Player Search** | Search similar players (cosine similarity)                       |
| **Scout Report** | PDF reports with percentile rankings and charts                  |
| **Player Comparison** | Side-by-side comparison with radar charts and category breakdowns |
| **Team Analysis** | Squad analysis, composition pie chart, Goals vs xG, league comparison |
| **Rankings** | Position-based ratings                                           |
| **Transfer Finder** | Find transfer targets filtered by position, age, and team        |

---

## Project Structure
```
football-alpha-analysis/
├── src/
│   ├── analysis.py                 # AWS Athena data pipeline
│   ├── visualization.py            # 10 chart types
│   ├── similarity.py               # Player similarity engine
│   ├── comparison.py               # Player comparison
│   ├── team_analysis.py            # Team analytics
│   ├── scout_report.py             # PDF report generation
│   ├── player_ranking.py           # Position-based rankings
│   ├── transfer_recommendation.py  # Transfer targets
│   └── dashboard.py                # Streamlit web app
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_Alpha_Analysis.ipynb     # Alpha metrics deep dive
│   ├── 03_Player_Similarity.ipynb  # ML similarity algorithm
│   └── 04_Visualizations.ipynb     # Chart gallery
├── data/
├── outputs/
├── requirements.txt
└── README.md
```

---

## Data Pipeline
```
Kaggle Dataset → AWS S3 → AWS Athena → PyAthena → Pandas → Streamlit
```

- **Source:** 2,341 players from Top 5 European Leagues
- **Filtered:** 1,369 players with 5+ 90s played
- **Features:** 200+ statistical columns

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

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_EDA.ipynb` | Data exploration, missing values, distributions |
| `02_Alpha_Analysis.ipynb` | Alpha metrics calculation and analysis |
| `03_Player_Similarity.ipynb` | Cosine similarity algorithm explained |
| `04_Visualizations.ipynb` | Complete chart gallery |

Run notebooks:
```bash
jupyter notebook
```

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
