import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
import re
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

LEAGUES = {
    'EPL': 'Premier League',
    'La_Liga': 'La Liga',
    'Bundesliga': 'Bundesliga',
    'Serie_A': 'Serie A',
    'Ligue_1': 'Ligue 1'
}

CURRENT_SEASON = 2025

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36'
}


def scrape_league_players(league_code, season=CURRENT_SEASON):
    url = f"https://understat.com/league/{league_code}/{season}"
    print(f"  [{league_code}] Fetching {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"    ✗ Request failed: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    scripts = soup.find_all('script')

    players_data = None
    for script in scripts:
        if script.string and 'playersData' in script.string:
            match = re.search(r"playersData\s*=\s*JSON\.parse\('(.+?)'\)", script.string)
            if match:
                json_str = match.group(1)
                json_str = json_str.encode('utf-8').decode('unicode_escape')
                players_data = json.loads(json_str)
                break

    if players_data is None:
        print(f"    ✗ Could not find playersData")
        return None

    df = pd.DataFrame(players_data)
    df['league'] = LEAGUES[league_code]
    df['league_code'] = league_code

    print(f"    ✓ {len(df)} players")
    return df


def scrape_all_leagues(season=CURRENT_SEASON):
    print("=" * 60)
    print("Understat Scraper - Big 5 European Leagues")
    print(f"Season: {season}/{season + 1}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_dfs = []
    for league_code in LEAGUES:
        df = scrape_league_players(league_code, season)
        if df is not None:
            all_dfs.append(df)
        time.sleep(2)

    if not all_dfs:
        print("\n✗ No data scraped!")
        return None

    return pd.concat(all_dfs, ignore_index=True)


def clean_understat_data(df):
    rename_map = {
        'player_name': 'player',
        'team_title': 'squad',
        'league': 'comp',
        'goals': 'us_goals',
        'assists': 'us_assists',
        'shots': 'us_shots',
        'key_passes': 'us_key_passes',
        'yellow_cards': 'us_yellow',
        'red_cards': 'us_red',
        'games': 'us_games',
        'time': 'us_minutes',
        'npg': 'us_npg',
    }
    df = df.rename(columns=rename_map)

    numeric_cols = ['us_goals', 'us_assists', 'xG', 'xA', 'npxG',
                    'xGChain', 'xGBuildup', 'us_shots', 'us_key_passes',
                    'us_yellow', 'us_red', 'us_games', 'us_minutes', 'us_npg']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    keep_cols = ['id', 'player', 'squad', 'comp', 'position',
                 'us_games', 'us_minutes', 'us_goals', 'us_assists',
                 'xG', 'xA', 'npxG', 'xGChain', 'xGBuildup',
                 'us_shots', 'us_key_passes', 'us_npg']

    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()
    df['player_clean'] = df['player'].str.strip()
    df['squad_clean'] = df['squad'].str.strip()

    print(f"\n  Cleaned: {len(df)} players, {len(df.columns)} columns")
    return df


def run_understat_scraper(season=CURRENT_SEASON):
    raw_df = scrape_all_leagues(season)
    if raw_df is None:
        return None

    print("\nCleaning data...")
    cleaned = clean_understat_data(raw_df)

    os.makedirs(DATA_DIR, exist_ok=True)
    raw_df.to_csv(DATA_DIR / 'understat_raw.csv', index=False)
    cleaned.to_csv(DATA_DIR / 'understat_players.csv', index=False)

    print(f"\nUNDERSTAT SCRAPING COMPLETE!")
    print(f"Total Players: {len(cleaned)}")
    print(f"Leagues: {cleaned['comp'].nunique()}")
    return cleaned


if __name__ == "__main__":
    run_understat_scraper()