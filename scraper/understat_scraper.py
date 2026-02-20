from playwright.sync_api import sync_playwright
import json
import pandas as pd
import time
import re
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

CURRENT_SEASON = 2025

# Detect CI environment
IS_CI = os.environ.get('CI', 'false').lower() == 'true'

# 2025-26 season teams (Understat URL format)
LEAGUE_TEAMS = {
    'Premier League': [
        'Arsenal', 'Aston_Villa', 'Bournemouth', 'Brentford', 'Brighton',
        'Burnley', 'Chelsea', 'Crystal_Palace', 'Everton', 'Fulham',
        'Leeds', 'Liverpool', 'Manchester_City', 'Manchester_United',
        'Newcastle_United', 'Nottingham_Forest', 'Sunderland', 'Tottenham',
        'West_Ham', 'Wolverhampton_Wanderers'
    ],
    'La Liga': [
        'Alaves', 'Athletic_Club', 'Atletico_Madrid', 'Barcelona',
        'Real_Betis', 'Celta_Vigo', 'Elche', 'Espanyol', 'Getafe',
        'Girona', 'Levante', 'Mallorca', 'Osasuna', 'Rayo_Vallecano',
        'Real_Madrid', 'Real_Oviedo', 'Real_Sociedad', 'Sevilla',
        'Valencia', 'Villarreal'
    ],
    'Bundesliga': [
        'Augsburg', 'Bayer_Leverkusen', 'Bayern_Munich', 'Borussia_Dortmund',
        'Borussia_M.Gladbach', 'Eintracht_Frankfurt', 'Freiburg',
        'Hamburger_SV', 'FC Heidenheim', 'Hoffenheim', 'FC_Cologne',
        'Mainz_05', 'RasenBallsport_Leipzig', 'St._Pauli', 'VfB_Stuttgart',
        'Union_Berlin', 'Werder_Bremen', 'Wolfsburg'
    ],
    'Serie A': [
        'AC_Milan', 'Atalanta', 'Bologna', 'Cagliari', 'Como',
        'Cremonese', 'Fiorentina', 'Genoa', 'Verona', 'Inter',
        'Juventus', 'Lazio', 'Lecce', 'Napoli',
        'Parma_Calcio_1913', 'Pisa', 'Roma', 'Sassuolo',
        'Torino', 'Udinese'
    ],
    'Ligue 1': [
        'Angers', 'Auxerre', 'Brest', 'Le_Havre', 'Lens',
        'Lille', 'Lorient', 'Lyon', 'Marseille', 'Metz',
        'Monaco', 'Nantes', 'Nice', 'Paris_FC',
        'Paris_Saint_Germain', 'Rennes', 'Strasbourg', 'Toulouse'
    ]
}


def scrape_team_players(page, team_name, league, season=CURRENT_SEASON):
    """Scrape player data from a team page using Playwright"""
    url = f"https://understat.com/team/{team_name}/{season}"

    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)

        # Method 1: Execute JS to get playersData
        players_data = None
        try:
            players_data = page.evaluate("() => { try { return playersData; } catch(e) { return null; } }")
        except:
            pass

        # Method 2: Regex on page content
        if not players_data:
            content = page.content()
            match = re.search(r"playersData\s*=\s*JSON\.parse\('(.+?)'\)", content)
            if match:
                json_str = match.group(1)
                json_str = json_str.encode('utf-8').decode('unicode_escape')
                players_data = json.loads(json_str)

        if not players_data:
            print(f"      X {team_name}: no player data found")
            return None

        df = pd.DataFrame(players_data)
        df['league'] = league
        print(f"      OK {team_name}: {len(df)} players")
        return df

    except Exception as e:
        error_msg = str(e)[:80]
        print(f"      X {team_name}: {error_msg}")
        return None


def scrape_all_leagues(season=CURRENT_SEASON):
    print("=" * 60)
    print("Understat Scraper - Big 5 European Leagues (Playwright)")
    print(f"Season: {season}/{season + 1}")
    print(f"CI mode: {IS_CI}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_dfs = []
    total_teams = sum(len(t) for t in LEAGUE_TEAMS.values())
    scraped = 0
    failed_teams = []

    with sync_playwright() as p:
        print("\nLaunching browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()

        # Warm up
        print("Warming up...")
        page.goto("https://understat.com", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(5000)

        for league, teams in LEAGUE_TEAMS.items():
            print(f"\n  --- {league} ({len(teams)} teams) ---")
            league_count = 0

            for team in teams:
                df = scrape_team_players(page, team, league, season)
                if df is not None:
                    all_dfs.append(df)
                    league_count += 1
                else:
                    failed_teams.append(f"{team} ({league})")
                scraped += 1
                if scraped % 10 == 0:
                    print(f"  ... progress: {scraped}/{total_teams} teams")
                page.wait_for_timeout(2000)

            print(f"  [{league}] Done: {league_count}/{len(teams)} teams scraped")

        browser.close()

    if failed_teams:
        print(f"\nFailed teams ({len(failed_teams)}):")
        for t in failed_teams:
            print(f"  - {t}")

    if not all_dfs:
        print("\nNo data scraped!")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal raw: {len(combined)} player entries")
    return combined


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