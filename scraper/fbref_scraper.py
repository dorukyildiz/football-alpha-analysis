import os
import pandas as pd
import time
from io import StringIO
from pathlib import Path
from datetime import datetime
from scrapling.fetchers import StealthySession

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# Post-Opta: only 5 available tables
URLS = {
    'https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats': 'stats_standard',
    'https://fbref.com/en/comps/Big5/shooting/players/Big-5-European-Leagues-Stats': 'stats_shooting',
    'https://fbref.com/en/comps/Big5/keepers/players/Big-5-European-Leagues-Stats': 'stats_keeper',
    'https://fbref.com/en/comps/Big5/playingtime/players/Big-5-European-Leagues-Stats': 'stats_playing_time',
    'https://fbref.com/en/comps/Big5/misc/players/Big-5-European-Leagues-Stats': 'stats_misc',
}


def scrape_table(session, url, table_id, retries=3):
    """Scrape a single FBref table using StealthySession"""
    for attempt in range(retries):
        try:
            print(f"  [{table_id}] Loading... (attempt {attempt + 1})")
            response = session.fetch(url)

            table = response.css(f'#{table_id}')
            if not table:
                print(f"    Table #{table_id} not found in page")
                if attempt < retries - 1:
                    time.sleep(5)
                continue

            df = pd.read_html(StringIO(table[0].html_content))[0]

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)

            df = df.loc[:, ~df.columns.duplicated()]
            if "Player" in df.columns:
                df = df[df["Player"] != "Player"]

            if len(df) < 10:
                print(f"    Warning: Table seems empty ({len(df)} rows)")
                return None

            print(f"    OK: {len(df)} players, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"    Error: {str(e)[:100]}")
            if attempt < retries - 1:
                time.sleep(5)
    return None


def scrape_all_tables():
    """Scrape all FBref tables using a single StealthySession"""
    dfs = {}
    failed = []

    print("\nStarting Scrapling StealthySession...")
    with StealthySession(headless=True, solve_cloudflare=True, network_idle=True, timeout=90000) as session:
        for url, table_id in URLS.items():
            df = scrape_table(session, url, table_id)
            if df is not None:
                dfs[table_id] = df
            else:
                failed.append(table_id)
            time.sleep(3)

    print(f"\n  Successfully scraped: {len(dfs)} tables")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    return dfs


def merge_dataframes(dfs):
    if 'stats_standard' not in dfs:
        raise ValueError("Missing stats_standard!")

    merged = dfs['stats_standard']
    merge_count = 1
    for name, df in dfs.items():
        if name != 'stats_standard':
            try:
                merged = merged.merge(df, on=['Player', 'Squad'],
                                      how='left', suffixes=('', f'_{name}'))
                merge_count += 1
            except Exception as e:
                print(f"  Warning: Could not merge {name}: {e}")

    print(f"  Merged {merge_count} tables")
    return merged


def clean_dataframe(df):
    cols_to_drop = []
    for col in df.columns:
        if any(x in col for x in ['Rk_stats_', 'Nation_stats_', 'Pos_stats_',
                                   'Comp_stats_', 'Age_stats_', 'Born_stats_', '90s_stats_']):
            cols_to_drop.append(col)
        if 'matches' in col.lower():
            cols_to_drop.append(col)

    df = df.drop(columns=cols_to_drop, errors='ignore')

    if 'Age' in df.columns:
        df['Age'] = df['Age'].astype(str).str.split('-').str[0]
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    print(f"  Final: {len(df)} players, {len(df.columns)} columns")
    return df


def run_scraper():
    print("=" * 60)
    print("FBref Scraper - Big 5 European Leagues (Scrapling)")
    print(f"Tables: {len(URLS)} (post-Opta)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    dfs = scrape_all_tables()
    if not dfs:
        print("\nNo data scraped!")
        return None

    print("\nMerging tables...")
    merged = merge_dataframes(dfs)
    print("Cleaning data...")
    cleaned = clean_dataframe(merged)

    os.makedirs(DATA_DIR, exist_ok=True)
    cleaned.to_csv(DATA_DIR / 'fbref_players.csv', index=False)

    print(f"\nFBREF SCRAPING COMPLETE!")
    print(f"Total Players: {len(cleaned)}")
    return cleaned


if __name__ == "__main__":
    run_scraper()