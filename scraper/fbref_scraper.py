import os
import pandas as pd
import time
import random
from playwright.sync_api import sync_playwright
from io import StringIO
from pathlib import Path
from datetime import datetime

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


def scrape_table(page, url, table_id, retries=3):
    for attempt in range(retries):
        try:
            print(f"  [{table_id}] Loading... (attempt {attempt + 1})")
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(3000)

            # Wait for table to appear
            page.wait_for_selector(f"#{table_id}", timeout=30000)

            html = page.content()
            df = pd.read_html(StringIO(html), attrs={"id": table_id})[0]

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
            error_msg = str(e)[:100]
            print(f"    Error: {error_msg}")
            if attempt < retries - 1:
                wait = random.uniform(5, 10)
                print(f"    Waiting {wait:.0f}s before retry...")
                time.sleep(wait)
    return None


def scrape_all_tables():
    print("\nLaunching browser...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()

        # Warm up - let Cloudflare pass
        print("Warming up (Cloudflare)...")
        page.goto("https://fbref.com", wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(5000)

        dfs = {}
        failed = []
        for url, table_id in URLS.items():
            df = scrape_table(page, url, table_id)
            if df is not None:
                dfs[table_id] = df
            else:
                failed.append(table_id)
            # Rate limit - FBref blocks rapid requests
            wait = random.uniform(4, 7)
            print(f"    Waiting {wait:.0f}s...")
            time.sleep(wait)

        browser.close()

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
    print("FBref Scraper - Big 5 European Leagues (Playwright)")
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