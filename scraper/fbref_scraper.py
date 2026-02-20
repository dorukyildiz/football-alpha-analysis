import os
import pandas as pd
import time
import random
import subprocess
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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


def get_chrome_version():
    """Auto-detect installed Chrome major version"""
    try:
        # macOS
        output = subprocess.check_output(
            ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '--version']
        ).decode()
        version = int(output.strip().split()[-1].split('.')[0])
        print(f"  Detected Chrome version: {version}")
        return version
    except Exception:
        pass
    try:
        # Linux
        output = subprocess.check_output(['google-chrome', '--version']).decode()
        version = int(output.strip().split()[-1].split('.')[0])
        print(f"  Detected Chrome version: {version}")
        return version
    except Exception:
        print("  Could not detect Chrome version, using default")
        return None


def create_driver():
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    chrome_version = get_chrome_version()
    if chrome_version:
        driver = uc.Chrome(options=options, version_main=chrome_version)
    else:
        driver = uc.Chrome(options=options)
    return driver


def scrape_table(driver, url, table_id, retries=3):
    for attempt in range(retries):
        try:
            print(f"  [{table_id}] Loading... (attempt {attempt + 1})")
            driver.get(url)
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.ID, table_id))
            )
            html = driver.page_source
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
            print(f"    Error: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return None


def scrape_all_tables():
    print("\nStarting browser...")
    driver = create_driver()
    print("Waiting for Cloudflare...")
    driver.get("https://fbref.com")
    time.sleep(5)

    dfs = {}
    failed = []
    for url, table_id in URLS.items():
        df = scrape_table(driver, url, table_id)
        if df is not None:
            dfs[table_id] = df
        else:
            failed.append(table_id)
        time.sleep(random.uniform(3, 5))

    driver.quit()
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
    print("FBref Scraper - Big 5 European Leagues")
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