import pandas as pd
import time
from datetime import datetime
import warnings
import re
import os

warnings.filterwarnings('ignore')

# 13 Leagues Configuration
LEAGUES = {
    'Premier League': {
        'url': 'https://fbref.com/en/comps/9/stats/Premier-League-Stats',
        'country': 'ENG'
    },
    'La Liga': {
        'url': 'https://fbref.com/en/comps/12/stats/La-Liga-Stats',
        'country': 'ESP'
    },
    'Serie A': {
        'url': 'https://fbref.com/en/comps/11/stats/Serie-A-Stats',
        'country': 'ITA'
    },
    'Bundesliga': {
        'url': 'https://fbref.com/en/comps/20/stats/Bundesliga-Stats',
        'country': 'GER'
    },
    'Ligue 1': {
        'url': 'https://fbref.com/en/comps/13/stats/Ligue-1-Stats',
        'country': 'FRA'
    },
    'Primeira Liga': {
        'url': 'https://fbref.com/en/comps/32/stats/Primeira-Liga-Stats',
        'country': 'POR'
    },
    'Eredivisie': {
        'url': 'https://fbref.com/en/comps/23/stats/Eredivisie-Stats',
        'country': 'NED'
    },
    'Pro League': {
        'url': 'https://fbref.com/en/comps/37/stats/Belgian-Pro-League-Stats',
        'country': 'BEL'
    },
    'Super Lig': {
        'url': 'https://fbref.com/en/comps/26/stats/Super-Lig-Stats',
        'country': 'TUR'
    },
    'Austrian Bundesliga': {
        'url': 'https://fbref.com/en/comps/56/stats/Austrian-Bundesliga-Stats',
        'country': 'AUT'
    },
    'Swiss Super League': {
        'url': 'https://fbref.com/en/comps/57/stats/Swiss-Super-League-Stats',
        'country': 'SUI'
    },
    'Super League Greece': {
        'url': 'https://fbref.com/en/comps/27/stats/Super-League-Greece-Stats',
        'country': 'GRE'
    },
    'Scottish Premiership': {
        'url': 'https://fbref.com/en/comps/40/stats/Scottish-Premiership-Stats',
        'country': 'SCO'
    }
}

# Stat types to scrape
STAT_TYPES = {
    'standard': 'stats',
    'shooting': 'shooting',
    'passing': 'passing',
    'gca': 'gca',
    'defense': 'defense',
    'possession': 'possession',
    'playing_time': 'playingtime',
    'misc': 'misc',
    'keeper': 'keepers',
    'keeper_adv': 'keepersadv'
}


def get_stat_url(base_url, stat_type):
    """Convert standard stats URL to specific stat type URL"""
    if stat_type == 'stats':
        return base_url
    return base_url.replace('/stats/', f'/{stat_type}/')


def scrape_table(url, retries=3):
    """Scrape table from URL with retries"""
    for attempt in range(retries):
        try:
            tables = pd.read_html(url)
            if tables:
                for table in tables:
                    cols = table.columns.tolist()
                    if 'Player' in cols:
                        return table
                    if any('Player' in str(c) for c in cols):
                        return table
                return max(tables, key=len)
            return None
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return None


def flatten_columns(df):
    """Flatten multi-level column names"""
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            if col[0] == col[1]:
                new_cols.append(col[0])
            elif col[1] == '':
                new_cols.append(col[0])
            else:
                new_cols.append(f"{col[0]}_{col[1]}")
        df.columns = new_cols
    return df


def clean_dataframe(df):
    """Clean dataframe"""
    df = flatten_columns(df)

    if 'Player' in df.columns:
        df = df[df['Player'] != 'Player']
        df = df[df['Player'].notna()]

    df = df.reset_index(drop=True)
    return df


def scrape_league(league_name, league_config):
    """Scrape all stats for a single league"""
    print(f"\n{'=' * 60}")
    print(f"Scraping: {league_name}")
    print(f"{'=' * 60}")

    base_url = league_config['url']
    all_stats = {}

    for stat_name, stat_type in STAT_TYPES.items():
        url = get_stat_url(base_url, stat_type)
        print(f"  [{stat_name}] Fetching...")

        df = scrape_table(url)

        if df is not None:
            df = clean_dataframe(df)
            all_stats[stat_name] = df
            print(f"    ✓ {len(df)} players")
        else:
            print(f"    ✗ Failed")

        time.sleep(3)

    if 'standard' not in all_stats:
        print(f"  [ERROR] No standard stats for {league_name}")
        return None

    merged = all_stats['standard'].copy()
    merged['Comp'] = league_name
    merged['Country'] = league_config['country']

    for stat_name, stat_df in all_stats.items():
        if stat_name == 'standard':
            continue

        suffix = f'_{stat_name}'
        stat_df = stat_df.add_suffix(suffix)

        merge_cols = []
        if f'Player{suffix}' in stat_df.columns:
            stat_df = stat_df.rename(columns={f'Player{suffix}': 'Player'})
            merge_cols.append('Player')
        if f'Squad{suffix}' in stat_df.columns:
            stat_df = stat_df.rename(columns={f'Squad{suffix}': 'Squad'})
            if 'Squad' in merged.columns:
                merge_cols.append('Squad')

        if not merge_cols:
            continue

        try:
            merged = merged.merge(stat_df, on=merge_cols, how='left')
        except Exception as e:
            print(f"    [WARN] Could not merge {stat_name}: {e}")

    print(f"  [DONE] {len(merged)} players, {len(merged.columns)} columns")
    return merged


def scrape_all_leagues():
    """Scrape all 13 leagues"""
    print("=" * 60)
    print("FBref Football Data Scraper")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Leagues: {len(LEAGUES)}")
    print("=" * 60)

    all_data = []

    for league_name, league_config in LEAGUES.items():
        df = scrape_league(league_name, league_config)

        if df is not None:
            all_data.append(df)

        time.sleep(5)

    if not all_data:
        print("\n[ERROR] No data scraped!")
        return None

    combined = pd.concat(all_data, ignore_index=True)

    print(f"\n{'=' * 60}")
    print("SCRAPING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total Players: {len(combined)}")
    print(f"Total Columns: {len(combined.columns)}")
    print(f"Leagues: {combined['Comp'].nunique()}")

    return combined


def clean_column_names(df):
    """Clean column names for Athena compatibility"""
    new_columns = []
    seen = {}

    for col in df.columns:
        clean = str(col).lower()
        clean = re.sub(r'[^a-z0-9_]', '_', clean)
        clean = re.sub(r'_+', '_', clean)
        clean = clean.strip('_')
        if not clean:
            clean = 'col'
        if clean[0].isdigit():
            clean = 'col_' + clean

        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 1

        new_columns.append(clean)

    df.columns = new_columns
    return df


def save_data(df, output_path='data/players_data.csv'):
    """Save data to CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = clean_column_names(df)
    df.to_csv(output_path, index=False)
    print(f"\n[SAVED] {output_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    return output_path


if __name__ == "__main__":
    df = scrape_all_leagues()

    if df is not None:
        save_data(df, 'data/players_data.csv')
        print("\n[OK] Scraping complete!")
        print("Next: Upload to S3 and update Athena table")