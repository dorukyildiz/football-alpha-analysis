import pandas as pd
import numpy as np
import os
import unicodedata
import re
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

TEAM_NAME_MAP = {
    'Manchester United': 'Manchester Utd',
    'Newcastle United': 'Newcastle Utd',
    'Wolverhampton Wanderers': 'Wolves',
    'Nottingham Forest': "Nott'ham Forest",
    'Leicester': 'Leicester City',
    'Atletico Madrid': 'Atlético Madrid',
    'Real Betis': 'Betis',
    'Bayer Leverkusen': 'Leverkusen',
    'Borussia Dortmund': 'Dortmund',
    'Borussia M.Gladbach': "M'Gladbach",
    'RasenBallsport Leipzig': 'RB Leipzig',
    'Eintracht Frankfurt': 'Eint Frankfurt',
    'VfB Stuttgart': 'Stuttgart',
    'VfL Wolfsburg': 'Wolfsburg',
    'SC Freiburg': 'Freiburg',
    'FC Augsburg': 'Augsburg',
    'FSV Mainz 05': 'Mainz 05',
    '1. FC Heidenheim': 'Heidenheim',
    'FC St. Pauli': 'St. Pauli',
    '1. FC Union Berlin': 'Union Berlin',
    'TSG Hoffenheim': 'Hoffenheim',
    'VfL Bochum': 'Bochum',
    'AC Milan': 'Milan',
    'Paris Saint Germain': 'Paris S-G',
    'Olympique Marseille': 'Marseille',
    'Olympique Lyonnais': 'Lyon',
    'AS Monaco': 'Monaco',
    'LOSC Lille': 'Lille',
    'Stade Rennais': 'Rennes',
    'RC Strasbourg Alsace': 'Strasbourg',
    'Stade Brestois 29': 'Brest',
    'RC Lens': 'Lens',
    'OGC Nice': 'Nice',
    'FC Nantes': 'Nantes',
    'Stade de Reims': 'Reims',
    'AJ Auxerre': 'Auxerre',
    'Angers SCO': 'Angers',
    'Le Havre AC': 'Le Havre',
    'AS Saint-Etienne': 'Saint-Étienne',
}

LEAGUE_MAP = {
    'Premier League': 'eng Premier League',
    'La Liga': 'es La Liga',
    'Bundesliga': 'de Bundesliga',
    'Serie A': 'it Serie A',
    'Ligue 1': 'fr Ligue 1',
}


def normalize_name(name):
    if pd.isna(name):
        return ''
    nfkd = unicodedata.normalize('NFKD', str(name))
    ascii_name = nfkd.encode('ASCII', 'ignore').decode('ASCII')
    ascii_name = ascii_name.lower().strip()
    ascii_name = re.sub(r'\s+', ' ', ascii_name)
    return ascii_name


def normalize_squad(squad):
    if pd.isna(squad):
        return ''
    return str(squad).strip().lower()


def load_fbref_data(path=None):
    path = path or DATA_DIR / 'fbref_players.csv'
    df = pd.read_csv(path)
    df.columns = [c.lower().replace('+', '_').replace('-', '_').replace('/', '_')
                  .replace('%', 'pct').replace(' ', '_') for c in df.columns]
    df['player_norm'] = df['player'].apply(normalize_name)
    df['squad_norm'] = df['squad'].apply(normalize_squad)
    print(f"  FBref: {len(df)} players loaded")
    return df


def load_understat_data(path=None):
    path = path or DATA_DIR / 'understat_players.csv'
    df = pd.read_csv(path)
    df['squad_mapped'] = df['squad'].map(TEAM_NAME_MAP).fillna(df['squad'])
    df['comp_mapped'] = df['comp'].map(LEAGUE_MAP).fillna(df['comp'])
    df['player_norm'] = df['player'].apply(normalize_name)
    df['squad_norm'] = df['squad_mapped'].apply(normalize_squad)
    print(f"  Understat: {len(df)} players loaded")
    return df


def merge_exact(fbref_df, us_df):
    xg_cols = ['player_norm', 'squad_norm', 'xG', 'xA', 'npxG',
               'xGChain', 'xGBuildup', 'us_shots', 'us_key_passes', 'us_npg']
    available = [c for c in xg_cols if c in us_df.columns]

    merged = fbref_df.merge(us_df[available], on=['player_norm', 'squad_norm'], how='left')
    matched = merged['xG'].notna().sum()
    print(f"  Exact match: {matched}/{len(merged)} ({matched / len(merged) * 100:.1f}%)")
    return merged


def merge_fuzzy_name_only(merged_df, us_df):
    unmatched_mask = merged_df['xG'].isna()
    unmatched_count = unmatched_mask.sum()
    if unmatched_count == 0:
        return merged_df

    print(f"  Attempting name-only match for {unmatched_count} unmatched players...")
    us_name_lookup = us_df.drop_duplicates(subset='player_norm', keep='first')
    us_name_dict = us_name_lookup.set_index('player_norm')[
        ['xG', 'xA', 'npxG', 'xGChain', 'xGBuildup', 'us_shots', 'us_key_passes', 'us_npg']
    ].to_dict('index')

    filled = 0
    for idx in merged_df[unmatched_mask].index:
        player_norm = merged_df.loc[idx, 'player_norm']
        if player_norm in us_name_dict:
            for col, val in us_name_dict[player_norm].items():
                merged_df.loc[idx, col] = val
            filled += 1

    print(f"  Name-only match: +{filled} players matched")
    return merged_df


def merge_by_last_name(merged_df, us_df):
    unmatched_mask = merged_df['xG'].isna()
    unmatched_count = unmatched_mask.sum()
    if unmatched_count == 0:
        return merged_df

    print(f"  Attempting last-name + team match for {unmatched_count} remaining...")

    def get_last_name(name):
        parts = str(name).split()
        return parts[-1] if parts else ''

    us_df_copy = us_df.copy()
    us_df_copy['last_name'] = us_df_copy['player_norm'].apply(get_last_name)
    us_lastname_dict = {}
    for _, row in us_df_copy.drop_duplicates(subset=['last_name', 'squad_norm'], keep='first').iterrows():
        key = (row['last_name'], row['squad_norm'])
        us_lastname_dict[key] = {
            'xG': row['xG'], 'xA': row['xA'], 'npxG': row['npxG'],
            'xGChain': row.get('xGChain'), 'xGBuildup': row.get('xGBuildup'),
            'us_shots': row.get('us_shots'), 'us_key_passes': row.get('us_key_passes'),
            'us_npg': row.get('us_npg')
        }

    filled = 0
    for idx in merged_df[unmatched_mask].index:
        last_name = get_last_name(merged_df.loc[idx, 'player_norm'])
        squad_norm = merged_df.loc[idx, 'squad_norm']
        key = (last_name, squad_norm)
        if key in us_lastname_dict:
            for col, val in us_lastname_dict[key].items():
                merged_df.loc[idx, col] = val
            filled += 1

    print(f"  Last-name match: +{filled} players matched")
    return merged_df


def calculate_alpha_metrics(df):
    if 'col_90s' not in df.columns and '90s' in df.columns:
        df['col_90s'] = pd.to_numeric(df['90s'], errors='coerce')
    elif 'col_90s' not in df.columns and 'min' in df.columns:
        df['col_90s'] = pd.to_numeric(df['min'], errors='coerce') / 90

    for col in ['gls', 'ast', 'xG', 'xA', 'npxG', 'col_90s']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'xG' in df.columns and 'gls' in df.columns:
        df['finishing_alpha'] = df['gls'] - df['xG']
    if 'xA' in df.columns and 'ast' in df.columns:
        df['xag'] = df['xA']
        df['playmaking_alpha'] = df['ast'] - df['xA']

    if 'xG' in df.columns:
        df['xg'] = df['xG']
        df = df.drop(columns=['xG'])
    if 'xA' in df.columns:
        df = df.drop(columns=['xA'])
    if 'npxG' in df.columns:
        df['npxg'] = df['npxG']
        df = df.drop(columns=['npxG'])
    if 'xGChain' in df.columns:
        df['xgchain'] = df['xGChain']
        df = df.drop(columns=['xGChain'])
    if 'xGBuildup' in df.columns:
        df['xgbuildup'] = df['xGBuildup']
        df = df.drop(columns=['xGBuildup'])

    if 'col_90s' in df.columns:
        mask = df['col_90s'] > 0
        if 'gls' in df.columns:
            df.loc[mask, 'gls_per90'] = df.loc[mask, 'gls'] / df.loc[mask, 'col_90s']
        if 'xg' in df.columns:
            df.loc[mask, 'xg_per90'] = df.loc[mask, 'xg'] / df.loc[mask, 'col_90s']
        if 'ast' in df.columns:
            df.loc[mask, 'ast_per90'] = df.loc[mask, 'ast'] / df.loc[mask, 'col_90s']
        if 'xag' in df.columns:
            df.loc[mask, 'xag_per90'] = df.loc[mask, 'xag'] / df.loc[mask, 'col_90s']
        if 'finishing_alpha' in df.columns:
            df.loc[mask, 'alpha_per90'] = df.loc[mask, 'finishing_alpha'] / df.loc[mask, 'col_90s']

    print(f"  Alpha metrics calculated")
    return df


def print_merge_report(df):
    total = len(df)
    xg_col = 'xg' if 'xg' in df.columns else None
    has_xg = df[xg_col].notna().sum() if xg_col else 0
    missing_xg = total - has_xg

    print(f"\nMERGE REPORT")
    print(f"Total players: {total}")
    print(f"With xG data: {has_xg} ({has_xg / total * 100:.1f}%)")
    print(f"Missing xG: {missing_xg} ({missing_xg / total * 100:.1f}%)")

    if 'comp' in df.columns and xg_col:
        print(f"\nBy league:")
        for league in df['comp'].dropna().unique():
            league_df = df[df['comp'] == league]
            league_xg = league_df[xg_col].notna().sum()
            print(f"  {league}: {league_xg}/{len(league_df)} ({league_xg / len(league_df) * 100:.1f}%)")

    if has_xg > 0 and 'finishing_alpha' in df.columns:
        print(f"\nTop 5 Finishing Alpha:")
        top5 = df.dropna(subset=['finishing_alpha']).nlargest(5, 'finishing_alpha')
        for _, row in top5.iterrows():
            print(f"  {row['player']}: {row['finishing_alpha']:.2f} "
                  f"(Goals: {row['gls']:.0f}, xG: {row[xg_col]:.1f})")

    if missing_xg > 0 and 'gls' in df.columns and xg_col:
        print(f"\nTop unmatched players by goals:")
        unmatched = df[df[xg_col].isna()].nlargest(10, 'gls')
        for _, row in unmatched.iterrows():
            print(f"  {row['player']} ({row.get('squad', '?')}) - {row['gls']:.0f} goals")


def run_merge(fbref_path=None, understat_path=None, output_path=None):
    fbref_path = fbref_path or DATA_DIR / 'fbref_players.csv'
    understat_path = understat_path or DATA_DIR / 'understat_players.csv'
    output_path = output_path or DATA_DIR / 'players_data.csv'

    print("=" * 60)
    print("Data Merge: FBref + Understat")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\nLoading data...")
    fbref_df = load_fbref_data(fbref_path)
    us_df = load_understat_data(understat_path)

    print("\nMerging data (3-step process)...")
    merged = merge_exact(fbref_df, us_df)
    merged = merge_fuzzy_name_only(merged, us_df)
    merged = merge_by_last_name(merged, us_df)

    print("\nCalculating alpha metrics...")
    merged = calculate_alpha_metrics(merged)

    temp_cols = ['player_norm', 'squad_norm', 'player_clean', 'squad_clean']
    merged = merged.drop(columns=[c for c in temp_cols if c in merged.columns])

    os.makedirs(DATA_DIR, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    print_merge_report(merged)
    print(f"\nMERGE COMPLETE!")
    return merged


if __name__ == "__main__":
    run_merge()