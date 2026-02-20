import pandas as pd
import os
import streamlit as st

DATA_SOURCE = os.environ.get('DATA_SOURCE', 'local')
LOCAL_CSV_PATH = os.environ.get('LOCAL_CSV', 'data/players_data.csv')


def _load_from_athena():
    from pyathena import connect
    conn = connect(
        s3_staging_dir="s3://football-alpha-athena-results-doruk/",
        region_name="eu-central-1"
    )
    return pd.read_sql("SELECT * FROM football_analytics.players", conn)


def _load_from_csv(path=None):
    search_paths = [
        path or LOCAL_CSV_PATH,
        'data/players_data.csv',
        '../data/players_data.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'players_data.csv'),
    ]
    for p in search_paths:
        if os.path.exists(p):
            print(f"  Loaded from: {p}")
            return pd.read_csv(p)
    raise FileNotFoundError("Could not find players_data.csv")


@st.cache_data(ttl=43200)
def get_data():
    try:
        if DATA_SOURCE == 'local':
            raise Exception("Using local CSV")
        df = _load_from_athena()
        print(f"  Data loaded from Athena: {len(df)} rows")
    except Exception as e:
        print(f"  Athena unavailable ({e}), loading from CSV...")
        df = _load_from_csv()
        print(f"  Data loaded from CSV: {len(df)} rows")

    df.columns = [c.lower().strip() for c in df.columns]

    # Post-Opta numeric columns (Standard + Shooting + Keeper + Playing Time + Misc + Understat)
    numeric_cols = [
        # Standard stats
        'gls', 'ast', 'g_a', 'g_pk', 'pk', 'pkatt', 'crdy', 'crdr', 'g_a_pk',
        'mp', 'starts', 'min', 'col_90s', 'age',
        # Shooting
        'sh', 'sot', 'sotpct', 'sh_90', 'sot_90', 'g_sh', 'g_sot',
        # Keeper
        'ga', 'ga90', 'sota', 'saves', 'savepct', 'cs', 'cspct', 'pka', 'pksv', 'pkm',
        # Playing Time
        'mn_mp', 'minpct', 'mn_start', 'compl', 'subs', 'mn_sub', 'unsub', 'ppm',
        'ong', 'onga', 'on_off',
        # Misc
        'fls', 'fld', 'off', 'crs', 'int', 'tklw', 'pkwon', 'pkcon', 'og',
        # Understat xG
        'xg', 'xag', 'npxg', 'xgchain', 'xgbuildup',
        'us_shots', 'us_key_passes', 'us_npg',
        # Alpha metrics
        'finishing_alpha', 'playmaking_alpha',
        'gls_per90', 'xg_per90', 'ast_per90', 'xag_per90', 'alpha_per90',
    ]

    for col in numeric_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except (TypeError, ValueError):
                pass

    if 'col_90s' not in df.columns:
        if '90s' in df.columns:
            df['col_90s'] = pd.to_numeric(df['90s'], errors='coerce')
        elif 'min' in df.columns:
            df['col_90s'] = pd.to_numeric(df['min'], errors='coerce') / 90

    if 'col_90s' in df.columns:
        df = df[df['col_90s'] >= 5].copy()

    if 'finishing_alpha' not in df.columns or df['finishing_alpha'].isna().all():
        if 'gls' in df.columns and 'xg' in df.columns:
            df['finishing_alpha'] = df['gls'] - df['xg']

    if 'playmaking_alpha' not in df.columns or df['playmaking_alpha'].isna().all():
        if 'ast' in df.columns and 'xag' in df.columns:
            df['playmaking_alpha'] = df['ast'] - df['xag']

    if 'col_90s' in df.columns:
        mask = df['col_90s'] > 0
        for base, per90 in [('gls', 'gls_per90'), ('xg', 'xg_per90'),
                            ('ast', 'ast_per90'), ('xag', 'xag_per90')]:
            if base in df.columns and (per90 not in df.columns or df[per90].isna().all()):
                df.loc[mask, per90] = df.loc[mask, base] / df.loc[mask, 'col_90s']
        if 'finishing_alpha' in df.columns and ('alpha_per90' not in df.columns or df['alpha_per90'].isna().all()):
            df.loc[mask, 'alpha_per90'] = df.loc[mask, 'finishing_alpha'] / df.loc[mask, 'col_90s']

    print(f"  Final dataset: {len(df)} players, {len(df.columns)} columns")
    print(f"  xG coverage: {df['xg'].notna().sum()}/{len(df)} players" if 'xg' in df.columns else "  No xG data")
    return df


def print_analysis(df):
    print(f"Players with 5+ 90s played: {len(df)}")

    if 'finishing_alpha' in df.columns and df['finishing_alpha'].notna().any():
        print("\nTOP 10 CLINICAL FINISHERS")
        print(df.dropna(subset=['finishing_alpha']).nlargest(10, 'finishing_alpha')
              [['player', 'squad', 'comp', 'gls', 'xg', 'finishing_alpha']].to_string(index=False))

        print("\nTOP 10 UNDERPERFORMERS")
        print(df.dropna(subset=['finishing_alpha']).nsmallest(10, 'finishing_alpha')
              [['player', 'squad', 'comp', 'gls', 'xg', 'finishing_alpha']].to_string(index=False))

    if 'playmaking_alpha' in df.columns and df['playmaking_alpha'].notna().any():
        print("\nTOP 10 CREATIVE OVERPERFORMERS")
        print(df.dropna(subset=['playmaking_alpha']).nlargest(10, 'playmaking_alpha')
              [['player', 'squad', 'comp', 'ast', 'xag', 'playmaking_alpha']].to_string(index=False))

    print("\nData: FBref (basic stats) + Understat (xG/xA)")
    return df


if __name__ == "__main__":
    df = get_data()
    print_analysis(df)