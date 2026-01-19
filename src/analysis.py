import pandas as pd
from pyathena import connect
import streamlit as st


@st.cache_data(ttl=3600)
def get_data():
    """Fetch data from Athena and calculate alpha metrics"""

    conn = connect(
        s3_staging_dir="s3://football-alpha-athena-results-doruk/",
        region_name="eu-central-1"
    )

    query = """
    SELECT player, nation, pos, squad, comp, age, 
           gls, ast, g_a, xg, xag, npxg, g_pk,
           col_90s, sh, sot, mp, starts, min,
           tkl, tklw, blocks, int, tkl_int, clr, err,
           prgp, prgc, prgr_stats_possession as prgr,
           kp, xa, ppa,
           touches, carries, mis, dis,
           crdy, crdr, recov,
           pkwon, pkcon,
           ga, saves, save, cs, pka, pksv
    FROM football_analytics.players
    WHERE col_90s >= 5
    """
    df = pd.read_sql(query, conn)

    # Alpha calculations
    df['finishing_alpha'] = df['gls'] - df['xg']
    df['playmaking_alpha'] = df['ast'] - df['xag']

    # Per 90 metrics
    df['gls_per90'] = df['gls'] / df['col_90s']
    df['xg_per90'] = df['xg'] / df['col_90s']
    df['ast_per90'] = df['ast'] / df['col_90s']
    df['xag_per90'] = df['xag'] / df['col_90s']
    df['alpha_per90'] = df['finishing_alpha'] / df['col_90s']

    return df


def print_analysis(df):
    """Print analysis results"""

    print(f"Players with 5+ 90s played: {len(df)}")

    print("\n" + "=" * 60)
    print("TOP 10 CLINICAL FINISHERS (Highest Finishing Alpha)")
    print("=" * 60)
    top_finishers = df.nlargest(10, 'finishing_alpha')[['player', 'squad', 'comp', 'gls', 'xg', 'finishing_alpha']]
    print(top_finishers.to_string(index=False))

    print("\n" + "=" * 60)
    print("TOP 10 UNDERPERFORMERS (Lowest Finishing Alpha)")
    print("=" * 60)
    worst_finishers = df.nsmallest(10, 'finishing_alpha')[['player', 'squad', 'comp', 'gls', 'xg', 'finishing_alpha']]
    print(worst_finishers.to_string(index=False))

    print("\n" + "=" * 60)
    print("TOP 10 CREATIVE OVERPERFORMERS (Highest Playmaking Alpha)")
    print("=" * 60)
    top_playmakers = df.nlargest(10, 'playmaking_alpha')[['player', 'squad', 'comp', 'ast', 'xag', 'playmaking_alpha']]
    print(top_playmakers.to_string(index=False))

    print("\n" + "=" * 60)
    print("LEAGUE COMPARISON")
    print("=" * 60)
    league_stats = df.groupby('comp').agg({
        'finishing_alpha': 'mean',
        'playmaking_alpha': 'mean',
        'gls_per90': 'mean',
        'player': 'count'
    }).rename(columns={'player': 'num_players'}).round(3)
    print(league_stats.sort_values('finishing_alpha', ascending=False))

    return league_stats


if __name__ == "__main__":
    df = get_data()
    print_analysis(df)
    print("\n[OK] Analysis complete!")