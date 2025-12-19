import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from analysis import get_data

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'teams'


def get_team_list(df):
    """Get list of all teams"""
    return sorted(df['squad'].unique().tolist())


def get_team_players(df, team_name):
    """Get all players from a team"""
    team_df = df[df['squad'].str.lower() == team_name.lower()]

    if len(team_df) == 0:
        team_df = df[df['squad'].str.lower().str.contains(team_name.lower())]

    return team_df


def analyze_team(df, team_name):
    """Comprehensive team analysis"""

    team_df = get_team_players(df, team_name)

    if len(team_df) == 0:
        return None, f"Team '{team_name}' not found"

    team_name_full = team_df['squad'].iloc[0]
    league = team_df['comp'].iloc[0]

    analysis = {
        'team': team_name_full,
        'league': league,
        'total_players': len(team_df),
        'total_goals': team_df['gls'].sum(),
        'total_assists': team_df['ast'].sum(),
        'total_xg': team_df['xg'].sum(),
        'total_xag': team_df['xag'].sum(),
        'avg_age': team_df['age'].mean(),
        'team_finishing_alpha': team_df['gls'].sum() - team_df['xg'].sum(),
        'team_playmaking_alpha': team_df['ast'].sum() - team_df['xag'].sum(),
        'avg_finishing_alpha': team_df['finishing_alpha'].mean(),
        'avg_playmaking_alpha': team_df['playmaking_alpha'].mean(),
    }

    # Top performers
    analysis['top_scorer'] = team_df.nlargest(1, 'gls')[['player', 'gls']].values[0] if len(team_df) > 0 else None
    analysis['top_assister'] = team_df.nlargest(1, 'ast')[['player', 'ast']].values[0] if len(team_df) > 0 else None
    analysis['best_finisher'] = team_df.nlargest(1, 'finishing_alpha')[['player', 'finishing_alpha']].values[0] if len(
        team_df) > 0 else None
    analysis['worst_finisher'] = team_df.nsmallest(1, 'finishing_alpha')[['player', 'finishing_alpha']].values[
        0] if len(team_df) > 0 else None
    analysis['best_playmaker'] = team_df.nlargest(1, 'playmaking_alpha')[['player', 'playmaking_alpha']].values[
        0] if len(team_df) > 0 else None

    return analysis, team_df


def print_team_analysis(analysis, team_df):
    """Print team analysis results"""

    print("\n" + "=" * 80)
    print(f"TEAM ANALYSIS: {analysis['team']}")
    print(f"League: {analysis['league']}")
    print("=" * 80)

    # Overview
    print("\n--- OVERVIEW ---")
    print(f"Total Players (5+ 90s): {analysis['total_players']}")
    print(f"Average Age: {analysis['avg_age']:.1f}")
    print(f"Total Goals: {analysis['total_goals']:.0f}")
    print(f"Total Assists: {analysis['total_assists']:.0f}")
    print(f"Total xG: {analysis['total_xg']:.1f}")
    print(f"Total xAG: {analysis['total_xag']:.1f}")

    # Alpha metrics
    print("\n--- ALPHA METRICS ---")
    print(f"Team Finishing Alpha: {analysis['team_finishing_alpha']:.2f}")
    print(f"Team Playmaking Alpha: {analysis['team_playmaking_alpha']:.2f}")
    print(f"Avg Player Finishing Alpha: {analysis['avg_finishing_alpha']:.2f}")
    print(f"Avg Player Playmaking Alpha: {analysis['avg_playmaking_alpha']:.2f}")

    # Top performers
    print("\n--- TOP PERFORMERS ---")
    if analysis['top_scorer'] is not None:
        print(f"Top Scorer: {analysis['top_scorer'][0]} ({analysis['top_scorer'][1]:.0f} goals)")
    if analysis['top_assister'] is not None:
        print(f"Top Assister: {analysis['top_assister'][0]} ({analysis['top_assister'][1]:.0f} assists)")
    if analysis['best_finisher'] is not None:
        print(f"Best Finisher: {analysis['best_finisher'][0]} ({analysis['best_finisher'][1]:.2f} alpha)")
    if analysis['worst_finisher'] is not None:
        print(f"Worst Finisher: {analysis['worst_finisher'][0]} ({analysis['worst_finisher'][1]:.2f} alpha)")
    if analysis['best_playmaker'] is not None:
        print(f"Best Playmaker: {analysis['best_playmaker'][0]} ({analysis['best_playmaker'][1]:.2f} alpha)")

    # Full squad
    print("\n--- FULL SQUAD ---")
    print(f"{'Player':<25} {'Pos':<10} {'Gls':<6} {'Ast':<6} {'xG':<8} {'F.Alpha':<10}")
    print("-" * 80)

    for _, row in team_df.sort_values('gls', ascending=False).iterrows():
        fa = row['finishing_alpha'] if pd.notna(row['finishing_alpha']) else 0
        print(
            f"{row['player']:<25} {row['pos']:<10} {row['gls']:<6.0f} {row['ast']:<6.0f} {row['xg']:<8.1f} {fa:<10.2f}")


def create_top_players_radar(analysis, team_df):
    """Create radar chart for top 5 players"""

    team_name = analysis['team'].replace(' ', '_')

    # Get top 5 by goals + assists
    team_df_copy = team_df.copy()
    team_df_copy['g_a'] = team_df_copy['gls'] + team_df_copy['ast']
    top5 = team_df_copy.nlargest(5, 'g_a')

    if len(top5) < 3:
        return

    # Metrics for radar
    metrics = ['gls', 'ast', 'xg', 'xag', 'finishing_alpha', 'playmaking_alpha']
    available_metrics = [m for m in metrics if m in top5.columns]

    if len(available_metrics) < 3:
        return

    # Normalize values (0-100)
    max_vals = {m: team_df_copy[m].max() for m in available_metrics}

    angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for i, (_, player) in enumerate(top5.iterrows()):
        vals = []
        for m in available_metrics:
            max_val = max_vals[m] if max_vals[m] != 0 else 1
            # Handle negative values for alpha metrics
            if 'alpha' in m:
                min_val = team_df_copy[m].min()
                range_val = max_val - min_val if max_val != min_val else 1
                normalized = (player[m] - min_val) / range_val * 100
            else:
                normalized = (player[m] / max_val) * 100 if max_val != 0 else 0
            vals.append(normalized)

        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=player['player'], color=colors[i % len(colors)])
        ax.fill(angles, vals, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics, size=10)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title(f"{analysis['team']} - Top 5 Players Comparison", size=14, y=1.08)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{team_name}_top5_radar.png", dpi=150)
    plt.close()


def create_team_charts(analysis, team_df):
    """Create team visualization charts"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    team_name = analysis['team'].replace(' ', '_')

    # 1. Goals vs xG by player
    fig, ax = plt.subplots(figsize=(12, 8))

    sorted_df = team_df.sort_values('gls', ascending=True)
    y_pos = np.arange(len(sorted_df))

    ax.barh(y_pos, sorted_df['gls'], height=0.4, label='Goals', color='#3498db', align='center')
    ax.barh(y_pos + 0.4, sorted_df['xg'], height=0.4, label='xG', color='#e74c3c', alpha=0.7, align='center')

    ax.set_yticks(y_pos + 0.2)
    ax.set_yticklabels(sorted_df['player'])
    ax.set_xlabel('Goals / xG')
    ax.set_title(f"{analysis['team']} - Goals vs Expected Goals")
    ax.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{team_name}_goals_vs_xg.png", dpi=150)
    plt.close()

    # 2. Finishing Alpha by player
    fig, ax = plt.subplots(figsize=(12, 8))

    sorted_df = team_df.sort_values('finishing_alpha', ascending=True)
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in sorted_df['finishing_alpha']]

    ax.barh(sorted_df['player'], sorted_df['finishing_alpha'], color=colors)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Finishing Alpha (Goals - xG)')
    ax.set_title(f"{analysis['team']} - Finishing Alpha by Player")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{team_name}_finishing_alpha.png", dpi=150)
    plt.close()

    # 3. Team composition pie (by position)
    fig, ax = plt.subplots(figsize=(10, 10))

    def get_main_pos(pos):
        if pd.isna(pos):
            return 'Unknown'
        pos = pos.upper()
        if 'GK' in pos:
            return 'GK'
        elif 'DF' in pos:
            return 'DF'
        elif 'MF' in pos:
            return 'MF'
        elif 'FW' in pos:
            return 'FW'
        return 'Unknown'

    team_df_copy = team_df.copy()
    team_df_copy['main_pos'] = team_df_copy['pos'].apply(get_main_pos)
    pos_counts = team_df_copy['main_pos'].value_counts()

    colors_pie = {'GK': '#9b59b6', 'DF': '#3498db', 'MF': '#2ecc71', 'FW': '#e74c3c', 'Unknown': '#95a5a6'}
    pie_colors = [colors_pie.get(p, '#95a5a6') for p in pos_counts.index]

    ax.pie(pos_counts.values, labels=pos_counts.index, autopct='%1.0f%%', colors=pie_colors, startangle=90)
    ax.set_title(f"{analysis['team']} - Squad Composition by Position")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{team_name}_composition.png", dpi=150)
    plt.close()

    # 4. Goals + Assists contribution
    fig, ax = plt.subplots(figsize=(12, 8))

    team_df_copy = team_df.copy()
    team_df_copy['g_a'] = team_df_copy['gls'] + team_df_copy['ast']
    sorted_df = team_df_copy.sort_values('g_a', ascending=True)

    ax.barh(sorted_df['player'], sorted_df['gls'], label='Goals', color='#3498db')
    ax.barh(sorted_df['player'], sorted_df['ast'], left=sorted_df['gls'], label='Assists', color='#2ecc71')

    ax.set_xlabel('Goals + Assists')
    ax.set_title(f"{analysis['team']} - Goal Contributions")
    ax.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{team_name}_contributions.png", dpi=150)
    plt.close()

    # 5. Top 5 players radar chart
    if len(team_df) >= 3:
        create_top_players_radar(analysis, team_df)

    print(f"\n[OK] Team charts saved to: {OUTPUT_DIR}")


def create_league_comparison_chart(analysis, team_avg, league_avg):
    """Create league comparison visualization"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    team_name = analysis['team'].replace(' ', '_')

    metrics = list(team_avg.keys())
    team_vals = list(team_avg.values())
    league_vals = list(league_avg.values())

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width / 2, team_vals, width, label=analysis['team'], color='#3498db')
    bars2 = ax.bar(x + width / 2, league_vals, width, label=f"{analysis['league']} Avg", color='#95a5a6')

    ax.set_ylabel('Value')
    ax.set_title(f"{analysis['team']} vs {analysis['league']} Average")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Value labels
    for bar, val in zip(bars1, team_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    for bar, val in zip(bars2, league_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{team_name}_league_comparison.png", dpi=150)
    plt.close()

    print(f"[OK] League comparison chart saved")


def compare_to_league(df, analysis, team_df):
    """Compare team to league average with visualization"""

    league_df = df[df['comp'] == analysis['league']]

    league_avg = {
        'Goals/Player': league_df['gls'].mean(),
        'Assists/Player': league_df['ast'].mean(),
        'Finishing Alpha': league_df['finishing_alpha'].mean(),
        'Playmaking Alpha': league_df['playmaking_alpha'].mean(),
    }

    team_avg = {
        'Goals/Player': team_df['gls'].mean(),
        'Assists/Player': team_df['ast'].mean(),
        'Finishing Alpha': team_df['finishing_alpha'].mean(),
        'Playmaking Alpha': team_df['playmaking_alpha'].mean(),
    }

    print("\n--- LEAGUE COMPARISON ---")
    print(f"{'Metric':<25} {'Team':<15} {'League Avg':<15} {'Diff':<10}")
    print("-" * 65)

    for key in league_avg:
        diff = team_avg[key] - league_avg[key]
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
        print(f"{key:<25} {team_avg[key]:<15.2f} {league_avg[key]:<15.2f} {diff_str:<10}")

    # Create comparison chart
    create_league_comparison_chart(analysis, team_avg, league_avg)


def main():
    print("Loading data...")
    df = get_data()
    print(f"Loaded {len(df)} players")

    teams = get_team_list(df)
    print(f"Found {len(teams)} teams")
    print("Ready!\n")

    while True:
        print("=" * 60)
        print("Commands:")
        print("  [team name] - Analyze a team")
        print("  list - Show all teams")
        print("  quit - Exit")
        print("=" * 60)

        cmd = input("Enter team name: ").strip()

        if cmd.lower() == 'quit':
            break
        elif cmd.lower() == 'list':
            print("\nAvailable teams:")
            for i, team in enumerate(teams, 1):
                print(f"  {i}. {team}")
            continue

        analysis, result = analyze_team(df, cmd)

        if analysis is None:
            print(f"[ERROR] {result}")
            continue

        team_df = result
        print_team_analysis(analysis, team_df)
        compare_to_league(df, analysis, team_df)
        create_team_charts(analysis, team_df)


if __name__ == "__main__":
    main()