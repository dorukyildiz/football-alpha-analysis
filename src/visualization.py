import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
from adjustText import adjust_text
from analysis import get_data

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'charts'


def create_visualizations(df):
    """Create and save all charts"""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_size = (12, 6)

    # League stats for chart 4
    league_stats = df.groupby('comp').agg({
        'finishing_alpha': 'mean',
        'playmaking_alpha': 'mean',
        'gls_per90': 'mean',
        'player': 'count'
    }).rename(columns={'player': 'num_players'}).round(3)

    # 1. Top 15 Clinical Finishers
    plt.figure(figsize=fig_size)
    top15 = df.nlargest(15, 'finishing_alpha')
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top15['finishing_alpha']]
    plt.barh(top15['player'], top15['finishing_alpha'], color=colors)
    plt.xlabel('Finishing Alpha (Goals - xG)')
    plt.ylabel('Player')
    plt.title('Top 15 Clinical Finishers (2025-26 Season)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_top_finishers.png', dpi=150)
    plt.close()
    print("✅ Saved: 01_top_finishers.png")

    # 2. Worst 15 Finishers
    plt.figure(figsize=fig_size)
    worst15 = df.nsmallest(15, 'finishing_alpha')
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in worst15['finishing_alpha']]
    plt.barh(worst15['player'], worst15['finishing_alpha'], color=colors)
    plt.xlabel('Finishing Alpha (Goals - xG)')
    plt.ylabel('Player')
    plt.title('Top 15 Underperforming Finishers (2025-26 Season)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_worst_finishers.png', dpi=150)
    plt.close()
    print("✅ Saved: 02_worst_finishers.png")

    # 3. xG vs Actual Goals Scatter with player labels
    plt.figure(figsize=(14, 12))
    plt.scatter(df['xg'], df['gls'], alpha=0.5, c=df['finishing_alpha'], cmap='RdYlGn', s=50)
    plt.colorbar(label='Finishing Alpha')
    plt.plot([0, df['xg'].max()], [0, df['xg'].max()], 'k--', label='Perfect conversion (Goals = xG)')


    top_outliers = df.nlargest(10, 'finishing_alpha')
    worst_outliers = df.nsmallest(10, 'finishing_alpha')
    top_scorers = df.nlargest(5, 'gls')
    outliers = pd.concat([top_outliers, worst_outliers, top_scorers]).drop_duplicates(subset='player')

    texts = []
    for _, row in outliers.iterrows():
        texts.append(plt.text(row['xg'], row['gls'], row['player'], fontsize=8, fontweight='bold'))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.xlabel('Expected Goals (xG)')
    plt.ylabel('Actual Goals')
    plt.title('Expected Goals vs Actual Goals (2025-26 Season)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_xg_vs_goals.png', dpi=150)
    plt.close()
    print("✅ Saved: 03_xg_vs_goals.png")

    # 4. League Comparison
    plt.figure(figsize=fig_size)
    league_order = league_stats.sort_values('finishing_alpha', ascending=True).index
    colors = ['#2ecc71' if league_stats.loc[l, 'finishing_alpha'] > 0 else '#e74c3c' for l in league_order]
    plt.barh(league_order, league_stats.loc[league_order, 'finishing_alpha'], color=colors)
    plt.xlabel('Average Finishing Alpha')
    plt.ylabel('League')
    plt.title('League Comparison: Average Finishing Efficiency')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_league_comparison.png', dpi=150)
    plt.close()
    print("✅ Saved: 04_league_comparison.png")

    # 5. Top 15 Scorers
    plt.figure(figsize=fig_size)
    top_scorers = df.nlargest(15, 'gls')
    colors = ['#3498db' for _ in range(15)]
    bars = plt.barh(top_scorers['player'], top_scorers['gls'], color=colors)

    # Add xG as overlay
    for i, (_, row) in enumerate(top_scorers.iterrows()):
        plt.plot(row['xg'], i, 'r|', markersize=20, markeredgewidth=3, label='xG' if i == 0 else '')

    plt.xlabel('Goals (bars) vs xG (red line)')
    plt.ylabel('Player')
    plt.title('Top 15 Goal Scorers - Goals vs Expected Goals')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_top_scorers.png', dpi=150)
    plt.close()
    print("✅ Saved: 05_top_scorers.png")

    # 6. Assists vs xAG Scatter (Playmaking Analysis)
    plt.figure(figsize=(14, 12))
    plt.scatter(df['xag'], df['ast'], alpha=0.5, c=df['playmaking_alpha'], cmap='RdYlGn', s=50)
    plt.colorbar(label='Playmaking Alpha')
    plt.plot([0, df['xag'].max()], [0, df['xag'].max()], 'k--', label='Perfect conversion (Ast = xAG)')


    top_playmakers = df.nlargest(10, 'playmaking_alpha')
    worst_playmakers = df.nsmallest(10, 'playmaking_alpha')
    top_assisters = df.nlargest(5, 'ast')
    outliers = pd.concat([top_playmakers, worst_playmakers, top_assisters]).drop_duplicates(subset='player')

    texts = []
    for _, row in outliers.iterrows():
        texts.append(plt.text(row['xag'], row['ast'], row['player'], fontsize=8, fontweight='bold'))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.xlabel('Expected Assists (xAG)')
    plt.ylabel('Actual Assists')
    plt.title('Expected Assists vs Actual Assists (2025-26 Season)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_xag_vs_assists.png', dpi=150)
    plt.close()
    print("✅ Saved: 06_xag_vs_assists.png")

    # 7. Shot Conversion Analysis
    plt.figure(figsize=fig_size)
    shooters = df[df['sh'] >= 10].copy()
    shooters['shot_accuracy'] = (shooters['sot'] / shooters['sh']) * 100
    shooters['conversion_rate'] = (shooters['gls'] / shooters['sh']) * 100

    plt.scatter(shooters['shot_accuracy'], shooters['conversion_rate'],
                alpha=0.6, c=shooters['gls'], cmap='YlOrRd', s=60)
    plt.colorbar(label='Total Goals')

    top_converters = shooters.nlargest(5, 'conversion_rate')
    for _, row in top_converters.iterrows():
        plt.annotate(row['player'], xy=(row['shot_accuracy'], row['conversion_rate']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

    plt.xlabel('Shot Accuracy (% on target)')
    plt.ylabel('Conversion Rate (% goals per shot)')
    plt.title('Shot Accuracy vs Goal Conversion Rate')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_shot_conversion.png', dpi=150)
    plt.close()
    print("✅ Saved: 07_shot_conversion.png")

    # 8. Team Efficiency (Top 20 teams by finishing alpha)
    plt.figure(figsize=(12, 8))
    team_stats = df.groupby('squad').agg({
        'finishing_alpha': 'mean',
        'gls': 'sum',
        'xg': 'sum',
        'player': 'count'
    }).rename(columns={'player': 'num_players'})
    team_stats = team_stats[team_stats['num_players'] >= 5]
    team_stats = team_stats.sort_values('finishing_alpha', ascending=True).tail(20)

    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in team_stats['finishing_alpha']]
    plt.barh(team_stats.index, team_stats['finishing_alpha'], color=colors)
    plt.xlabel('Average Finishing Alpha')
    plt.ylabel('Team')
    plt.title('Top 20 Most Clinical Teams (2025-26 Season)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_team_efficiency.png', dpi=150)
    plt.close()
    print("✅ Saved: 08_team_efficiency.png")

    # 9. Worst Teams (Bottom 20 by finishing alpha)
    plt.figure(figsize=(12, 8))
    team_stats_all = df.groupby('squad').agg({
        'finishing_alpha': 'mean',
        'gls': 'sum',
        'xg': 'sum',
        'player': 'count'
    }).rename(columns={'player': 'num_players'})
    team_stats_all = team_stats_all[team_stats_all['num_players'] >= 5]
    worst_teams = team_stats_all.sort_values('finishing_alpha', ascending=True).head(20)

    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in worst_teams['finishing_alpha']]
    plt.barh(worst_teams.index, worst_teams['finishing_alpha'], color=colors)
    plt.xlabel('Average Finishing Alpha')
    plt.ylabel('Team')
    plt.title('Top 20 Least Clinical Teams (2025-26 Season)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_worst_teams.png', dpi=150)
    plt.close()
    print("✅ Saved: 09_worst_teams.png")

    # 10. Minutes vs Goals
    plt.figure(figsize=(12, 10))
    plt.scatter(df['min'], df['gls'], alpha=0.5, c=df['gls_per90'], cmap='YlOrRd', s=50)
    plt.colorbar(label='Goals per 90')

    top_scorers = df.nlargest(10, 'gls')
    texts = []
    for _, row in top_scorers.iterrows():
        texts.append(plt.text(row['min'], row['gls'], row['player'], fontsize=8, fontweight='bold'))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.xlabel('Minutes Played')
    plt.ylabel('Goals')
    plt.title('Minutes Played vs Goals Scored')

    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_minutes_vs_goals.png', dpi=150)
    plt.close()
    print("✅ Saved: 10_minutes_vs_goals.png")

    print(f"\n✅ All visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    print("Fetching data from Athena...")
    df = get_data()
    print(f"Loaded {len(df)} players")
    create_visualizations(df)