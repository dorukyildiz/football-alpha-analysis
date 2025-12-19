import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from analysis import get_data

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'rankings'

# Position-based weights for scoring
FW_WEIGHTS = {
    'gls': 0.25,
    'ast': 0.10,
    'xg': 0.15,
    'xag': 0.05,
    'finishing_alpha': 0.20,
    'playmaking_alpha': 0.05,
    'gls_per90': 0.10,
    'ast_per90': 0.05,
    'col_90s': 0.05
}

MF_WEIGHTS = {
    'gls': 0.10,
    'ast': 0.15,
    'xg': 0.05,
    'xag': 0.10,
    'finishing_alpha': 0.10,
    'playmaking_alpha': 0.15,
    'gls_per90': 0.05,
    'ast_per90': 0.10,
    'col_90s': 0.10,
    'kp': 0.10
}

DF_WEIGHTS = {
    'gls': 0.05,
    'ast': 0.05,
    'tkl': 0.15,
    'int': 0.15,
    'clr': 0.10,
    'blocks': 0.10,
    'col_90s': 0.15,
    'finishing_alpha': 0.05,
    'playmaking_alpha': 0.05,
    'recov': 0.15
}

GK_WEIGHTS = {
    'saves': 0.25,
    'cs': 0.25,
    'ga': -0.20,
    'col_90s': 0.15,
    'save': 0.15
}

LEAGUES = [
    'Premier League',
    'La Liga',
    'Serie A',
    'Bundesliga',
    'Ligue 1'
]


def get_primary_position(pos):
    """Extract primary position"""
    if pd.isna(pos):
        return None
    pos = pos.upper().replace('"', '').strip()
    if ',' in pos:
        return pos.split(',')[0].strip()
    return pos


def calculate_position_score(df, position, league=None):
    """Calculate score for players of a specific position"""

    if position == 'FW':
        weights = FW_WEIGHTS
        pos_df = df[df['pos'].str.upper().str.contains('FW', na=False)].copy()
    elif position == 'MF':
        weights = MF_WEIGHTS
        pos_df = df[df['pos'].str.upper().str.contains('MF', na=False)].copy()
    elif position == 'DF':
        weights = DF_WEIGHTS
        pos_df = df[df['pos'].str.upper().str.contains('DF', na=False)].copy()
    elif position == 'GK':
        weights = GK_WEIGHTS
        pos_df = df[df['pos'].str.upper().str.contains('GK', na=False)].copy()
    else:
        return None

    # Filter by league if specified
    if league:
        pos_df = pos_df[pos_df['comp'].str.contains(league, case=False, na=False)].copy()

    if len(pos_df) == 0:
        return None

    # Calculate percentiles for each metric
    score = pd.Series(0.0, index=pos_df.index)

    for metric, weight in weights.items():
        if metric in pos_df.columns:
            if weight < 0:
                percentile = (1 - pos_df[metric].rank(pct=True)) * 100
                score += percentile * abs(weight)
            else:
                percentile = pos_df[metric].rank(pct=True) * 100
                score += percentile * weight

    # Convert to 60-95 scale (FIFA style)
    pos_df['raw_score'] = score
    pos_df['rating'] = 60 + (score * 0.35)
    pos_df['rating'] = pos_df['rating'].clip(60, 95).round(1)

    return pos_df


def get_grade(rating):
    """Get grade based on rating"""
    if rating >= 90:
        return "WORLD CLASS"
    elif rating >= 85:
        return "EXCELLENT"
    elif rating >= 80:
        return "VERY GOOD"
    elif rating >= 75:
        return "GOOD"
    elif rating >= 70:
        return "AVERAGE"
    else:
        return "BELOW AVERAGE"


def print_rankings(pos_df, position, top_n=20, league=None):
    """Print top players for a position"""

    title = f"TOP {top_n} {position}"
    if league:
        title += f" - {league}"

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    top_players = pos_df.nlargest(top_n, 'rating')

    print(f"\n{'Rank':<6} {'Player':<25} {'Team':<20} {'League':<15} {'Rating':<10} {'Grade':<15}")
    print("-" * 100)

    for i, (_, row) in enumerate(top_players.iterrows(), 1):
        grade = get_grade(row['rating'])
        league_short = row['comp'][:12] + '..' if len(row['comp']) > 14 else row['comp']
        print(f"{i:<6} {row['player']:<25} {row['squad']:<20} {league_short:<15} {row['rating']:<10.1f} {grade:<15}")

    return top_players


def create_ranking_chart(pos_df, position, top_n=15, league=None):
    """Create ranking visualization"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    top_players = pos_df.nlargest(top_n, 'rating')

    fig, ax = plt.subplots(figsize=(12, 10))

    colors = []
    for rating in top_players['rating']:
        if rating >= 90:
            colors.append('#27ae60')
        elif rating >= 85:
            colors.append('#2ecc71')
        elif rating >= 80:
            colors.append('#3498db')
        elif rating >= 75:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')

    bars = ax.barh(top_players['player'], top_players['rating'], color=colors)

    for bar, rating in zip(bars, top_players['rating']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{rating:.1f}', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Rating')
    title = f'Top {top_n} {position} Rankings (2025-26)'
    if league:
        title += f' - {league}'
    ax.set_title(title)
    ax.set_xlim(60, 100)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', label='World Class (90+)'),
        Patch(facecolor='#2ecc71', label='Excellent (85-89)'),
        Patch(facecolor='#3498db', label='Very Good (80-84)'),
        Patch(facecolor='#f39c12', label='Good (75-79)'),
        Patch(facecolor='#e74c3c', label='Average (<75)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    filename = f"top_{position.lower()}"
    if league:
        filename += f"_{league.lower().replace(' ', '_')}"
    filename += "_rankings.png"

    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

    print(f"[OK] Chart saved: {filename}")


def create_comparison_chart(all_rankings, league=None):
    """Create comparison of top players across positions"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    positions = ['FW', 'MF', 'DF', 'GK']

    for ax, position in zip(axes, positions):
        if position in all_rankings and all_rankings[position] is not None:
            top5 = all_rankings[position].nlargest(5, 'rating')

            colors = []
            for rating in top5['rating']:
                if rating >= 90:
                    colors.append('#27ae60')
                elif rating >= 85:
                    colors.append('#2ecc71')
                elif rating >= 80:
                    colors.append('#3498db')
                else:
                    colors.append('#f39c12')

            bars = ax.barh(top5['player'], top5['rating'], color=colors)

            for bar, rating in zip(bars, top5['rating']):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        f'{rating:.1f}', va='center', fontsize=9)

            ax.set_xlim(70, 100)
            ax.set_title(f'Top 5 {position}', fontsize=12, fontweight='bold')
            ax.invert_yaxis()

    title = 'Top Players by Position (2025-26)'
    if league:
        title += f' - {league}'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = "all_positions_comparison"
    if league:
        filename += f"_{league.lower().replace(' ', '_')}"
    filename += ".png"

    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

    print(f"[OK] Chart saved: {filename}")


def create_league_comparison_chart(df, position):
    """Create league comparison for a position"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, league in enumerate(LEAGUES):
        ax = axes[i]
        pos_df = calculate_position_score(df, position, league)

        if pos_df is not None and len(pos_df) > 0:
            top5 = pos_df.nlargest(5, 'rating')

            colors = []
            for rating in top5['rating']:
                if rating >= 90:
                    colors.append('#27ae60')
                elif rating >= 85:
                    colors.append('#2ecc71')
                elif rating >= 80:
                    colors.append('#3498db')
                else:
                    colors.append('#f39c12')

            bars = ax.barh(top5['player'], top5['rating'], color=colors)

            for bar, rating in zip(bars, top5['rating']):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        f'{rating:.1f}', va='center', fontsize=8)

            ax.set_xlim(70, 100)
            ax.set_title(league, fontsize=11, fontweight='bold')
            ax.invert_yaxis()
        else:
            ax.set_title(f"{league} (No data)")

    # Hide last subplot if empty
    axes[5].axis('off')

    plt.suptitle(f'Top 5 {position} by League (2025-26)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{position.lower()}_by_league.png", dpi=150)
    plt.close()

    print(f"[OK] Chart saved: {position.lower()}_by_league.png")


def create_overall_best_xi(df):
    """Create best XI visualization"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_xi = {}

    # Get best at each position
    for position in ['GK', 'DF', 'MF', 'FW']:
        pos_df = calculate_position_score(df, position)
        if pos_df is not None:
            best = pos_df.nlargest(1, 'rating').iloc[0]
            best_xi[position] = {
                'player': best['player'],
                'squad': best['squad'],
                'rating': best['rating']
            }

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Formation positions (4-3-3 style layout)
    positions_coords = {
        'GK': (0.5, 0.1),
        'DF': (0.5, 0.3),
        'MF': (0.5, 0.55),
        'FW': (0.5, 0.8)
    }

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw field
    ax.add_patch(plt.Rectangle((0.05, 0.02), 0.9, 0.96, fill=False, edgecolor='green', linewidth=2))
    ax.axhline(y=0.5, xmin=0.05, xmax=0.95, color='green', linewidth=1)

    for position, coords in positions_coords.items():
        if position in best_xi:
            player = best_xi[position]

            # Color based on rating
            if player['rating'] >= 90:
                color = '#27ae60'
            elif player['rating'] >= 85:
                color = '#2ecc71'
            else:
                color = '#3498db'

            circle = plt.Circle(coords, 0.06, color=color, ec='white', linewidth=2)
            ax.add_patch(circle)

            ax.text(coords[0], coords[1], f"{player['rating']:.0f}",
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')
            ax.text(coords[0], coords[1] - 0.1, player['player'],
                    ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(coords[0], coords[1] - 0.14, player['squad'],
                    ha='center', va='center', fontsize=8, color='gray')

    ax.set_title('Best XI by Position (2025-26)', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_xi.png", dpi=150)
    plt.close()

    print(f"[OK] Chart saved: best_xi.png")


def export_rankings_csv(all_rankings):
    """Export rankings to CSV"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for position, pos_df in all_rankings.items():
        if pos_df is not None:
            export_df = pos_df[['player', 'squad', 'comp', 'pos', 'age', 'rating']].copy()
            export_df['grade'] = export_df['rating'].apply(get_grade)
            export_df = export_df.sort_values('rating', ascending=False)
            export_df.to_csv(OUTPUT_DIR / f"{position.lower()}_rankings.csv", index=False)

    print(f"[OK] CSV files saved to: {OUTPUT_DIR}")


def main():
    print("Loading data...")
    df = get_data()
    print(f"Loaded {len(df)} players")

    # Calculate rankings for all positions (global)
    all_rankings = {}

    for position in ['FW', 'MF', 'DF', 'GK']:
        print(f"\nCalculating {position} rankings...")
        pos_df = calculate_position_score(df, position)
        if pos_df is not None:
            all_rankings[position] = pos_df
            print(f"  Found {len(pos_df)} players")

    print("\nReady!\n")

    while True:
        print("=" * 70)
        print("Commands:")
        print("  fw / mf / df / gk      - Show position rankings (global)")
        print("  fw pl / mf laliga etc  - Show position rankings by league")
        print("  leagues fw             - Compare position across all leagues")
        print("  all                    - Show all rankings + comparison")
        print("  bestxi                 - Create Best XI")
        print("  export                 - Export all rankings to CSV")
        print("  quit                   - Exit")
        print("")
        print("  Leagues: pl (Premier League), laliga, seriea, bundesliga, ligue1")
        print("=" * 70)

        cmd = input("Enter command: ").strip().lower()

        if cmd == 'quit':
            break

        # Parse command
        parts = cmd.split()

        if len(parts) == 0:
            continue

        position_cmd = parts[0]
        league_filter = parts[1] if len(parts) > 1 else None

        # Map league shortcuts
        league_map = {
            'pl': 'Premier League',
            'premierleague': 'Premier League',
            'laliga': 'La Liga',
            'seriea': 'Serie A',
            'bundesliga': 'Bundesliga',
            'ligue1': 'Ligue 1'
        }

        if league_filter:
            league_filter = league_map.get(league_filter, league_filter)

        if position_cmd == 'fw':
            pos_df = calculate_position_score(df, 'FW', league_filter)
            if pos_df is not None:
                print_rankings(pos_df, 'FW', league=league_filter)
                create_ranking_chart(pos_df, 'FW', league=league_filter)

        elif position_cmd == 'mf':
            pos_df = calculate_position_score(df, 'MF', league_filter)
            if pos_df is not None:
                print_rankings(pos_df, 'MF', league=league_filter)
                create_ranking_chart(pos_df, 'MF', league=league_filter)

        elif position_cmd == 'df':
            pos_df = calculate_position_score(df, 'DF', league_filter)
            if pos_df is not None:
                print_rankings(pos_df, 'DF', league=league_filter)
                create_ranking_chart(pos_df, 'DF', league=league_filter)

        elif position_cmd == 'gk':
            pos_df = calculate_position_score(df, 'GK', league_filter)
            if pos_df is not None:
                print_rankings(pos_df, 'GK', league=league_filter)
                create_ranking_chart(pos_df, 'GK', league=league_filter)

        elif position_cmd == 'leagues':
            if len(parts) > 1:
                pos = parts[1].upper()
                if pos in ['FW', 'MF', 'DF', 'GK']:
                    create_league_comparison_chart(df, pos)
                    for league in LEAGUES:
                        pos_df = calculate_position_score(df, pos, league)
                        if pos_df is not None:
                            print_rankings(pos_df, pos, top_n=5, league=league)

        elif position_cmd == 'all':
            for position in ['FW', 'MF', 'DF', 'GK']:
                if position in all_rankings:
                    print_rankings(all_rankings[position], position, top_n=10)
                    create_ranking_chart(all_rankings[position], position)
            create_comparison_chart(all_rankings)
            create_overall_best_xi(df)

        elif position_cmd == 'bestxi':
            create_overall_best_xi(df)
            print("\nBest XI:")
            for position in ['GK', 'DF', 'MF', 'FW']:
                pos_df = calculate_position_score(df, position)
                if pos_df is not None:
                    best = pos_df.nlargest(1, 'rating').iloc[0]
                    print(f"  {position}: {best['player']} ({best['squad']}) - {best['rating']:.1f}")

        elif position_cmd == 'export':
            export_rankings_csv(all_rankings)

        else:
            print("[ERROR] Unknown command")


if __name__ == "__main__":
    main()