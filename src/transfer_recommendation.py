import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from analysis import get_data
from similarity import calculate_similarity, FW_METRICS, MF_METRICS, DF_METRICS, GK_METRICS

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'transfers'

# Position metrics for scoring
POSITION_METRICS = {
    'FW': ['gls', 'ast', 'xg', 'xag', 'finishing_alpha'],
    'MF': ['gls', 'ast', 'xg', 'xag', 'kp', 'prgp', 'prgc'],
    'DF': ['tkl', 'int', 'clr', 'blocks', 'recov'],
    'GK': ['saves', 'cs', 'save']
}


def get_team_players(df, team_name):
    """Get all players from a team"""
    team_df = df[df['squad'].str.lower() == team_name.lower()]

    if len(team_df) == 0:
        team_df = df[df['squad'].str.lower().str.contains(team_name.lower())]

    return team_df


def get_primary_position(pos):
    """Extract primary position"""
    if pd.isna(pos):
        return None
    pos = pos.upper().replace('"', '').strip()
    if 'GK' in pos:
        return 'GK'
    elif 'DF' in pos:
        return 'DF'
    elif 'MF' in pos:
        return 'MF'
    elif 'FW' in pos:
        return 'FW'
    return None


def analyze_team_needs(df, team_name):
    """Analyze team's weak positions"""

    team_df = get_team_players(df, team_name)

    if len(team_df) == 0:
        return None, None, f"Team '{team_name}' not found"

    team_name_full = team_df['squad'].iloc[0]
    league = team_df['comp'].iloc[0]

    # Count players by position
    position_counts = {'GK': 0, 'DF': 0, 'MF': 0, 'FW': 0}
    position_ratings = {'GK': [], 'DF': [], 'MF': [], 'FW': []}

    for _, row in team_df.iterrows():
        pos = get_primary_position(row['pos'])
        if pos:
            position_counts[pos] += 1

            # Calculate simple rating based on key metrics
            metrics = POSITION_METRICS.get(pos, [])
            rating = 0
            count = 0
            for m in metrics:
                if m in row and pd.notna(row[m]):
                    # Get league percentile
                    league_df = df[df['comp'] == league]
                    if m in league_df.columns:
                        percentile = (league_df[m] < row[m]).sum() / len(league_df) * 100
                        rating += percentile
                        count += 1

            if count > 0:
                position_ratings[pos].append(rating / count)

    # Calculate average rating per position
    avg_ratings = {}
    for pos, ratings in position_ratings.items():
        if ratings:
            avg_ratings[pos] = np.mean(ratings)
        else:
            avg_ratings[pos] = 0

    # Identify weak positions (below 50 percentile or low count)
    needs = []
    for pos in ['FW', 'MF', 'DF', 'GK']:
        rating = avg_ratings.get(pos, 0)
        count = position_counts.get(pos, 0)

        priority = 'LOW'
        reason = ''

        if count == 0:
            priority = 'CRITICAL'
            reason = 'No players'
        elif count <= 2 and pos != 'GK':
            priority = 'HIGH'
            reason = f'Only {count} players'
        elif rating < 40:
            priority = 'HIGH'
            reason = f'Low quality (avg {rating:.0f}%)'
        elif rating < 55:
            priority = 'MEDIUM'
            reason = f'Below average (avg {rating:.0f}%)'

        needs.append({
            'position': pos,
            'count': count,
            'avg_rating': rating,
            'priority': priority,
            'reason': reason
        })

    # Sort by priority
    priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    needs.sort(key=lambda x: priority_order[x['priority']])

    analysis = {
        'team': team_name_full,
        'league': league,
        'total_players': len(team_df),
        'position_counts': position_counts,
        'avg_ratings': avg_ratings,
        'needs': needs
    }

    return analysis, team_df, None


def find_transfer_targets(df, team_df, position, budget_tier='all', top_n=10):
    """Find transfer targets for a position"""

    team_name = team_df['squad'].iloc[0]
    league = team_df['comp'].iloc[0]

    # Filter by position
    if position == 'FW':
        candidates = df[df['pos'].str.upper().str.contains('FW', na=False)].copy()
    elif position == 'MF':
        candidates = df[df['pos'].str.upper().str.contains('MF', na=False)].copy()
    elif position == 'DF':
        candidates = df[df['pos'].str.upper().str.contains('DF', na=False)].copy()
    elif position == 'GK':
        candidates = df[df['pos'].str.upper().str.contains('GK', na=False)].copy()
    else:
        return None

    # Exclude current team players
    candidates = candidates[candidates['squad'] != team_name]

    # Calculate rating for each candidate
    metrics = POSITION_METRICS.get(position, [])

    scores = []
    for idx, row in candidates.iterrows():
        score = 0
        count = 0
        for m in metrics:
            if m in row and pd.notna(row[m]):
                percentile = (candidates[m] < row[m]).sum() / len(candidates) * 100
                score += percentile
                count += 1

        if count > 0:
            scores.append(score / count)
        else:
            scores.append(0)

    candidates['score'] = scores
    candidates['rating'] = 60 + (candidates['score'] * 0.35)
    candidates['rating'] = candidates['rating'].clip(60, 95).round(1)

    # Budget tier filter (based on age as proxy)
    if budget_tier == 'young':
        candidates = candidates[candidates['age'] <= 23]
    elif budget_tier == 'prime':
        candidates = candidates[(candidates['age'] > 23) & (candidates['age'] <= 29)]
    elif budget_tier == 'experienced':
        candidates = candidates[candidates['age'] > 29]

    # Get top candidates
    top_candidates = candidates.nlargest(top_n, 'rating')

    return top_candidates


def print_team_analysis(analysis):
    """Print team needs analysis"""

    print("\n" + "=" * 80)
    print(f"TRANSFER NEEDS ANALYSIS: {analysis['team']}")
    print(f"League: {analysis['league']}")
    print("=" * 80)

    print(f"\nTotal Players (5+ 90s): {analysis['total_players']}")

    print("\n--- SQUAD COMPOSITION ---")
    print(f"{'Position':<12} {'Count':<10} {'Avg Rating':<15}")
    print("-" * 40)
    for pos in ['GK', 'DF', 'MF', 'FW']:
        count = analysis['position_counts'][pos]
        rating = analysis['avg_ratings'][pos]
        print(f"{pos:<12} {count:<10} {rating:.1f}%")

    print("\n--- TRANSFER PRIORITIES ---")
    print(f"{'Position':<12} {'Priority':<12} {'Reason':<30}")
    print("-" * 60)
    for need in analysis['needs']:
        if need['priority'] != 'LOW':
            print(f"{need['position']:<12} {need['priority']:<12} {need['reason']:<30}")


def print_transfer_targets(targets, position):
    """Print transfer targets"""

    print(f"\n--- TOP {len(targets)} {position} TARGETS ---")
    print(f"{'Rank':<6} {'Player':<25} {'Team':<20} {'Age':<6} {'Rating':<10}")
    print("-" * 75)

    for i, (_, row) in enumerate(targets.iterrows(), 1):
        print(f"{i:<6} {row['player']:<25} {row['squad']:<20} {row['age']:<6.0f} {row['rating']:<10.1f}")


def create_transfer_chart(targets, position, team_name):
    """Create transfer targets visualization"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = []
    for rating in targets['rating']:
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

    bars = ax.barh(targets['player'], targets['rating'], color=colors)

    for bar, (_, row) in zip(bars, targets.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{row['rating']:.1f} ({row['squad']})", va='center', fontsize=8)

    ax.set_xlabel('Rating')
    ax.set_title(f"Transfer Targets for {team_name} - {position}")
    ax.set_xlim(60, 100)
    ax.invert_yaxis()

    plt.tight_layout()

    filename = f"{team_name.replace(' ', '_')}_{position}_targets.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

    print(f"[OK] Chart saved: {filename}")


def create_team_needs_chart(analysis):
    """Create team needs visualization"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Squad composition
    positions = ['GK', 'DF', 'MF', 'FW']
    counts = [analysis['position_counts'][p] for p in positions]
    colors = ['#9b59b6', '#3498db', '#2ecc71', '#e74c3c']

    ax1.bar(positions, counts, color=colors)
    ax1.set_ylabel('Number of Players')
    ax1.set_title('Squad Composition by Position')
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for i, (pos, count) in enumerate(zip(positions, counts)):
        ax1.text(i, count + 0.1, str(count), ha='center', fontweight='bold')

    # Position ratings
    ratings = [analysis['avg_ratings'][p] for p in positions]

    bar_colors = []
    for r in ratings:
        if r >= 60:
            bar_colors.append('#2ecc71')
        elif r >= 45:
            bar_colors.append('#f39c12')
        else:
            bar_colors.append('#e74c3c')

    ax2.bar(positions, ratings, color=bar_colors)
    ax2.set_ylabel('Average Percentile Rating')
    ax2.set_title('Position Quality (League Percentile)')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='League Average')
    ax2.set_ylim(0, 100)
    ax2.legend()

    for i, (pos, rating) in enumerate(zip(positions, ratings)):
        ax2.text(i, rating + 2, f'{rating:.0f}%', ha='center', fontweight='bold')

    plt.suptitle(f"{analysis['team']} - Squad Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f"{analysis['team'].replace(' ', '_')}_squad_analysis.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

    print(f"[OK] Chart saved: {filename}")


def generate_transfer_report(df, team_name):
    """Generate full transfer report"""

    analysis, team_df, error = analyze_team_needs(df, team_name)

    if error:
        return error

    print_team_analysis(analysis)
    create_team_needs_chart(analysis)

    # Find targets for high priority positions
    for need in analysis['needs']:
        if need['priority'] in ['CRITICAL', 'HIGH', 'MEDIUM']:
            print(f"\nFinding {need['position']} targets...")
            targets = find_transfer_targets(df, team_df, need['position'])
            if targets is not None and len(targets) > 0:
                print_transfer_targets(targets, need['position'])
                create_transfer_chart(targets, need['position'], analysis['team'])

    return None


def main():
    print("Loading data...")
    df = get_data()
    print(f"Loaded {len(df)} players")

    # Get team list
    teams = sorted(df['squad'].unique().tolist())
    print(f"Found {len(teams)} teams")
    print("Ready!\n")

    while True:
        print("=" * 70)
        print("Commands:")
        print("  [team name]           - Full transfer analysis")
        print("  [team] [position]     - Specific position targets")
        print("  [team] [pos] young    - Young targets (<=23)")
        print("  [team] [pos] prime    - Prime age targets (24-29)")
        print("  list                  - Show all teams")
        print("  quit                  - Exit")
        print("=" * 70)

        cmd = input("Enter command: ").strip()

        if cmd.lower() == 'quit':
            break
        elif cmd.lower() == 'list':
            print("\nAvailable teams:")
            for i, team in enumerate(teams, 1):
                print(f"  {i}. {team}")
            continue

        parts = cmd.split()

        if len(parts) == 0:
            continue

        # Parse command
        if len(parts) >= 2 and parts[-1].upper() in ['FW', 'MF', 'DF', 'GK']:
            # Specific position search
            position = parts[-1].upper()
            team_name = ' '.join(parts[:-1])
            budget = 'all'

            analysis, team_df, error = analyze_team_needs(df, team_name)
            if error:
                print(f"[ERROR] {error}")
                continue

            targets = find_transfer_targets(df, team_df, position, budget)
            if targets is not None and len(targets) > 0:
                print_transfer_targets(targets, position)
                create_transfer_chart(targets, position, analysis['team'])

        elif len(parts) >= 3 and parts[-2].upper() in ['FW', 'MF', 'DF', 'GK']:
            # Position + budget tier
            budget = parts[-1].lower()
            position = parts[-2].upper()
            team_name = ' '.join(parts[:-2])

            analysis, team_df, error = analyze_team_needs(df, team_name)
            if error:
                print(f"[ERROR] {error}")
                continue

            targets = find_transfer_targets(df, team_df, position, budget)
            if targets is not None and len(targets) > 0:
                print_transfer_targets(targets, position)
                create_transfer_chart(targets, position, analysis['team'])

        else:
            # Full team analysis
            team_name = cmd
            error = generate_transfer_report(df, team_name)
            if error:
                print(f"[ERROR] {error}")


if __name__ == "__main__":
    main()