import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from analysis import get_data

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'comparisons'

# Post-Opta category metrics (Standard + Shooting + Misc + Understat)
FW_METRICS = {
    'attacking': ['gls', 'ast', 'g_a', 'xg', 'xag', 'npxg', 'g_pk'],
    'shooting': ['sh', 'sot', 'g_sh', 'g_sot'],
    'alpha': ['finishing_alpha', 'playmaking_alpha'],
    'misc': ['fld', 'crs', 'pkwon'],
}

MF_METRICS = {
    'attacking': ['gls', 'ast', 'g_a', 'xg', 'xag', 'npxg', 'g_pk'],
    'shooting': ['sh', 'sot', 'g_sh'],
    'defensive': ['tklw', 'int', 'fls', 'fld'],
    'alpha': ['finishing_alpha', 'playmaking_alpha'],
    'discipline': ['crdy', 'crdr'],
}

DF_METRICS = {
    'defensive': ['tklw', 'int', 'fls', 'fld'],
    'attacking': ['gls', 'ast'],
    'alpha': ['finishing_alpha', 'playmaking_alpha'],
    'discipline': ['crdy', 'crdr', 'crs'],
}

GK_METRICS = {
    'goalkeeping': ['ga', 'ga90', 'saves', 'savepct', 'cs', 'cspct', 'pka', 'pksv'],
}

ALL_OUTFIELD_METRICS = {
    'attacking': ['gls', 'ast', 'g_a', 'xg', 'xag', 'npxg', 'g_pk'],
    'shooting': ['sh', 'sot', 'g_sh', 'g_sot'],
    'defensive': ['tklw', 'int', 'fls', 'fld'],
    'alpha': ['finishing_alpha', 'playmaking_alpha'],
    'discipline': ['crdy', 'crdr'],
    'misc': ['crs', 'pkwon', 'pkcon', 'off'],
}


def find_player(df, player_name):
    """Find player by name"""
    player_idx = df[df['player'].str.lower() == player_name.lower()].index

    if len(player_idx) == 0:
        player_idx = df[df['player'].str.lower().str.contains(player_name.lower())].index
        if len(player_idx) == 0:
            return None, f"Player '{player_name}' not found"
        elif len(player_idx) > 1:
            matches = df.loc[player_idx, 'player'].tolist()
            return None, f"Multiple matches: {matches}"

    return df.loc[player_idx[0]], None


def get_primary_position(pos):
    """Extract primary position"""
    if pd.isna(pos):
        return None
    pos = pos.upper().replace('"', '').strip()
    if ',' in pos:
        return pos.split(',')[0].strip()
    return pos


def is_goalkeeper(pos):
    """Check if player is goalkeeper"""
    if pd.isna(pos):
        return False
    return 'GK' in pos.upper()


def get_comparison_metrics(pos1, pos2):
    """Get metrics based on positions"""
    primary1 = get_primary_position(pos1)
    primary2 = get_primary_position(pos2)

    is_gk1 = is_goalkeeper(pos1)
    is_gk2 = is_goalkeeper(pos2)

    if is_gk1 and is_gk2:
        return GK_METRICS

    if is_gk1 or is_gk2:
        return None

    if primary1 == primary2:
        if primary1 == 'FW':
            return FW_METRICS
        elif primary1 == 'MF':
            return MF_METRICS
        elif primary1 == 'DF':
            return DF_METRICS

    return ALL_OUTFIELD_METRICS


def compare_players(df, player1_name, player2_name):
    """Compare two players"""

    p1, err1 = find_player(df, player1_name)
    if err1:
        return None, None, None, err1

    p2, err2 = find_player(df, player2_name)
    if err2:
        return None, None, None, err2

    if is_goalkeeper(p1['pos']) != is_goalkeeper(p2['pos']):
        return None, None, None, "Cannot compare goalkeeper with outfield player"

    metrics = get_comparison_metrics(p1['pos'], p2['pos'])

    if metrics is None:
        return None, None, None, "Cannot determine comparison metrics"

    available_metrics = {}
    for category, metric_list in metrics.items():
        available = [m for m in metric_list if m in df.columns]
        if available:
            available_metrics[category] = available

    return p1, p2, available_metrics, None


def print_comparison(p1, p2, metrics_dict, df):
    """Print side-by-side comparison by category"""

    print("\n" + "=" * 80)
    print("PLAYER COMPARISON")
    print("=" * 80)

    print(f"\n{'ATTRIBUTE':<20} {p1['player']:<25} {p2['player']:<25}")
    print("-" * 80)
    print(f"{'Team':<20} {p1['squad']:<25} {p2['squad']:<25}")
    print(f"{'Position':<20} {p1['pos']:<25} {p2['pos']:<25}")
    print(f"{'League':<20} {p1['comp']:<25} {p2['comp']:<25}")
    print(f"{'Age':<20} {str(p1['age']):<25} {str(p2['age']):<25}")

    p1_wins = 0
    p2_wins = 0

    lower_is_better = ['ga', 'ga90', 'fls', 'crdy', 'crdr', 'pkcon', 'og', 'off']

    for category, metrics in metrics_dict.items():
        print("\n" + "-" * 80)
        print(f"{category.upper()}")
        print("-" * 80)

        for m in metrics:
            val1 = p1[m] if m in p1 and pd.notna(p1[m]) else 0
            val2 = p2[m] if m in p2 and pd.notna(p2[m]) else 0

            if m in lower_is_better:
                if val1 < val2:
                    marker1, marker2 = "[+]", ""
                    p1_wins += 1
                elif val2 < val1:
                    marker1, marker2 = "", "[+]"
                    p2_wins += 1
                else:
                    marker1, marker2 = "", ""
            else:
                if val1 > val2:
                    marker1, marker2 = "[+]", ""
                    p1_wins += 1
                elif val2 > val1:
                    marker1, marker2 = "", "[+]"
                    p2_wins += 1
                else:
                    marker1, marker2 = "", ""

            print(f"{m:<20} {val1:<10.1f} {marker1:<5} {val2:<10.1f} {marker2:<5}")

    print("\n" + "=" * 80)
    print(f"RESULT: {p1['player']} wins {p1_wins} | {p2['player']} wins {p2_wins}")
    print("=" * 80)


def create_comparison_chart(p1, p2, metrics_dict, df):
    """Create visual comparison chart by category"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = []
    for category, metrics in metrics_dict.items():
        all_metrics.extend(metrics[:4])

    display_metrics = all_metrics[:12]

    vals1 = [p1[m] if m in p1 and pd.notna(p1[m]) else 0 for m in display_metrics]
    vals2 = [p2[m] if m in p2 and pd.notna(p2[m]) else 0 for m in display_metrics]

    x = np.arange(len(display_metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width / 2, vals1, width, label=p1['player'], color='#3498db')
    bars2 = ax.bar(x + width / 2, vals2, width, label=p2['player'], color='#e74c3c')

    ax.set_ylabel('Value')
    ax.set_title(f"Player Comparison: {p1['player']} vs {p2['player']}")
    ax.set_xticks(x)
    ax.set_xticklabels(display_metrics, rotation=45, ha='right')
    ax.legend()

    for bar, val in zip(bars1, vals1):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    for bar, val in zip(bars2, vals2):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()

    filename = f"bar_{p1['player'].replace(' ', '_')}_vs_{p2['player'].replace(' ', '_')}.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"\n[OK] Bar chart saved: {filepath}")


def create_radar_chart(p1, p2, metrics_dict, df):
    """Create radar chart comparison"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = []
    for category, metrics in metrics_dict.items():
        all_metrics.extend(metrics[:2])

    display_metrics = all_metrics[:8]

    if len(display_metrics) < 3:
        print("[WARN] Not enough metrics for radar chart")
        return

    vals1 = []
    vals2 = []

    for m in display_metrics:
        max_val = df[m].max() if m in df.columns else 1
        if max_val == 0:
            max_val = 1

        v1 = (p1[m] if m in p1 and pd.notna(p1[m]) else 0) / max_val * 100
        v2 = (p2[m] if m in p2 and pd.notna(p2[m]) else 0) / max_val * 100
        vals1.append(v1)
        vals2.append(v2)

    angles = np.linspace(0, 2 * np.pi, len(display_metrics), endpoint=False).tolist()

    vals1 += vals1[:1]
    vals2 += vals2[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.plot(angles, vals1, 'o-', linewidth=2, label=p1['player'], color='#3498db')
    ax.fill(angles, vals1, alpha=0.25, color='#3498db')

    ax.plot(angles, vals2, 'o-', linewidth=2, label=p2['player'], color='#e74c3c')
    ax.fill(angles, vals2, alpha=0.25, color='#e74c3c')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_metrics, size=10)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title(f"Radar: {p1['player']} vs {p2['player']}", size=14, y=1.08)

    plt.tight_layout()

    filename = f"radar_{p1['player'].replace(' ', '_')}_vs_{p2['player'].replace(' ', '_')}.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"[OK] Radar chart saved: {filepath}")


def create_category_charts(p1, p2, metrics_dict, df):
    """Create separate chart for each category"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for category, metrics in metrics_dict.items():
        if not metrics:
            continue

        vals1 = [p1[m] if m in p1 and pd.notna(p1[m]) else 0 for m in metrics]
        vals2 = [p2[m] if m in p2 and pd.notna(p2[m]) else 0 for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 1.5), 6))
        ax.bar(x - width / 2, vals1, width, label=p1['player'], color='#3498db')
        ax.bar(x + width / 2, vals2, width, label=p2['player'], color='#e74c3c')

        ax.set_ylabel('Value')
        ax.set_title(f"{category.upper()}: {p1['player']} vs {p2['player']}")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()

        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()

        filename = f"cat_{category}_{p1['player'].replace(' ', '_')}_vs_{p2['player'].replace(' ', '_')}.png"
        filepath = OUTPUT_DIR / filename
        plt.savefig(filepath, dpi=150)
        plt.close()

    print(f"[OK] Category charts saved to: {OUTPUT_DIR}")


def main():
    print("Loading data...")
    df = get_data()
    print(f"Loaded {len(df)} players")
    print("Ready!\n")

    while True:
        print("=" * 60)
        print("Enter two player names to compare (or 'quit' to exit)")

        player1 = input("Player 1: ").strip()
        if player1.lower() == 'quit':
            break

        player2 = input("Player 2: ").strip()
        if player2.lower() == 'quit':
            break

        p1, p2, metrics, error = compare_players(df, player1, player2)

        if error:
            print(f"[ERROR] {error}")
            continue

        print_comparison(p1, p2, metrics, df)
        create_comparison_chart(p1, p2, metrics, df)
        create_radar_chart(p1, p2, metrics, df)
        create_category_charts(p1, p2, metrics, df)


if __name__ == "__main__":
    main()