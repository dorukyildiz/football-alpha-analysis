import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import os
from analysis import get_data

# PDF imports
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'scout_reports'


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


def get_percentile(df, player, metric, position_filter=True):
    """Calculate player's percentile for a metric"""
    if position_filter and pd.notna(player['pos']):
        pos = player['pos'].upper()
        if 'FW' in pos:
            compare_df = df[df['pos'].str.upper().str.contains('FW', na=False)]
        elif 'MF' in pos:
            compare_df = df[df['pos'].str.upper().str.contains('MF', na=False)]
        elif 'DF' in pos:
            compare_df = df[df['pos'].str.upper().str.contains('DF', na=False)]
        elif 'GK' in pos:
            compare_df = df[df['pos'].str.upper().str.contains('GK', na=False)]
        else:
            compare_df = df
    else:
        compare_df = df

    if metric not in compare_df.columns:
        return None

    val = player[metric] if pd.notna(player[metric]) else 0
    percentile = (compare_df[metric] < val).sum() / len(compare_df) * 100

    return round(percentile, 1)


def get_player_rating(percentiles):
    """Calculate overall rating based on percentiles (FIFA-style 60-95 scale)"""
    if not percentiles:
        return 60
    valid = [p for p in percentiles.values() if p is not None]
    if not valid:
        return 60

    avg_percentile = sum(valid) / len(valid)

    # FIFA-style scale: 60 (worst) to 95 (best)
    rating = 60 + (avg_percentile * 0.35)

    return round(rating, 1)


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


def generate_scout_report(df, player):
    """Generate comprehensive scout report"""

    report = {
        'player': player['player'],
        'team': player['squad'],
        'league': player['comp'],
        'position': player['pos'],
        'age': player['age'],
        'nation': player['nation'] if 'nation' in player else 'N/A',
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }

    # Basic stats
    report['stats'] = {
        'matches': player['mp'] if pd.notna(player['mp']) else 0,
        'minutes': player['min'] if pd.notna(player['min']) else 0,
        '90s': player['col_90s'] if pd.notna(player['col_90s']) else 0,
        'goals': player['gls'] if pd.notna(player['gls']) else 0,
        'assists': player['ast'] if pd.notna(player['ast']) else 0,
        'xg': player['xg'] if pd.notna(player['xg']) else 0,
        'xag': player['xag'] if pd.notna(player['xag']) else 0,
    }

    # Alpha metrics
    report['alpha'] = {
        'finishing_alpha': player['finishing_alpha'] if pd.notna(player['finishing_alpha']) else 0,
        'playmaking_alpha': player['playmaking_alpha'] if pd.notna(player['playmaking_alpha']) else 0,
    }

    # Percentiles
    metrics_to_check = ['gls', 'ast', 'xg', 'xag', 'finishing_alpha', 'playmaking_alpha']
    report['percentiles'] = {}
    for m in metrics_to_check:
        if m in df.columns:
            report['percentiles'][m] = get_percentile(df, player, m)

    # Overall rating
    report['rating'] = get_player_rating(report['percentiles'])

    # Strengths and weaknesses
    report['strengths'] = []
    report['weaknesses'] = []

    for m, p in report['percentiles'].items():
        if p is not None:
            if p >= 80:
                report['strengths'].append(f"{m}: Top {100 - p:.0f}%")
            elif p <= 20:
                report['weaknesses'].append(f"{m}: Bottom {p:.0f}%")

    return report


def print_scout_report(report):
    """Print scout report to console"""

    print("\n" + "=" * 80)
    print("SCOUT REPORT")
    print("=" * 80)

    print(f"\nPlayer: {report['player']}")
    print(f"Team: {report['team']} ({report['league']})")
    print(f"Position: {report['position']}")
    print(f"Age: {report['age']}")
    print(f"Nation: {report['nation']}")
    print(f"Generated: {report['generated']}")

    print("\n" + "-" * 40)
    rating = report['rating']
    grade = get_grade(rating)
    print(f"OVERALL RATING: {rating}/100H ({grade})")
    print("-" * 40)

    print("\n--- SEASON STATISTICS ---")
    stats = report['stats']
    print(f"Matches: {stats['matches']:.0f}")
    print(f"Minutes: {stats['minutes']:.0f}")
    print(f"90s Played: {stats['90s']:.1f}")
    print(f"Goals: {stats['goals']:.0f} (xG: {stats['xg']:.1f})")
    print(f"Assists: {stats['assists']:.0f} (xAG: {stats['xag']:.1f})")

    print("\n--- PERFORMANCE ALPHA ---")
    alpha = report['alpha']
    fa = alpha['finishing_alpha']
    pa = alpha['playmaking_alpha']
    fa_str = f"+{fa:.2f}" if fa > 0 else f"{fa:.2f}"
    pa_str = f"+{pa:.2f}" if pa > 0 else f"{pa:.2f}"
    print(f"Finishing Alpha: {fa_str} ({'Overperforming' if fa > 0 else 'Underperforming'})")
    print(f"Playmaking Alpha: {pa_str} ({'Overperforming' if pa > 0 else 'Underperforming'})")

    print("\n--- PERCENTILE RANKINGS (vs Position) ---")
    for m, p in report['percentiles'].items():
        if p is not None:
            bar = "#" * int(p / 5) + "-" * (20 - int(p / 5))
            print(f"{m:<20} [{bar}] {p:.0f}%")

    if report['strengths']:
        print("\n--- STRENGTHS ---")
        for s in report['strengths']:
            print(f"  [+] {s}")

    if report['weaknesses']:
        print("\n--- AREAS TO IMPROVE ---")
        for w in report['weaknesses']:
            print(f"  [-] {w}")

    print("\n" + "=" * 80)


def create_report_charts(df, player, report):
    """Create visualization for scout report"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    player_name = player['player'].replace(' ', '_')
    chart_paths = {}

    # 1. Percentile radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    metrics = list(report['percentiles'].keys())
    values = [report['percentiles'][m] if report['percentiles'][m] is not None else 0 for m in metrics]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax.fill(angles, values, alpha=0.25, color='#3498db')

    ref_values = [50] * len(metrics) + [50]
    ax.plot(angles, ref_values, '--', linewidth=1, color='gray', alpha=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=10)
    ax.set_ylim(0, 100)
    ax.set_title(f"Percentile Rankings", size=12, y=1.08)

    plt.tight_layout()
    radar_path = OUTPUT_DIR / f"{player_name}_radar.png"
    plt.savefig(radar_path, dpi=150)
    plt.close()
    chart_paths['radar'] = radar_path

    # 2. Goals vs xG comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ['Goals', 'Assists']
    actual = [report['stats']['goals'], report['stats']['assists']]
    expected = [report['stats']['xg'], report['stats']['xag']]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, actual, width, label='Actual', color='#3498db')
    bars2 = ax.bar(x + width / 2, expected, width, label='Expected', color='#e74c3c', alpha=0.7)

    ax.set_ylabel('Count')
    ax.set_title(f"Actual vs Expected")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for bar, val in zip(bars1, actual):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    for bar, val in zip(bars2, expected):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    actual_path = OUTPUT_DIR / f"{player_name}_actual_vs_expected.png"
    plt.savefig(actual_path, dpi=150)
    plt.close()
    chart_paths['actual_vs_expected'] = actual_path

    # 3. League comparison bar
    fig, ax = plt.subplots(figsize=(8, 5))

    league_df = df[df['comp'] == player['comp']]

    metrics_compare = ['gls', 'ast', 'xg', 'xag']
    player_vals = [player[m] if pd.notna(player[m]) else 0 for m in metrics_compare]
    league_avg = [league_df[m].mean() for m in metrics_compare]

    x = np.arange(len(metrics_compare))
    width = 0.35

    bars1 = ax.bar(x - width / 2, player_vals, width, label=player['player'], color='#3498db')
    bars2 = ax.bar(x + width / 2, league_avg, width, label=f"League Avg", color='#95a5a6')

    ax.set_ylabel('Value')
    ax.set_title(f"Player vs League Average")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_compare)
    ax.legend()
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    league_path = OUTPUT_DIR / f"{player_name}_league_comparison.png"
    plt.savefig(league_path, dpi=150)
    plt.close()
    chart_paths['league_comparison'] = league_path

    return chart_paths


def create_pdf_report(report, chart_paths):
    """Create PDF scout report"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    player_name = report['player'].replace(' ', '_')
    pdf_path = OUTPUT_DIR / f"{player_name}_scout_report.pdf"

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            rightMargin=1.5 * cm, leftMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm)

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=20,
        alignment=1  # Center
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15
    )

    normal_style = styles['Normal']

    # Build content
    content = []

    # Title
    content.append(Paragraph("SCOUT REPORT", title_style))
    content.append(Spacer(1, 0.3 * inch))

    # Player info table
    rating = report['rating']
    grade = get_grade(rating)

    info_data = [
        ['Player:', report['player'], 'Rating:', f"{rating}/100 ({grade})"],
        ['Team:', report['team'], 'Position:', report['position']],
        ['League:', report['league'], 'Age:', str(report['age'])],
        ['Nation:', report['nation'], 'Generated:', report['generated']],
    ]

    info_table = Table(info_data, colWidths=[1.5 * inch, 2 * inch, 1.5 * inch, 2 * inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    content.append(info_table)
    content.append(Spacer(1, 0.3 * inch))

    # Season Statistics
    content.append(Paragraph("Season Statistics", heading_style))

    stats = report['stats']
    stats_data = [
        ['Matches', 'Minutes', '90s Played', 'Goals', 'Assists'],
        [f"{stats['matches']:.0f}", f"{stats['minutes']:.0f}", f"{stats['90s']:.1f}",
         f"{stats['goals']:.0f}", f"{stats['assists']:.0f}"],
        ['xG', 'xAG', 'Finishing Alpha', 'Playmaking Alpha', ''],
        [f"{stats['xg']:.1f}", f"{stats['xag']:.1f}",
         f"{report['alpha']['finishing_alpha']:+.2f}",
         f"{report['alpha']['playmaking_alpha']:+.2f}", ''],
    ]

    stats_table = Table(stats_data, colWidths=[1.4 * inch] * 5)
    stats_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 2), (-1, 2), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('BACKGROUND', (0, 2), (-1, 2), colors.lightgrey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    content.append(stats_table)
    content.append(Spacer(1, 0.3 * inch))

    # Percentile Rankings
    content.append(Paragraph("Percentile Rankings (vs Same Position)", heading_style))

    perc_data = [['Metric', 'Percentile', 'Assessment']]
    for m, p in report['percentiles'].items():
        if p is not None:
            if p >= 80:
                assessment = "Excellent"
            elif p >= 60:
                assessment = "Good"
            elif p >= 40:
                assessment = "Average"
            else:
                assessment = "Below Average"
            perc_data.append([m, f"{p:.0f}%", assessment])

    perc_table = Table(perc_data, colWidths=[2 * inch, 1.5 * inch, 2 * inch])
    perc_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    content.append(perc_table)
    content.append(Spacer(1, 0.3 * inch))

    # Strengths and Weaknesses
    if report['strengths']:
        content.append(Paragraph("Strengths", heading_style))
        for s in report['strengths']:
            content.append(Paragraph(f"[+] {s}", normal_style))
        content.append(Spacer(1, 0.2 * inch))

    if report['weaknesses']:
        content.append(Paragraph("Areas to Improve", heading_style))
        for w in report['weaknesses']:
            content.append(Paragraph(f"[-] {w}", normal_style))
        content.append(Spacer(1, 0.2 * inch))

    # Charts
    content.append(Paragraph("Visual Analysis", heading_style))

    # Radar chart
    if 'radar' in chart_paths:
        img = Image(str(chart_paths['radar']), width=4 * inch, height=4 * inch)
        content.append(img)
        content.append(Spacer(1, 0.2 * inch))

    # Actual vs Expected
    if 'actual_vs_expected' in chart_paths:
        img = Image(str(chart_paths['actual_vs_expected']), width=5 * inch, height=3 * inch)
        content.append(img)
        content.append(Spacer(1, 0.2 * inch))

    # League comparison
    if 'league_comparison' in chart_paths:
        img = Image(str(chart_paths['league_comparison']), width=5 * inch, height=3 * inch)
        content.append(img)

    # Build PDF
    doc.build(content)

    print(f"[OK] PDF report saved: {pdf_path}")
    return pdf_path


def main():
    print("Loading data...")
    df = get_data()
    print(f"Loaded {len(df)} players")
    print("Ready!\n")

    while True:
        print("=" * 60)
        player_name = input("Enter player name for scout report (or 'quit'): ").strip()

        if player_name.lower() == 'quit':
            break

        player, error = find_player(df, player_name)

        if error:
            print(f"[ERROR] {error}")
            continue

        report = generate_scout_report(df, player)
        print_scout_report(report)
        chart_paths = create_report_charts(df, player, report)
        create_pdf_report(report, chart_paths)


if __name__ == "__main__":
    main()