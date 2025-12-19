import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os
import io
from adjustText import adjust_text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Streamlit Cloud secrets for AWS
try:
    if 'aws' in st.secrets:
        os.environ['AWS_ACCESS_KEY_ID'] = st.secrets['aws']['AWS_ACCESS_KEY_ID']
        os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets['aws']['AWS_SECRET_ACCESS_KEY']
        os.environ['AWS_DEFAULT_REGION'] = st.secrets['aws']['AWS_DEFAULT_REGION']
except:
    pass

sys.path.append(str(Path(__file__).parent))
from analysis import get_data

# Page config
st.set_page_config(
    page_title="Football Alpha Analysis",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 20px 0;
        border-bottom: 2px solid rgba(255,255,255,0.1);
        margin-bottom: 20px;
    }
    .sidebar-subtitle {
        font-size: 12px;
        color: #888;
        text-align: center;
        margin-top: -15px;
        margin-bottom: 20px;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Position metrics
FW_METRICS = ['gls', 'ast', 'g_a', 'xg', 'xag', 'npxg', 'g_pk', 'kp', 'xa', 'ppa', 'touches', 'carries', 'prgr', 'mis', 'pkwon']
MF_METRICS = ['gls', 'ast', 'g_a', 'xg', 'xag', 'npxg', 'g_pk', 'tkl', 'tklw', 'int', 'tkl_int', 'prgp', 'prgc', 'kp', 'xa', 'ppa', 'touches', 'carries', 'prgr', 'mis', 'dis', 'crdy', 'crdr', 'recov']
DF_METRICS = ['tkl', 'tklw', 'blocks', 'int', 'tkl_int', 'clr', 'err', 'prgp', 'prgc', 'touches', 'carries', 'mis', 'dis', 'crdy', 'crdr', 'recov']
GK_METRICS = ['ga', 'saves', 'save', 'cs', 'pka', 'pksv']

@st.cache_data
def load_data():
    return get_data()

def get_primary_position(pos):
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

def get_grade(rating):
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

def get_metrics_for_position(pos):
    primary = get_primary_position(pos)
    if primary == 'GK':
        return GK_METRICS
    elif primary == 'DF':
        return DF_METRICS
    elif primary == 'MF':
        return MF_METRICS
    else:
        return FW_METRICS


def find_similar_players(df, player_name, top_n=5):
    """Find similar players"""
    player_row = df[df['player'] == player_name]
    if len(player_row) == 0:
        return None, None

    player = player_row.iloc[0]
    pos = player['pos']
    metrics = get_metrics_for_position(pos)

    # Filter same position
    primary_pos = get_primary_position(pos)
    if primary_pos == 'GK':
        filtered_df = df[df['pos'].str.contains('GK', na=False, case=False)].copy()
    else:
        filtered_df = df[~df['pos'].str.contains('GK', na=False, case=False)].copy()

    available_metrics = [m for m in metrics if m in filtered_df.columns]
    if not available_metrics:
        return None, None

    metrics_df = filtered_df[available_metrics].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(metrics_df)

    similarity_matrix = cosine_similarity(scaled)
    player_idx = filtered_df.index.get_loc(player_row.index[0])
    similarities = similarity_matrix[player_idx]

    similar_indices = np.argsort(similarities)[::-1][1:top_n + 1]

    results = []
    for idx in similar_indices:
        similar_player = filtered_df.iloc[idx]
        results.append({
            'player': similar_player['player'],
            'squad': similar_player['squad'],
            'pos': similar_player['pos'],
            'comp': similar_player['comp'],
            'similarity': round(similarities[idx] * 100, 1)
        })

    return results, available_metrics


def generate_scout_report_pdf(player, df):
    """Generate PDF scout report"""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    elements = []
    styles = getSampleStyleSheet()

    # Get position-specific data
    pos = get_primary_position(player['pos'])
    if pos == 'GK':
        pos_df = df[df['pos'].str.contains('GK', na=False, case=False)]
    else:
        pos_df = df[~df['pos'].str.contains('GK', na=False, case=False)]

    # Calculate percentiles
    metrics = ['gls', 'ast', 'xg', 'xag', 'finishing_alpha', 'playmaking_alpha']
    percentiles = {}
    for m in metrics:
        if m in player and m in pos_df.columns:
            percentiles[m] = (pos_df[m] < player[m]).mean() * 100

    # Calculate overall rating
    avg_pct = np.mean(list(percentiles.values()))
    rating = round(60 + avg_pct * 0.35, 1)
    grade = get_grade(rating)

    # Title style
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, spaceAfter=5,
                                 textColor=colors.darkblue)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=12, textColor=colors.grey)
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=14, spaceBefore=15, spaceAfter=10,
                                   textColor=colors.darkblue)

    # Header
    elements.append(Paragraph("SCOUT REPORT", title_style))
    elements.append(Spacer(1, 5))

    # Player info box
    info_data = [
        [f"Player: {player['player']}", f"Rating: {rating}/100 ({grade})"],
        [f"Team: {player['squad']}", f"Position: {player['pos']}"],
        [f"League: {player['comp']}", f"Age: {int(player['age'])}"],
        [f"Nation: {player.get('nation', 'N/A')}", f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"]
    ]
    info_table = Table(info_data, colWidths=[250, 250])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.Color(0.95, 0.95, 0.95)),
        ('BOX', (0, 0), (-1, -1), 1, colors.darkblue),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 15))

    # Season Statistics
    elements.append(Paragraph("Season Statistics", section_style))
    stats_data = [
        ['Matches', 'Minutes', '90s Played', 'Goals', 'Assists'],
        [str(int(player.get('mp', 0))), str(int(player.get('min', 0))), f"{player['col_90s']:.1f}",
         str(int(player['gls'])), str(int(player['ast']))]
    ]
    stats_table = Table(stats_data, colWidths=[100, 100, 100, 100, 100])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 10))

    # Expected stats
    exp_data = [
        ['xG', 'xAG', 'Finishing Alpha', 'Playmaking Alpha'],
        [f"{player['xg']:.1f}", f"{player['xag']:.1f}", f"{player['finishing_alpha']:+.2f}",
         f"{player['playmaking_alpha']:+.2f}"]
    ]
    exp_table = Table(exp_data, colWidths=[125, 125, 125, 125])
    exp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(exp_table)
    elements.append(Spacer(1, 15))

    # Percentile Rankings
    elements.append(Paragraph("Percentile Rankings (vs Same Position)", section_style))
    pct_data = [['Metric', 'Percentile', 'Assessment']]
    for m in metrics:
        if m in percentiles:
            pct = percentiles[m]
            if pct >= 80:
                assessment = "Excellent"
            elif pct >= 60:
                assessment = "Good"
            elif pct >= 40:
                assessment = "Average"
            else:
                assessment = "Below Average"
            pct_data.append([m, f"{pct:.0f}%", assessment])

    pct_table = Table(pct_data, colWidths=[166, 166, 168])
    pct_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(pct_table)
    elements.append(Spacer(1, 15))

    # Strengths and Weaknesses
    strengths = [m for m, p in percentiles.items() if p >= 80]
    weaknesses = [m for m, p in percentiles.items() if p < 20]

    if strengths:
        elements.append(Paragraph("Key Strengths", section_style))
        for s in strengths:
            elements.append(Paragraph(f"[+] {s}: Top {100 - percentiles[s]:.0f}%", styles['Normal']))

    if weaknesses:
        elements.append(Paragraph("Areas to Improve", section_style))
        for w in weaknesses:
            elements.append(Paragraph(f"[-] {w}: Bottom {percentiles[w]:.0f}%", styles['Normal']))

    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Visual Analysis", section_style))

    # Create charts and save to temp files
    temp_files = []

    # 1. Radar Chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    pct_vals = [percentiles.get(m, 0) for m in metrics]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    pct_vals += pct_vals[:1]
    angles += angles[:1]
    ax.plot(angles, pct_vals, 'o-', linewidth=2, color='#3498db')
    ax.fill(angles, pct_vals, alpha=0.25, color='#3498db')
    ax.plot(angles, [50] * len(angles), '--', color='gray', alpha=0.5, linewidth=1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_title('Percentile Rankings')
    plt.tight_layout()

    radar_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(radar_file.name, dpi=100, bbox_inches='tight')
    plt.close()
    temp_files.append(radar_file.name)
    elements.append(Image(radar_file.name, width=4 * inch, height=4 * inch))
    elements.append(Spacer(1, 10))

    # 2. Actual vs Expected
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, [player['gls'], player['ast']], width, label='Actual', color='#3498db')
    ax.bar(x + width / 2, [player['xg'], player['xag']], width, label='Expected', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(['Goals', 'Assists'])
    ax.legend()
    ax.set_title('Actual vs Expected')
    ax.set_ylabel('Count')
    for i, (actual, expected) in enumerate([(player['gls'], player['xg']), (player['ast'], player['xag'])]):
        ax.text(i - width / 2, actual + 0.1, f'{int(actual)}', ha='center', fontsize=9)
        ax.text(i + width / 2, expected + 0.1, f'{expected:.1f}', ha='center', fontsize=9)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()

    actual_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(actual_file.name, dpi=100, bbox_inches='tight')
    plt.close()
    temp_files.append(actual_file.name)
    elements.append(Image(actual_file.name, width=5 * inch, height=3.5 * inch))
    elements.append(Spacer(1, 10))

    # 3. League Comparison
    league_df = df[df['comp'] == player['comp']]
    league_avg = {m: league_df[m].mean() for m in ['gls', 'ast', 'xg', 'xag']}

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(4)
    width = 0.35
    player_vals = [player['gls'], player['ast'], player['xg'], player['xag']]
    league_vals = [league_avg['gls'], league_avg['ast'], league_avg['xg'], league_avg['xag']]
    ax.bar(x - width / 2, player_vals, width, label=player['player'], color='#3498db')
    ax.bar(x + width / 2, league_vals, width, label='League Avg', color='#95a5a6')
    ax.set_xticks(x)
    ax.set_xticklabels(['gls', 'ast', 'xg', 'xag'])
    ax.legend()
    ax.set_title('Player vs League Average')
    ax.set_ylabel('Value')
    plt.tight_layout()

    league_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(league_file.name, dpi=100, bbox_inches='tight')
    plt.close()
    temp_files.append(league_file.name)
    elements.append(Image(league_file.name, width=5 * inch, height=3.5 * inch))

    # Build PDF
    doc.build(elements)

    # Cleanup temp files
    for f in temp_files:
        try:
            os.unlink(f)
        except:
            pass

    buffer.seek(0)
    return buffer

def main():
    df = load_data()

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Football Alpha</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">2025-26 Season Analysis</div>', unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Overview", "Player Search", "Scout Report", "Player Comparison", "Team Analysis", "Rankings",
             "Transfer Finder"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### Quick Stats")
        st.markdown(f"**Players:** {len(df):,}")
        st.markdown(f"**Goals:** {int(df['gls'].sum()):,}")
        st.markdown(f"**Assists:** {int(df['ast'].sum()):,}")
        st.markdown("---")

        st.markdown("""
        <div style="text-align: center; padding: 20px 0; color: #888; font-size: 12px;">
            <p>Built by</p>
            <p style="font-size: 16px; font-weight: bold; color: #e94560;">
                <a href="https://github.com/dorukyildiz" target="_blank" style="color: #e94560; text-decoration: none;">dorukyildiz</a>
            </p>
            <p style="margin-top: 10px;">
                <a href="https://github.com/dorukyildiz/football-alpha-analysis" target="_blank" style="color: #888; text-decoration: none;">GitHub Repo</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.title("Football Alpha Analysis Dashboard")
    st.markdown("**Top 5 European Leagues | Data-Driven Insights**")

    # OVERVIEW PAGE
    if page == "Overview":
        st.header("Season Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Players", f"{len(df):,}")
        col2.metric("Total Goals", f"{int(df['gls'].sum()):,}")
        col3.metric("Total Assists", f"{int(df['ast'].sum()):,}")
        col4.metric("Avg Age", f"{df['age'].mean():.1f}")

        st.markdown("---")
        st.subheader("Top Performers")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 10 Scorers**")
            st.dataframe(df.nlargest(10, 'gls')[['player', 'squad', 'gls', 'xg', 'finishing_alpha']], hide_index=True,
                         use_container_width=True)
        with col2:
            st.markdown("**Top 10 Assisters**")
            st.dataframe(df.nlargest(10, 'ast')[['player', 'squad', 'ast', 'xag', 'playmaking_alpha']], hide_index=True,
                         use_container_width=True)

        # xG vs Goals
        st.markdown("---")
        st.subheader("Expected Goals vs Actual Goals")
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(df['xg'], df['gls'], alpha=0.5, c=df['finishing_alpha'], cmap='RdYlGn', s=50)
        plt.colorbar(scatter, label='Finishing Alpha')
        ax.plot([0, df['xg'].max()], [0, df['xg'].max()], 'k--', label='Perfect conversion')
        top_outliers = df.nlargest(10, 'finishing_alpha')
        worst_outliers = df.nsmallest(10, 'finishing_alpha')
        outliers = pd.concat([top_outliers, worst_outliers, df.nlargest(5, 'gls')]).drop_duplicates(subset='player')
        texts = [ax.text(row['xg'], row['gls'], row['player'], fontsize=8, fontweight='bold') for _, row in
                 outliers.iterrows()]
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        ax.set_xlabel('Expected Goals (xG)')
        ax.set_ylabel('Actual Goals')
        ax.legend()
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # xAG vs Assists
        st.markdown("---")
        st.subheader("Expected Assists vs Actual Assists")
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(df['xag'], df['ast'], alpha=0.5, c=df['playmaking_alpha'], cmap='RdYlGn', s=50)
        plt.colorbar(scatter, label='Playmaking Alpha')
        ax.plot([0, df['xag'].max()], [0, df['xag'].max()], 'k--', label='Perfect conversion')
        outliers = pd.concat([df.nlargest(10, 'playmaking_alpha'), df.nsmallest(10, 'playmaking_alpha'),
                              df.nlargest(5, 'ast')]).drop_duplicates(subset='player')
        texts = [ax.text(row['xag'], row['ast'], row['player'], fontsize=8, fontweight='bold') for _, row in
                 outliers.iterrows()]
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        ax.set_xlabel('Expected Assists (xAG)')
        ax.set_ylabel('Actual Assists')
        ax.legend()
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Finishers
        st.markdown("---")
        st.subheader("Finishers")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 15 Clinical Finishers**")
            fig, ax = plt.subplots(figsize=(10, 8))
            top15 = df.nlargest(15, 'finishing_alpha')
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top15['finishing_alpha']]
            ax.barh(top15['player'], top15['finishing_alpha'], color=colors)
            ax.set_xlabel('Finishing Alpha')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            st.markdown("**Top 15 Underperforming**")
            fig, ax = plt.subplots(figsize=(10, 8))
            worst15 = df.nsmallest(15, 'finishing_alpha')
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in worst15['finishing_alpha']]
            ax.barh(worst15['player'], worst15['finishing_alpha'], color=colors)
            ax.set_xlabel('Finishing Alpha')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # League + Team Comparison
        st.markdown("---")
        st.subheader("League Comparison")
        league_stats = df.groupby('comp').agg({'finishing_alpha': 'mean', 'playmaking_alpha': 'mean'}).round(2)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for i, (metric, title) in enumerate([('finishing_alpha', 'Finishing'), ('playmaking_alpha', 'Playmaking')]):
            order = league_stats.sort_values(metric, ascending=True).index
            colors = ['#2ecc71' if league_stats.loc[l, metric] > 0 else '#e74c3c' for l in order]
            axes[i].barh(order, league_stats.loc[order, metric], color=colors)
            axes[i].set_xlabel(f'Average {title} Alpha')
            axes[i].axvline(x=0, color='black', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Team Efficiency
        st.markdown("---")
        st.subheader("Team Efficiency")
        team_stats = df.groupby('squad').agg({'finishing_alpha': 'mean', 'player': 'count'}).rename(
            columns={'player': 'n'})
        team_stats = team_stats[team_stats['n'] >= 5]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Most Clinical Teams**")
            fig, ax = plt.subplots(figsize=(10, 8))
            top_teams = team_stats.nlargest(15, 'finishing_alpha')
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_teams['finishing_alpha']]
            ax.barh(top_teams.index, top_teams['finishing_alpha'], color=colors)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            st.markdown("**Least Clinical Teams**")
            fig, ax = plt.subplots(figsize=(10, 8))
            worst_teams = team_stats.nsmallest(15, 'finishing_alpha')
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in worst_teams['finishing_alpha']]
            ax.barh(worst_teams.index, worst_teams['finishing_alpha'], color=colors)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # PLAYER SEARCH PAGE
    elif page == "Player Search":
        st.header("Player Search")
        players = sorted(df['player'].tolist())
        selected_player = st.selectbox("Search player", options=players, index=None, placeholder="Type player name...")

        if selected_player:
            player = df[df['player'] == selected_player].iloc[0]

            st.markdown("---")
            st.subheader(f"{player['player']} - {player['squad']}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Position", player['pos'])
            col2.metric("Age", int(player['age']))
            col3.metric("League", player['comp'])
            col4.metric("90s Played", f"{player['col_90s']:.1f}")

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Goals", int(player['gls']))
            col2.metric("Assists", int(player['ast']))
            col3.metric("G+A", int(player['gls'] + player['ast']))

            col1, col2, col3 = st.columns(3)
            col1.metric("xG", f"{player['xg']:.1f}")
            col2.metric("xAG", f"{player['xag']:.1f}")
            col3.metric("xG+xAG", f"{player['xg'] + player['xag']:.1f}")

            col1, col2 = st.columns(2)
            col1.metric("Finishing Alpha", f"{player['finishing_alpha']:+.2f}")
            col2.metric("Playmaking Alpha", f"{player['playmaking_alpha']:+.2f}")

            # SIMILAR PLAYERS
            st.markdown("---")
            st.subheader("Similar Players")
            similar, metrics = find_similar_players(df, selected_player, top_n=5)

            if similar:
                similar_df = pd.DataFrame(similar)
                st.dataframe(similar_df, hide_index=True, use_container_width=True)

                # Radar chart
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                radar_metrics = ['gls', 'ast', 'xg', 'xag', 'finishing_alpha', 'playmaking_alpha']
                available = [m for m in radar_metrics if m in df.columns]
                max_vals = {m: df[m].max() for m in available}

                angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
                angles += angles[:1]

                # Original player
                vals = [(player[m] / max_vals[m] * 100) if max_vals[m] != 0 else 0 for m in available]
                vals += vals[:1]
                ax.plot(angles, vals, 'o-', linewidth=2, label=player['player'], color='#3498db')
                ax.fill(angles, vals, alpha=0.25, color='#3498db')

                # Top similar
                if similar:
                    sim_player = df[df['player'] == similar[0]['player']].iloc[0]
                    vals2 = [(sim_player[m] / max_vals[m] * 100) if max_vals[m] != 0 else 0 for m in available]
                    vals2 += vals2[:1]
                    ax.plot(angles, vals2, 'o-', linewidth=2,
                            label=f"{similar[0]['player']} ({similar[0]['similarity']}%)", color='#e74c3c')
                    ax.fill(angles, vals2, alpha=0.25, color='#e74c3c')

                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(available)
                ax.legend(loc='upper right')
                st.pyplot(fig)
                plt.close()

    # SCOUT REPORT PAGE
    elif page == "Scout Report":
        st.header("Scout Report")
        players = sorted(df['player'].tolist())
        selected_player = st.selectbox("Select player", options=players, index=None, placeholder="Type player name...")

        if selected_player:
            player = df[df['player'] == selected_player].iloc[0]

            st.markdown("---")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(f"{player['player']}")
                st.markdown(f"**Team:** {player['squad']} | **League:** {player['comp']}")
                st.markdown(f"**Position:** {player['pos']} | **Age:** {int(player['age'])}")

            with col2:
                # PDF Download
                pdf_buffer = generate_scout_report_pdf(player, df)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"scout_report_{player['player'].replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )

            st.markdown("---")
            st.subheader("Season Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Goals", int(player['gls']))
            col2.metric("xG", f"{player['xg']:.1f}")
            col3.metric("Assists", int(player['ast']))
            col4.metric("xAG", f"{player['xag']:.1f}")

            col1, col2 = st.columns(2)
            fa = player['finishing_alpha']
            pa = player['playmaking_alpha']

            col1.metric("Finishing Alpha", f"{fa:+.2f}")
            if fa >= 0:
                col1.markdown('<p style="color: #2ecc71; margin-top: -15px;">↑ Overperforming</p>',
                              unsafe_allow_html=True)
            else:
                col1.markdown('<p style="color: #e74c3c; margin-top: -15px;">↓ Underperforming</p>',
                              unsafe_allow_html=True)

            col2.metric("Playmaking Alpha", f"{pa:+.2f}")
            if pa >= 0:
                col2.markdown('<p style="color: #2ecc71; margin-top: -15px;">↑ Overperforming</p>',
                              unsafe_allow_html=True)
            else:
                col2.markdown('<p style="color: #e74c3c; margin-top: -15px;">↓ Underperforming</p>',
                              unsafe_allow_html=True)

            # Percentile Rankings
            st.markdown("---")
            st.subheader("Percentile Rankings")
            pos = get_primary_position(player['pos'])
            pos_df = df[df['pos'].str.contains('GK', na=False, case=False)] if pos == 'GK' else df[
                ~df['pos'].str.contains('GK', na=False, case=False)]

            ranking_data = []
            for metric in ['gls', 'ast', 'xg', 'xag', 'finishing_alpha', 'playmaking_alpha']:
                val = player[metric]
                pct = (pos_df[metric] < val).mean() * 100
                rating = round(60 + pct * 0.35, 1)
                ranking_data.append(
                    {'Metric': metric, 'Value': f"{val:.1f}", 'Percentile': f"{pct:.0f}%", 'Rating': rating,
                     'Grade': get_grade(rating)})

            st.dataframe(pd.DataFrame(ranking_data), hide_index=True, use_container_width=True)

            # Charts
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Actual vs Expected**")
                fig, ax = plt.subplots(figsize=(8, 5))
                x = np.arange(2)
                ax.bar(x - 0.2, [player['gls'], player['ast']], 0.4, label='Actual', color='#3498db')
                ax.bar(x + 0.2, [player['xg'], player['xag']], 0.4, label='Expected', color='#e74c3c')
                ax.set_xticks(x)
                ax.set_xticklabels(['Goals', 'Assists'])
                ax.legend()
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("**Percentile Radar**")
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                metrics = ['gls', 'ast', 'xg', 'xag', 'finishing_alpha', 'playmaking_alpha']
                pcts = [(pos_df[m] < player[m]).mean() * 100 for m in metrics]
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                pcts += pcts[:1]
                angles += angles[:1]
                ax.plot(angles, pcts, 'o-', linewidth=2, color='#3498db')
                ax.fill(angles, pcts, alpha=0.25, color='#3498db')
                ax.plot(angles, [50] * len(angles), '--', color='gray', alpha=0.5)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics)
                ax.set_ylim(0, 100)
                st.pyplot(fig)
                plt.close()

    # PLAYER COMPARISON PAGE
    elif page == "Player Comparison":
        st.header("Player Comparison")
        players = sorted(df['player'].tolist())

        col1, col2 = st.columns(2)
        with col1:
            player1 = st.selectbox("Select Player 1", players, index=None, placeholder="Type player name...")
        with col2:
            player2 = st.selectbox("Select Player 2", players, index=None, placeholder="Type player name...")

        if player1 and player2:
            p1 = df[df['player'] == player1].iloc[0]
            p2 = df[df['player'] == player2].iloc[0]

            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.markdown(f"### {player1}")
                st.markdown(f"**Team:** {p1['squad']}")
                st.markdown(f"**Position:** {p1['pos']}")
            with col3:
                st.markdown(f"### {player2}")
                st.markdown(f"**Team:** {p2['squad']}")
                st.markdown(f"**Position:** {p2['pos']}")

            # Bar Chart
            st.markdown("---")
            st.subheader("Metric Comparison")
            metrics = ['gls', 'ast', 'xg', 'xag', 'finishing_alpha', 'playmaking_alpha']
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(metrics))
            width = 0.35
            vals1 = [p1[m] if pd.notna(p1[m]) else 0 for m in metrics]
            vals2 = [p2[m] if pd.notna(p2[m]) else 0 for m in metrics]
            ax.bar(x - width / 2, vals1, width, label=player1, color='#3498db')
            ax.bar(x + width / 2, vals2, width, label=player2, color='#e74c3c')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Radar Chart
            st.markdown("---")
            st.subheader("Radar Comparison")
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            max_vals = {m: df[m].max() for m in metrics}
            vals1_norm = [(p1[m] / max_vals[m] * 100) if max_vals[m] != 0 else 0 for m in metrics]
            vals2_norm = [(p2[m] / max_vals[m] * 100) if max_vals[m] != 0 else 0 for m in metrics]
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            vals1_norm += vals1_norm[:1]
            vals2_norm += vals2_norm[:1]
            angles += angles[:1]
            ax.plot(angles, vals1_norm, 'o-', linewidth=2, label=player1, color='#3498db')
            ax.fill(angles, vals1_norm, alpha=0.25, color='#3498db')
            ax.plot(angles, vals2_norm, 'o-', linewidth=2, label=player2, color='#e74c3c')
            ax.fill(angles, vals2_norm, alpha=0.25, color='#e74c3c')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.legend(loc='upper right')
            st.pyplot(fig)
            plt.close()

            # Category Charts
            st.markdown("---")
            st.subheader("Category Breakdown")

            categories = {
                'Attacking': ['gls', 'ast', 'g_a', 'xg', 'xag'],
                'Passing': ['prgp', 'prgc', 'kp', 'ppa'],
                'Possession': ['touches', 'carries', 'prgr'],
                'Defensive': ['tkl', 'tklw', 'int', 'blocks']
            }

            for cat_name, cat_metrics in categories.items():
                available = [m for m in cat_metrics if m in df.columns]
                if available:
                    st.markdown(f"**{cat_name}**")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    x = np.arange(len(available))
                    width = 0.35
                    v1 = [p1[m] if m in p1 and pd.notna(p1[m]) else 0 for m in available]
                    v2 = [p2[m] if m in p2 and pd.notna(p2[m]) else 0 for m in available]
                    ax.bar(x - width / 2, v1, width, label=player1, color='#3498db')
                    ax.bar(x + width / 2, v2, width, label=player2, color='#e74c3c')
                    ax.set_xticks(x)
                    ax.set_xticklabels(available, rotation=45, ha='right')
                    ax.legend()
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

    # TEAM ANALYSIS PAGE
    elif page == "Team Analysis":
        st.header("Team Analysis")
        teams = sorted(df['squad'].unique().tolist())
        team = st.selectbox("Select Team", teams, index=None, placeholder="Type team name...")

        if team:
            team_df = df[df['squad'] == team]
            league = team_df['comp'].iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Players", len(team_df))
            col2.metric("Total Goals", int(team_df['gls'].sum()))
            col3.metric("Total Assists", int(team_df['ast'].sum()))
            col4.metric("Avg Age", f"{team_df['age'].mean():.1f}")

            col1, col2 = st.columns(2)
            col1.metric("Team Finishing Alpha", f"{team_df['finishing_alpha'].sum():+.2f}")
            col2.metric("Team Playmaking Alpha", f"{team_df['playmaking_alpha'].sum():+.2f}")

            # Squad Table
            st.markdown("---")
            st.subheader("Squad")
            st.dataframe(team_df[['player', 'pos', 'age', 'gls', 'ast', 'xg', 'finishing_alpha']].sort_values('gls',
                                                                                                              ascending=False),
                         hide_index=True, use_container_width=True)

            # Goals vs xG
            st.markdown("---")
            st.subheader("Goals vs xG")
            fig, ax = plt.subplots(figsize=(12, 8))
            sorted_df = team_df.sort_values('gls', ascending=True)
            y_pos = np.arange(len(sorted_df))
            ax.barh(y_pos, sorted_df['gls'], height=0.4, label='Goals', color='#3498db')
            ax.barh(y_pos + 0.4, sorted_df['xg'], height=0.4, label='xG', color='#e74c3c', alpha=0.7)
            ax.set_yticks(y_pos + 0.2)
            ax.set_yticklabels(sorted_df['player'])
            ax.legend()
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Finishing Alpha
            st.markdown("---")
            st.subheader("Finishing Alpha by Player")
            fig, ax = plt.subplots(figsize=(12, 8))
            sorted_df = team_df.sort_values('finishing_alpha', ascending=True)
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in sorted_df['finishing_alpha']]
            ax.barh(sorted_df['player'], sorted_df['finishing_alpha'], color=colors)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_xlabel('Finishing Alpha')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Squad Composition & Goal Contributions
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Squad Composition")

                def get_pos(p):
                    if pd.isna(p): return 'Unknown'
                    p = p.upper()
                    if 'GK' in p:
                        return 'GK'
                    elif 'DF' in p:
                        return 'DF'
                    elif 'MF' in p:
                        return 'MF'
                    elif 'FW' in p:
                        return 'FW'
                    return 'Unknown'

                pos_counts = team_df['pos'].apply(get_pos).value_counts()
                fig, ax = plt.subplots(figsize=(8, 8))
                colors_pie = {'GK': '#9b59b6', 'DF': '#3498db', 'MF': '#2ecc71', 'FW': '#e74c3c', 'Unknown': '#95a5a6'}
                ax.pie(pos_counts.values, labels=pos_counts.index, autopct='%1.0f%%',
                       colors=[colors_pie.get(p, '#95a5a6') for p in pos_counts.index])
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.subheader("Goal Contributions")
                fig, ax = plt.subplots(figsize=(10, 8))
                sorted_df = team_df.copy()
                sorted_df['g_a'] = sorted_df['gls'] + sorted_df['ast']
                sorted_df = sorted_df.sort_values('g_a', ascending=True)
                ax.barh(sorted_df['player'], sorted_df['gls'], label='Goals', color='#3498db')
                ax.barh(sorted_df['player'], sorted_df['ast'], left=sorted_df['gls'], label='Assists', color='#2ecc71')
                ax.legend()
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # League Comparison
            st.markdown("---")
            st.subheader("League Comparison")
            league_df = df[df['comp'] == league]
            team_avg = {'Goals/Player': team_df['gls'].mean(), 'Assists/Player': team_df['ast'].mean(),
                        'Finishing Alpha': team_df['finishing_alpha'].mean(),
                        'Playmaking Alpha': team_df['playmaking_alpha'].mean()}
            league_avg = {'Goals/Player': league_df['gls'].mean(), 'Assists/Player': league_df['ast'].mean(),
                          'Finishing Alpha': league_df['finishing_alpha'].mean(),
                          'Playmaking Alpha': league_df['playmaking_alpha'].mean()}

            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(team_avg))
            width = 0.35
            ax.bar(x - width / 2, list(team_avg.values()), width, label=team, color='#3498db')
            ax.bar(x + width / 2, list(league_avg.values()), width, label=f'{league} Avg', color='#95a5a6')
            ax.set_xticks(x)
            ax.set_xticklabels(list(team_avg.keys()), rotation=15)
            ax.legend()
            ax.axhline(y=0, color='black', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Top 5 Radar
            st.markdown("---")
            st.subheader("Top 5 Players Radar")
            team_df_copy = team_df.copy()
            team_df_copy['g_a'] = team_df_copy['gls'] + team_df_copy['ast']
            top5 = team_df_copy.nlargest(5, 'g_a')

            if len(top5) >= 3:
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
                radar_metrics = ['gls', 'ast', 'xg', 'xag', 'finishing_alpha', 'playmaking_alpha']
                available = [m for m in radar_metrics if m in top5.columns]
                max_vals = {m: team_df_copy[m].max() for m in available}
                angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
                angles += angles[:1]
                colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

                for i, (_, player) in enumerate(top5.iterrows()):
                    vals = []
                    for m in available:
                        max_val = max_vals[m] if max_vals[m] != 0 else 1
                        if 'alpha' in m:
                            min_val = team_df_copy[m].min()
                            range_val = max_val - min_val if max_val != min_val else 1
                            normalized = (player[m] - min_val) / range_val * 100
                        else:
                            normalized = (player[m] / max_val) * 100
                        vals.append(normalized)
                    vals += vals[:1]
                    ax.plot(angles, vals, 'o-', linewidth=2, label=player['player'], color=colors[i % len(colors)])
                    ax.fill(angles, vals, alpha=0.1, color=colors[i % len(colors)])

                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(available)
                ax.set_ylim(0, 100)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                st.pyplot(fig)
                plt.close()

# RANKINGS PAGE
    elif page == "Rankings":
        st.header("Player Rankings")
        col1, col2 = st.columns(2)
        with col1:
            position = st.selectbox("Select Position", ['FW', 'MF', 'DF', 'GK'])
        with col2:
            league = st.selectbox("Select League", ['All'] + sorted(df['comp'].unique().tolist()))

        if position == 'FW':
            pos_df = df[df['pos'].str.upper().str.contains('FW', na=False)].copy()
        elif position == 'MF':
            pos_df = df[df['pos'].str.upper().str.contains('MF', na=False)].copy()
        elif position == 'DF':
            pos_df = df[df['pos'].str.upper().str.contains('DF', na=False)].copy()
        else:
            pos_df = df[df['pos'].str.upper().str.contains('GK', na=False)].copy()

        if league != 'All':
            pos_df = pos_df[pos_df['comp'] == league]

        if position == 'FW':
            pos_df['score'] = pos_df['gls'].rank(pct=True) * 30 + pos_df['xg'].rank(pct=True) * 20 + pos_df['finishing_alpha'].rank(pct=True) * 30 + pos_df['ast'].rank(pct=True) * 20
        elif position == 'MF':
            pos_df['score'] = pos_df['gls'].rank(pct=True) * 15 + pos_df['ast'].rank(pct=True) * 25 + pos_df['playmaking_alpha'].rank(pct=True) * 30 + pos_df['xag'].rank(pct=True) * 30
        elif position == 'DF':
            pos_df['score'] = pos_df['tkl'].rank(pct=True) * 30 + pos_df['int'].rank(pct=True) * 30 + pos_df['clr'].rank(pct=True) * 20 + pos_df['blocks'].rank(pct=True) * 20
        else:
            pos_df['score'] = pos_df['saves'].rank(pct=True) * 40 + pos_df['cs'].rank(pct=True) * 40 + (1 - pos_df['ga'].rank(pct=True)) * 20

        pos_df['rating'] = (60 + pos_df['score'] * 0.35).clip(60, 95).round(1)
        pos_df['grade'] = pos_df['rating'].apply(get_grade)
        top20 = pos_df.nlargest(20, 'rating')

        st.markdown("---")
        st.dataframe(top20[['player', 'squad', 'comp', 'age', 'rating', 'grade']], hide_index=True, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#27ae60' if r >= 90 else '#2ecc71' if r >= 85 else '#3498db' if r >= 80 else '#f39c12' for r in top20['rating']]
        ax.barh(top20['player'], top20['rating'], color=colors)
        ax.set_xlim(60, 100)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # TRANSFER FINDER PAGE
    elif page == "Transfer Finder":
        st.header("Transfer Finder")
        col1, col2 = st.columns(2)
        with col1:
            teams = sorted(df['squad'].unique().tolist())
            team = st.selectbox("Select Your Team", teams, index=None, placeholder="Type team name...")
        with col2:
            position = st.selectbox("Position Needed", ['FW', 'MF', 'DF', 'GK'])

        col1, col2 = st.columns(2)
        with col1:
            min_age = st.slider("Min Age", 16, 40, 18)
        with col2:
            max_age = st.slider("Max Age", 16, 40, 35)

        exclude_team = st.checkbox("Exclude players from selected team", value=True)

        if team and position:
            if position == 'FW':
                candidates = df[df['pos'].str.upper().str.contains('FW', na=False)].copy()
            elif position == 'MF':
                candidates = df[df['pos'].str.upper().str.contains('MF', na=False)].copy()
            elif position == 'DF':
                candidates = df[df['pos'].str.upper().str.contains('DF', na=False)].copy()
            else:
                candidates = df[df['pos'].str.upper().str.contains('GK', na=False)].copy()

            candidates = candidates[(candidates['age'] >= min_age) & (candidates['age'] <= max_age)]
            if exclude_team:
                candidates = candidates[candidates['squad'] != team]

            if position == 'FW':
                candidates['score'] = candidates['gls'].rank(pct=True) * 30 + candidates['finishing_alpha'].rank(pct=True) * 40 + candidates['xg'].rank(pct=True) * 30
            elif position == 'MF':
                candidates['score'] = candidates['ast'].rank(pct=True) * 30 + candidates['playmaking_alpha'].rank(pct=True) * 40 + candidates['xag'].rank(pct=True) * 30
            elif position == 'DF':
                candidates['score'] = candidates['tkl'].rank(pct=True) * 35 + candidates['int'].rank(pct=True) * 35 + candidates['clr'].rank(pct=True) * 30
            else:
                candidates['score'] = candidates['saves'].rank(pct=True) * 50 + candidates['cs'].rank(pct=True) * 50

            candidates['rating'] = (60 + candidates['score'] * 0.35).clip(60, 95).round(1)
            top10 = candidates.nlargest(10, 'rating')

            st.markdown("---")
            st.dataframe(top10[['player', 'squad', 'comp', 'age', 'gls', 'ast', 'rating']], hide_index=True, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#3498db' if r >= 80 else '#f39c12' for r in top10['rating']]
            ax.barh(top10['player'], top10['rating'], color=colors)
            ax.set_xlim(60, 100)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


if __name__ == "__main__":
    main()