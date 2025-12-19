import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from analysis import get_data

# Page config
st.set_page_config(
    page_title="Football Alpha Analysis",
    page_icon="⚽",
    layout="wide"
)


# Cache data loading
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


def main():
    st.title("Football Alpha Analysis Dashboard")
    st.markdown("**2025-26 Season | Top 5 European Leagues**")

    # Load data
    df = load_data()

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Page", [
        "Overview",
        "Player Search",
        "Player Comparison",
        "Team Analysis",
        "Rankings",
        "Transfer Finder"
    ])

    # Overview Page
    if page == "Overview":
        st.header("Season Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Players", len(df))
        col2.metric("Total Goals", int(df['gls'].sum()))
        col3.metric("Total Assists", int(df['ast'].sum()))
        col4.metric("Avg Age", f"{df['age'].mean():.1f}")

        st.subheader("Top Performers")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 10 Scorers**")
            top_scorers = df.nlargest(10, 'gls')[['player', 'squad', 'gls', 'xg', 'finishing_alpha']]
            st.dataframe(top_scorers, hide_index=True)

        with col2:
            st.markdown("**Top 10 Assisters**")
            top_assisters = df.nlargest(10, 'ast')[['player', 'squad', 'ast', 'xag', 'playmaking_alpha']]
            st.dataframe(top_assisters, hide_index=True)

        st.subheader("League Comparison")

        league_stats = df.groupby('comp').agg({
            'gls': 'sum',
            'ast': 'sum',
            'finishing_alpha': 'mean',
            'player': 'count'
        }).rename(columns={'player': 'players'}).round(2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].barh(league_stats.index, league_stats['gls'], color='#3498db')
        axes[0].set_title('Total Goals by League')
        axes[0].set_xlabel('Goals')

        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in league_stats['finishing_alpha']]
        axes[1].barh(league_stats.index, league_stats['finishing_alpha'], color=colors)
        axes[1].set_title('Avg Finishing Alpha by League')
        axes[1].axvline(x=0, color='black', linewidth=0.5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Player Search Page
    elif page == "Player Search":
        st.header("Player Search")

        search = st.text_input("Search player name")

        if search:
            results = df[df['player'].str.lower().str.contains(search.lower())]

            if len(results) == 0:
                st.warning("No players found")
            else:
                st.success(f"Found {len(results)} players")

                for _, player in results.iterrows():
                    with st.expander(f"{player['player']} - {player['squad']}"):
                        col1, col2, col3 = st.columns(3)

                        col1.metric("Goals", int(player['gls']))
                        col2.metric("Assists", int(player['ast']))
                        col3.metric("xG", f"{player['xg']:.1f}")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Finishing Alpha", f"{player['finishing_alpha']:+.2f}")
                        col2.metric("Playmaking Alpha", f"{player['playmaking_alpha']:+.2f}")
                        col3.metric("Position", player['pos'])

    # Player Comparison Page
    elif page == "Player Comparison":
        st.header("Player Comparison")

        players = df['player'].tolist()

        col1, col2 = st.columns(2)

        with col1:
            player1 = st.selectbox("Select Player 1", players, index=0)

        with col2:
            player2 = st.selectbox("Select Player 2", players, index=1)

        if player1 and player2:
            p1 = df[df['player'] == player1].iloc[0]
            p2 = df[df['player'] == player2].iloc[0]

            st.subheader(f"{player1} vs {player2}")

            metrics = ['gls', 'ast', 'xg', 'xag', 'finishing_alpha', 'playmaking_alpha']

            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                st.markdown(f"**{player1}**")
                st.markdown(f"Team: {p1['squad']}")
                st.markdown(f"Position: {p1['pos']}")

            with col3:
                st.markdown(f"**{player2}**")
                st.markdown(f"Team: {p2['squad']}")
                st.markdown(f"Position: {p2['pos']}")

            # Comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(len(metrics))
            width = 0.35

            vals1 = [p1[m] if pd.notna(p1[m]) else 0 for m in metrics]
            vals2 = [p2[m] if pd.notna(p2[m]) else 0 for m in metrics]

            ax.bar(x - width / 2, vals1, width, label=player1, color='#3498db')
            ax.bar(x + width / 2, vals2, width, label=player2, color='#e74c3c')

            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            ax.set_title('Metric Comparison')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Radar chart
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            # Normalize values
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

    # Team Analysis Page
    elif page == "Team Analysis":
        st.header("Team Analysis")

        teams = sorted(df['squad'].unique().tolist())
        team = st.selectbox("Select Team", teams)

        if team:
            team_df = df[df['squad'] == team]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Players", len(team_df))
            col2.metric("Total Goals", int(team_df['gls'].sum()))
            col3.metric("Total Assists", int(team_df['ast'].sum()))
            col4.metric("Avg Age", f"{team_df['age'].mean():.1f}")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Team Finishing Alpha", f"{team_df['finishing_alpha'].sum():+.2f}")
            with col2:
                st.metric("Team Playmaking Alpha", f"{team_df['playmaking_alpha'].sum():+.2f}")

            st.subheader("Squad")
            st.dataframe(
                team_df[['player', 'pos', 'age', 'gls', 'ast', 'xg', 'finishing_alpha']].sort_values('gls',
                                                                                                     ascending=False),
                hide_index=True
            )

            # Goals vs xG chart
            st.subheader("Goals vs xG")

            fig, ax = plt.subplots(figsize=(10, 6))

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

    # Rankings Page
    elif page == "Rankings":
        st.header("Player Rankings")

        position = st.selectbox("Select Position", ['FW', 'MF', 'DF', 'GK'])
        league = st.selectbox("Select League", ['All'] + sorted(df['comp'].unique().tolist()))

        # Filter by position
        if position == 'FW':
            pos_df = df[df['pos'].str.upper().str.contains('FW', na=False)].copy()
        elif position == 'MF':
            pos_df = df[df['pos'].str.upper().str.contains('MF', na=False)].copy()
        elif position == 'DF':
            pos_df = df[df['pos'].str.upper().str.contains('DF', na=False)].copy()
        else:
            pos_df = df[df['pos'].str.upper().str.contains('GK', na=False)].copy()

        # Filter by league
        if league != 'All':
            pos_df = pos_df[pos_df['comp'] == league]

        # Calculate rating
        if position == 'FW':
            pos_df['score'] = pos_df['gls'].rank(pct=True) * 30 + pos_df['xg'].rank(pct=True) * 20 + pos_df[
                'finishing_alpha'].rank(pct=True) * 30 + pos_df['ast'].rank(pct=True) * 20
        elif position == 'MF':
            pos_df['score'] = pos_df['gls'].rank(pct=True) * 15 + pos_df['ast'].rank(pct=True) * 25 + pos_df[
                'playmaking_alpha'].rank(pct=True) * 30 + pos_df['xag'].rank(pct=True) * 30
        elif position == 'DF':
            pos_df['score'] = pos_df['tkl'].rank(pct=True) * 30 + pos_df['int'].rank(pct=True) * 30 + pos_df[
                'clr'].rank(pct=True) * 20 + pos_df['blocks'].rank(pct=True) * 20
        else:
            pos_df['score'] = pos_df['saves'].rank(pct=True) * 40 + pos_df['cs'].rank(pct=True) * 40 + (
                        1 - pos_df['ga'].rank(pct=True)) * 20

        pos_df['rating'] = (60 + pos_df['score'] * 0.35).clip(60, 95).round(1)
        pos_df['grade'] = pos_df['rating'].apply(get_grade)

        top20 = pos_df.nlargest(20, 'rating')

        st.subheader(f"Top 20 {position}")
        st.dataframe(
            top20[['player', 'squad', 'comp', 'age', 'rating', 'grade']],
            hide_index=True
        )

        # Chart
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = []
        for r in top20['rating']:
            if r >= 90:
                colors.append('#27ae60')
            elif r >= 85:
                colors.append('#2ecc71')
            elif r >= 80:
                colors.append('#3498db')
            else:
                colors.append('#f39c12')

        ax.barh(top20['player'], top20['rating'], color=colors)
        ax.set_xlim(60, 100)
        ax.set_xlabel('Rating')
        ax.invert_yaxis()
        ax.set_title(f'Top 20 {position} Rankings')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Transfer Finder Page
    elif page == "Transfer Finder":
        st.header("Transfer Finder")

        teams = sorted(df['squad'].unique().tolist())
        team = st.selectbox("Select Your Team", teams)
        position = st.selectbox("Position Needed", ['FW', 'MF', 'DF', 'GK'])

        col1, col2 = st.columns(2)
        with col1:
            min_age = st.slider("Min Age", 16, 40, 18)
        with col2:
            max_age = st.slider("Max Age", 16, 40, 35)

        exclude_team = st.checkbox("Exclude players from selected team", value=True)

        if team and position:
            # Filter candidates
            if position == 'FW':
                candidates = df[df['pos'].str.upper().str.contains('FW', na=False)].copy()
            elif position == 'MF':
                candidates = df[df['pos'].str.upper().str.contains('MF', na=False)].copy()
            elif position == 'DF':
                candidates = df[df['pos'].str.upper().str.contains('DF', na=False)].copy()
            else:
                candidates = df[df['pos'].str.upper().str.contains('GK', na=False)].copy()

            # Age filter
            candidates = candidates[(candidates['age'] >= min_age) & (candidates['age'] <= max_age)]

            # Exclude team
            if exclude_team:
                candidates = candidates[candidates['squad'] != team]

            # Calculate score
            if position == 'FW':
                candidates['score'] = candidates['gls'].rank(pct=True) * 30 + candidates['finishing_alpha'].rank(
                    pct=True) * 40 + candidates['xg'].rank(pct=True) * 30
            elif position == 'MF':
                candidates['score'] = candidates['ast'].rank(pct=True) * 30 + candidates['playmaking_alpha'].rank(
                    pct=True) * 40 + candidates['xag'].rank(pct=True) * 30
            elif position == 'DF':
                candidates['score'] = candidates['tkl'].rank(pct=True) * 35 + candidates['int'].rank(pct=True) * 35 + \
                                      candidates['clr'].rank(pct=True) * 30
            else:
                candidates['score'] = candidates['saves'].rank(pct=True) * 50 + candidates['cs'].rank(pct=True) * 50

            candidates['rating'] = (60 + candidates['score'] * 0.35).clip(60, 95).round(1)

            top10 = candidates.nlargest(10, 'rating')

            st.subheader(f"Top 10 {position} Targets")
            st.dataframe(
                top10[['player', 'squad', 'comp', 'age', 'gls', 'ast', 'rating']],
                hide_index=True
            )

            # Chart
            fig, ax = plt.subplots(figsize=(10, 6))

            colors = ['#3498db' if r >= 80 else '#f39c12' for r in top10['rating']]
            ax.barh(top10['player'], top10['rating'], color=colors)
            ax.set_xlim(60, 100)
            ax.set_xlabel('Rating')
            ax.invert_yaxis()
            ax.set_title(f'Transfer Targets - {position}')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


if __name__ == "__main__":
    main()