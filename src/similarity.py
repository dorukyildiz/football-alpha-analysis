import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from analysis import get_data

# Position-based metrics
FW_METRICS = ['gls', 'ast', 'g_a', 'xg', 'xag', 'npxg', 'g_pk', 'kp', 'xa', 'ppa',
              'touches', 'carries', 'prgr', 'mis', 'pkwon', 'pkcon']

MF_METRICS = ['gls', 'ast', 'g_a', 'xg', 'xag', 'npxg', 'g_pk', 'tkl', 'tklw',
              'int', 'tkl_int', 'prgp', 'prgc', 'kp', 'xa', 'ppa', 'touches',
              'carries', 'prgr', 'mis', 'dis', 'crdy', 'crdr', 'recov']

DF_METRICS = ['tkl', 'tklw', 'blocks', 'int', 'tkl_int', 'clr', 'err', 'prgp',
              'prgc', 'touches', 'carries', 'mis', 'dis', 'crdy', 'crdr', 'recov']

GK_METRICS = ['ga', 'saves', 'save', 'cs', 'pka', 'pksv']


def prepare_data():
    """Fetch and prepare data for similarity analysis"""
    df = get_data()
    df = df[df['col_90s'] >= 5].copy()
    return df


def get_primary_position(pos):
    """Extract primary position from position string"""
    if pd.isna(pos):
        return None
    pos = pos.upper().replace('"', '').strip()
    if ',' in pos:
        return pos.split(',')[0].strip()
    return pos


def get_all_positions(pos):
    """Get all positions a player can play"""
    if pd.isna(pos):
        return []
    pos = pos.upper().replace('"', '').strip()
    if ',' in pos:
        return [p.strip() for p in pos.split(',')]
    return [pos]


def filter_by_position(df, player_pos):
    """Filter players by compatible positions"""
    player_positions = get_all_positions(player_pos)

    if not player_positions:
        return df

    def is_compatible(row_pos):
        row_positions = get_all_positions(row_pos)
        # Check if any position matches
        for pp in player_positions:
            for rp in row_positions:
                if pp == rp:
                    return True
                # FW can match with FW,MF or MF,FW
                if pp in rp or rp in pp:
                    return True
        return False

    return df[df['pos'].apply(is_compatible)]


def get_metrics_for_position(pos):
    """Return relevant metrics based on position"""
    primary_pos = get_primary_position(pos)

    if primary_pos == 'GK':
        return GK_METRICS
    elif primary_pos == 'DF':
        return DF_METRICS
    elif primary_pos == 'MF':
        return MF_METRICS
    elif primary_pos == 'FW':
        return FW_METRICS
    else:
        return FW_METRICS + MF_METRICS  # Default to attacking metrics


def calculate_similarity(df, metrics):
    """Calculate similarity matrix using cosine similarity"""
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        return None, []

    metrics_df = df[available_metrics].copy()
    metrics_df = metrics_df.fillna(0)

    scaler = StandardScaler()
    metrics_scaled = scaler.fit_transform(metrics_df)
    similarity_matrix = cosine_similarity(metrics_scaled)

    return similarity_matrix, available_metrics


def find_similar_players(df, player_name, top_n=5):
    """Find most similar players to a given player"""

    # Find player
    player_idx = df[df['player'].str.lower() == player_name.lower()].index

    if len(player_idx) == 0:
        player_idx = df[df['player'].str.lower().str.contains(player_name.lower())].index
        if len(player_idx) == 0:
            return None, None, None, f"Player '{player_name}' not found"
        elif len(player_idx) > 1:
            matches = df.loc[player_idx, 'player'].tolist()
            return None, None, None, f"Multiple matches: {matches}"

    player_idx = player_idx[0]
    original = df.loc[player_idx]

    # Filter by position
    filtered_df = filter_by_position(df, original['pos'])

    if len(filtered_df) < 2:
        return None, None, None, f"Not enough players with position {original['pos']}"

    # Get position-specific metrics
    metrics = get_metrics_for_position(original['pos'])

    # Calculate similarity on filtered players
    similarity_matrix, used_metrics = calculate_similarity(filtered_df, metrics)

    if similarity_matrix is None:
        return None, None, None, "Could not calculate similarity"

    # Find player's new index in filtered df
    player_row = filtered_df.index.get_loc(player_idx)
    similarities = similarity_matrix[player_row]

    # Get top similar (excluding self)
    similar_indices = np.argsort(similarities)[::-1][1:top_n + 1]

    results = []
    for idx in similar_indices:
        similar_player_idx = filtered_df.index[idx]
        similarity_score = similarities[idx] * 100

        results.append({
            'player': filtered_df.loc[similar_player_idx, 'player'],
            'squad': filtered_df.loc[similar_player_idx, 'squad'],
            'pos': filtered_df.loc[similar_player_idx, 'pos'],
            'comp': filtered_df.loc[similar_player_idx, 'comp'],
            'similarity': round(similarity_score, 1)
        })

    return results, original, used_metrics, None


def compare_metrics(original, similar_players, df, metrics):
    """Show metric comparison between players"""

    display_metrics = [m for m in metrics if m in df.columns]

    print("\n" + "=" * 140)
    print(f"METRIC COMPARISON ({len(display_metrics)} metrics)")
    print("=" * 140)

    # Header
    header = f"{'Player':<25}"
    for m in display_metrics:
        header += f"{m:<8}"
    print(header)
    print("-" * 140)

    # Original player
    row = f">> {original['player']:<22}"
    for m in display_metrics:
        val = original[m] if m in original and pd.notna(original[m]) else 0
        row += f"{val:<8.1f}"
    print(row)

    # Similar players
    for sp in similar_players:
        p = df[df['player'] == sp['player']].iloc[0]
        row = f"({sp['similarity']:.0f}%) {p['player']:<19}"
        for m in display_metrics:
            val = p[m] if m in p and pd.notna(p[m]) else 0
            row += f"{val:<8.1f}"
        print(row)

    # Similar players
    for sp in similar_players:
        p = df[df['player'] == sp['player']].iloc[0]
        row = f"({sp['similarity']:.0f}%) {p['player']:<19}"
        for m in display_metrics:
            val = p[m] if m in p and pd.notna(p[m]) else 0
            row += f"{val:<10.1f}"
        print(row)


def main():
    print("Loading data...")
    df = prepare_data()
    print(f"Loaded {len(df)} players")
    print("Ready!\n")

    while True:
        print("=" * 60)
        player_name = input("Enter player name (or 'quit' to exit): ").strip()

        if player_name.lower() == 'quit':
            break

        similar, original, metrics, error = find_similar_players(df, player_name)

        if error:
            print(f"[ERROR] {error}")
            continue

        print(f"\nPLAYERS SIMILAR TO: {original['player']}")
        print(f"Team: {original['squad']} | Position: {original['pos']} | League: {original['comp']}")
        print(f"Metrics used: {len(metrics)} ({', '.join(metrics[:5])}...)")

        print("\n" + "-" * 60)
        print(f"{'Rank':<6} {'Player':<25} {'Team':<20} {'Position':<12} {'Similarity':<10}")
        print("-" * 60)

        for i, sp in enumerate(similar, 1):
            print(f"{i:<6} {sp['player']:<25} {sp['squad']:<20} {sp['pos']:<12} {sp['similarity']}%")

        compare_metrics(original, similar, df, metrics)


if __name__ == "__main__":
    main()