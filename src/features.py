import pandas as pd

def calculate_recent_form(df, n=5):
    df = df.sort_values('date')
    df['player1_recent'] = 0.0
    df['player2_recent'] = 0.0
    players = {}

    for i, row in df.iterrows():
        p1 = row['player1_name']
        p2 = row['player2_name']

        recent_p1 = players.get(p1, [])
        recent_p2 = players.get(p2, [])

        df.at[i, 'player1_recent'] = sum(recent_p1[-n:])/min(len(recent_p1), n) if recent_p1 else 0.0
        df.at[i, 'player2_recent'] = sum(recent_p2[-n:])/min(len(recent_p2), n) if recent_p2 else 0.0

        players[p1] = recent_p1 + [1 if row['winner_name'] == p1 else 0]
        players[p2] = recent_p2 + [1 if row['winner_name'] == p2 else 0]

    df['recent_form_diff'] = df['player1_recent'] - df['player2_recent']
    df = df.drop(columns=['player1_recent', 'player2_recent'])

    return df

def add_features(df):
    # Rank difference
    df['rank_diff'] = df['player1_rank'] - df['player2_rank']

    # Target variable
    df['target'] = df.apply(lambda row: 1 if row['winner_name'] == row['player1_name'] else 0, axis=1)

    # Store original surface
    surface_original = df['surface'].copy()

    # Encode surface
    df = pd.get_dummies(df, columns=['surface'], drop_first=True)

    # Ensure all expected surface columns exist
    for col in ['surface_clay', 'surface_grass']:
        if col not in df.columns:
            df[col] = 0

    # Add back original surface for visualizations
    df['surface'] = surface_original

    # Recent form difference
    df = calculate_recent_form(df, n=5)

    return df