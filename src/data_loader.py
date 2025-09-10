import pandas as pd

def load_data(path="data/atp_tennis.csv"):
    df = pd.read_csv(path)

    # Convert 'Date' to datetime and drop rows with missing dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # Filter for 2020 and later
    df = df[df['Date'].dt.year >= 2020].reset_index(drop=True)

    # Rename columns to match expected names in the rest of the code
    df = df.rename(columns={
        'Date': 'date',
        'Tournament': 'tourney_name',
        'Surface': 'surface',
        'Round': 'round',
        'Player_1': 'player1_name',
        'Player_2': 'player2_name',
        'Rank_1': 'player1_rank',
        'Rank_2': 'player2_rank',
        'Winner': 'winner_name',
        'Pts_1': 'player1_points',
        'Pts_2': 'player2_points',
        'Odd_1': 'player1_odds',
        'Odd_2': 'player2_odds',
        'Series': 'tournament_series'
    })

    # Keep only relevant columns (removed duplicate 'surface')
    columns_to_keep = [
        'date', 'tourney_name', 'surface', 'round',
        'player1_name', 'player2_name', 'player1_rank', 'player2_rank',
        'player1_points', 'player2_points', 'player1_odds', 'player2_odds',
        'tournament_series', 'winner_name'
    ]
    df = df[columns_to_keep]

    # Fill missing numeric values
    numeric_cols = ['player1_rank', 'player2_rank', 
                   'player1_points', 'player2_points',
                   'player1_odds', 'player2_odds']
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Drop rows with missing winner
    df = df.dropna(subset=['winner_name'])

    return df