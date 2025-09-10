import pandas as pd
import numpy as np

def calculate_h2h(df):
    """Calculate head-to-head record for player1 vs player2"""
    df['h2h_ratio'] = df.apply(lambda row: get_h2h_record(row, df), axis=1)
    return df

def get_h2h_record(row, df):
    """Get head-to-head win ratio for player1 against player2"""
    previous_matches = df[df['date'] < row['date']]
    h2h = previous_matches[
        ((previous_matches['player1_name'] == row['player1_name']) & 
         (previous_matches['player2_name'] == row['player2_name'])) |
        ((previous_matches['player1_name'] == row['player2_name']) & 
         (previous_matches['player2_name'] == row['player1_name']))
    ]
    if len(h2h) == 0:
        return 0.5
    wins = sum(h2h['winner_name'] == row['player1_name'])
    return wins / len(h2h)

def calculate_surface_performance(df):
    """Calculate win rate on each surface for both players"""
    surfaces = df['surface'].unique()
    players = set(df['player1_name'].unique()) | set(df['player2_name'].unique())
    
    surface_stats = {}
    for player in players:
        for surface in surfaces:
            matches = df[
                (df['surface'] == surface) & 
                ((df['player1_name'] == player) | (df['player2_name'] == player))
            ]
            wins = matches[matches['winner_name'] == player].shape[0]
            surface_stats[(player, surface)] = wins / len(matches) if len(matches) > 0 else 0.5
    
    def get_surface_winrate(row):
        p1_rate = surface_stats.get((row['player1_name'], row['surface']), 0.5)
        p2_rate = surface_stats.get((row['player2_name'], row['surface']), 0.5)
        return p1_rate - p2_rate
    
    df['surface_advantage'] = df.apply(get_surface_winrate, axis=1)
    return df

def encode_tournament_importance(df):
    """Encode tournament series importance"""
    importance_map = {
        'Grand Slam': 4,
        'Masters 1000': 3,
        'ATP 500': 2,
        'ATP 250': 1
    }
    df['tournament_importance'] = df['tournament_series'].map(importance_map).fillna(1)
    return df