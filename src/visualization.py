import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_win_probability(player1, player2, proba):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar([player1, player2], [proba[1], proba[0]], 
                  color=['skyblue', 'lightgreen'])
    
    ax.set_ylim(0, 1)
    ax.set_title('Win Probability', pad=20, size=14)
    plt.xticks(rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_player_stats(df, player1, player2):
    recent_matches1 = df[
        (df['player1_name'] == player1) | 
        (df['player2_name'] == player1)
    ].tail(10)
    
    recent_matches2 = df[
        (df['player1_name'] == player2) | 
        (df['player2_name'] == player2)
    ].tail(10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Win rates by surface
    surfaces = ['Hard', 'Clay', 'Grass']
    p1_rates = []
    p2_rates = []
    
    for surface in surfaces:
        # Player 1
        p1_surface = df[
            (df['surface'] == surface) & 
            ((df['player1_name'] == player1) | (df['player2_name'] == player1))
        ]
        wins = sum(p1_surface['winner_name'] == player1)
        p1_rates.append(wins/len(p1_surface) if len(p1_surface) > 0 else 0)
        
        # Player 2
        p2_surface = df[
            (df['surface'] == surface) & 
            ((df['player1_name'] == player2) | (df['player2_name'] == player2))
        ]
        wins = sum(p2_surface['winner_name'] == player2)
        p2_rates.append(wins/len(p2_surface) if len(p2_surface) > 0 else 0)
    
    x = range(len(surfaces))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], p1_rates, width, label=player1)
    ax1.bar([i + width/2 for i in x], p2_rates, width, label=player2)
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Surface Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(surfaces)
    ax1.legend()
    
    # Recent form
    dates1 = recent_matches1['date'].dt.date
    form1 = [1 if w == player1 else 0 for w in recent_matches1['winner_name']]
    dates2 = recent_matches2['date'].dt.date
    form2 = [1 if w == player2 else 0 for w in recent_matches2['winner_name']]
    
    ax2.plot(dates1, form1, 'o-', label=player1)
    ax2.plot(dates2, form2, 'o-', label=player2)
    ax2.set_ylabel('Win/Loss')
    ax2.set_title('Recent Form')
    ax2.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig