import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.features import add_features
from src.advanced_features import calculate_h2h, calculate_surface_performance, encode_tournament_importance
from src.model import train_model, show_model_performance
from src.visualization import plot_win_probability, plot_player_stats

def get_player_stats(player_name, df, current_surface):
    """Get player statistics from DataFrame"""
    recent_matches = df[
        (df['player1_name'] == player_name) | 
        (df['player2_name'] == player_name)
    ].sort_values('date').tail(5)
    
    # Get current rank
    last_match = recent_matches.iloc[-1]
    rank = (last_match['player1_rank'] 
           if last_match['player1_name'] == player_name 
           else last_match['player2_rank'])
    
    # Calculate recent form
    wins = sum(recent_matches['winner_name'] == player_name)
    recent_form = wins / len(recent_matches)
    
    # Calculate surface win rate
    surface_matches = df[df['surface'] == current_surface]
    surface_matches = surface_matches[
        (surface_matches['player1_name'] == player_name) | 
        (surface_matches['player2_name'] == player_name)
    ]
    surface_wins = sum(surface_matches['winner_name'] == player_name)
    surface_rate = surface_wins / len(surface_matches) if len(surface_matches) > 0 else 0.5
    
    return {
        'rank': rank,
        'recent_form': recent_form,
        'surface_rate': surface_rate,
        'h2h_wins': 0,
        'h2h_losses': 0
    }

def create_prediction_input(p1_stats, p2_stats, surface, tournament, df):
    """Create input data for model prediction"""
    importance_map = {
        'Grand Slam': 4,
        'Masters 1000': 3,
        'ATP 500': 2,
        'ATP 250': 1
    }
    
    return {
        'rank_diff': p1_stats['rank'] - p2_stats['rank'],
        'recent_form_diff': p1_stats['recent_form'] - p2_stats['recent_form'],
        'surface_clay': 1 if surface == 'Clay' else 0,
        'surface_grass': 1 if surface == 'Grass' else 0,
        'h2h_ratio': p1_stats['h2h_wins'] / (p1_stats['h2h_wins'] + p1_stats['h2h_losses']) if (p1_stats['h2h_wins'] + p1_stats['h2h_losses']) > 0 else 0.5,
        'surface_advantage': p1_stats['surface_rate'] - p2_stats['surface_rate'],
        'tournament_importance': importance_map[tournament]
    }

def main():
    st.set_page_config(layout="wide")
    st.title("ATP Tennis Match Winner Predictor (2020+)")

    # Load and prepare data
    df = load_data("data/atp_tennis.csv")
    df = calculate_surface_performance(df)
    df = calculate_h2h(df)
    df = encode_tournament_importance(df)
    df = add_features(df)

    # Define features
    features = [
        'rank_diff',
        'recent_form_diff',
        'surface_clay',
        'surface_grass',
        'h2h_ratio',
        'surface_advantage',
        'tournament_importance'
    ]

    # Train model once (but don't show performance yet)
    model = train_model(df, features)

    # Create tabs
    tab1, tab2 = st.tabs(["Make Prediction", "Model Performance"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            players = sorted(set(df['player1_name'].unique()) | set(df['player2_name'].unique()))
            player1 = st.selectbox("Select Player 1", players)
            surface = st.selectbox("Select Surface", ['Hard', 'Clay', 'Grass'])

        with col2:
            player2 = st.selectbox("Select Player 2", players)
            tournament = st.selectbox("Tournament Level", 
                                    ['Grand Slam', 'Masters 1000', 'ATP 500', 'ATP 250'])

        if st.button("Predict Winner"):
            if player1 == player2:
                st.error("Please select different players")
            else:
                # Get player stats
                p1_stats = get_player_stats(player1, df, surface)
                p2_stats = get_player_stats(player2, df, surface)
                
                # Calculate H2H stats
                h2h_matches = df[
                    ((df['player1_name'] == player1) & (df['player2_name'] == player2)) |
                    ((df['player2_name'] == player1) & (df['player1_name'] == player2))
                ]
                p1_stats['h2h_wins'] = sum(h2h_matches['winner_name'] == player1)
                p1_stats['h2h_losses'] = len(h2h_matches) - p1_stats['h2h_wins']
                
                # Make prediction
                input_data = create_prediction_input(p1_stats, p2_stats, surface, tournament, df)
                proba = model.predict_proba(pd.DataFrame([input_data]))[0]
                
                # Show predictions and visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.pyplot(plot_win_probability(player1, player2, proba))
                
                with col2:
                    st.pyplot(plot_player_stats(df, player1, player2))
                
                # Show detailed stats
                st.subheader("Player Statistics")
                stats_df = pd.DataFrame({
                    'Metric': ['Current Rank', 'Recent Form', 'Surface Win Rate', 'H2H Wins'],
                    player1: [p1_stats['rank'], p1_stats['recent_form'], 
                             p1_stats['surface_rate'], p1_stats['h2h_wins']],
                    player2: [p2_stats['rank'], p2_stats['recent_form'], 
                             p2_stats['surface_rate'], p2_stats['h2h_losses']]
                })
                st.table(stats_df)

    with tab2:
        if st.button("Show Model Performance"):
            accuracy, cm_fig, imp_fig = show_model_performance(model, df, features)
            st.write(f"Model Accuracy: {accuracy:.2%}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(cm_fig)
            with col2:
                st.pyplot(imp_fig)

if __name__ == "__main__":
    main()