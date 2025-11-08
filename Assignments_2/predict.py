import pandas as pd
import joblib
import numpy as np
import random

# --- Configuration ---
TEAM_FEATURES_FILE = 'team_features.csv'
RANKING_FILE = 'fifa_mens_rank.csv' 
MODEL_FILE = 'logistic_regression_model.joblib'
SCALER_FILE = 'scaler.joblib'

# --- Load All Necessary Files ---
print("--- Loading Models and Data for Prediction ---")
try:
    # Load the team features (Age, Value, Experience)
    team_features_df = pd.read_csv(TEAM_FEATURES_FILE)
    
    # Load the latest rankings
    rankings_df = pd.read_csv(RANKING_FILE)
    
    # --- !!! FIX 1: Filter to SINGLE latest ranking list ---
    # Convert 'date' (YYYY) to a proper datetime object
    rankings_df['date'] = pd.to_datetime(rankings_df['date'], format='%Y')
    # Find the most recent date in the file
    max_date = rankings_df['date'].max()
    # Filter for that year
    latest_rankings_all_semesters = rankings_df[rankings_df['date'] == max_date]
    # Find the most recent semester in that year
    max_semester = latest_rankings_all_semesters['semester'].max()
    # Filter for the single latest list
    latest_rankings = latest_rankings_all_semesters[latest_rankings_all_semesters['semester'] == max_semester].copy()
    print(f"Loaded {len(latest_rankings)} teams from most recent ranking: {max_date.year} (Semester {max_semester})")
    # --- END OF FIX 1 ---

    # --- !!! FIX 2: Standardize Team Names (EXPANDED) ---
    name_mapping = {
        # Previous Mappings
        "USA": "United States",
        "IR Iran": "Iran",
        "Czechia": "Czech Republic",
        
        # New Mappings from latest run
        "Korea Republic": "South Korea", # Original name in rank file
        "South Korea": "Korea, South",   # Map to the likely name in players.csv
        "Trkiye": "Turkey",           # Map the broken character name
        "Tükiye": "Turkey",            # Map the proper accent name
        "Cte d'Ivoire": "Ivory Coast",  # Map the broken character name
        "Cte d'Ivoire": "Ivory Coast",   # Map the common variation
        "Scotland": "Scotland",          # Map to itself (in case of whitespace issues)
        "Greece": "Greece",
        "Russia": "Russia"
        # Add any other mismatches you see here
    }
    latest_rankings['team'] = latest_rankings['team'].replace(name_mapping)
    print("Team names standardized.")
    

    latest_rankings = latest_rankings[['team', 'rank', 'total.points']]
    latest_rankings.rename(columns={'rank': 'team_rank', 'total.points': 'team_rank_points'}, inplace=True)

    # Load the trained model and scaler
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    print("SUCCESS: All models and data loaded.")

except FileNotFoundError as e:
    print(f"ERROR: Could not find a necessary file: {e}")
    print("Please make sure all files are in the same folder.")
    exit()
except Exception as e:
    print(f"An error occurred during loading: {e}")
    exit()

# --- Prediction Function (Same as before) ---
def predict_match(team1_name, team2_name, team_features_df, rankings_df, model, scaler):
    """
    Predicts the outcome of a single match between two teams.
    Returns the winning team's name.
    """
    try:
        # Get Team 1 (Home) Features
        team1_features = team_features_df[team_features_df['team'] == team1_name].iloc[0]
        team1_rank = rankings_df[rankings_df['team'] == team1_name].iloc[0]
        
        # Get Team 2 (Away) Features
        team2_features = team_features_df[team_features_df['team'] == team2_name].iloc[0]
        team2_rank = rankings_df[rankings_df['team'] == team2_name].iloc[0]
    except IndexError:
        # This will now only trigger if a team is in the Top 48 but NOT in our 100+ team_features file
        print(f"Warning: Could not find feature data for {team1_name} or {team2_name}.")
        # Default to higher-ranked team winning if data is missing
        try:
            r1 = rankings_df[rankings_df['team'] == team1_name].iloc[0]['team_rank']
            r2 = rankings_df[rankings_df['team'] == team2_name].iloc[0]['team_rank']
            if r1 < r2: return team1_name
            else: return team2_name
        except:
             return team1_name # Failsafe
    except Exception as e:
        print(f"An error occurred looking up team data: {e}")
        return None

    # --- Create the Feature Vector for the Model ---
    feature_order = [
        'neutral', 'home_team_avg_age', 'home_team_total_market_value', 'home_team_avg_experience',
        'away_team_avg_age', 'away_team_total_market_value', 'away_team_avg_experience',
        'home_team_rank', 'home_team_rank_points', 'away_team_rank', 'away_team_rank_points',
        'home_team_win_rate', 'away_team_win_rate'
    ]
    
    feature_vector_data = {
        'neutral': 1,
        'home_team_avg_age': team1_features['team_avg_age'],
        'home_team_total_market_value': team1_features['team_total_market_value'],
        'home_team_avg_experience': team1_features['team_avg_experience'],
        'away_team_avg_age': team2_features['team_avg_age'],
        'away_team_total_market_value': team2_features['team_total_market_value'],
        'away_team_avg_experience': team2_features['team_avg_experience'],
        'home_team_rank': team1_rank['team_rank'],
        'home_team_rank_points': team1_rank['team_rank_points'],
        'away_team_rank': team2_rank['team_rank'],
        'away_team_rank_points': team2_rank['team_rank_points'],
        'home_team_win_rate': 0.5, 
        'away_team_win_rate': 0.5
    }
    
    match_features = pd.DataFrame([feature_vector_data], columns=feature_order)
    match_features_scaled = scaler.transform(match_features)

    # --- Make Prediction ---
    prediction_proba = model.predict_proba(match_features_scaled)[0]
    
    prob_draw = prediction_proba[0]
    prob_team1_win = prediction_proba[1]
    prob_team2_win = prediction_proba[2]

    if prob_team1_win > prob_team2_win and prob_team1_win > prob_draw:
        return team1_name # Team 1 wins
    elif prob_team2_win > prob_team1_win and prob_team2_win > prob_draw:
        return team2_name # Team 2 wins
    else:
        # Draw is most likely, or probabilities are tied.
        # Default to the higher-ranked team.
        if team1_rank['team_rank'] < team2_rank['team_rank']:
            return team1_name
        else:
            return team2_name

# --- Simulate Tournament (Top 48) ---
print("\n--- Simulating 2026 World Cup Knockout Stage (Top 48 Teams) ---")

try:
    # Get the Top 48 teams from our rankings
    all_participants = latest_rankings.sort_values(by='team_rank').head(48)
    
    # Split into Top 16 (Byes) and 17-48 (Play-in)
    top_16_teams_byes = list(all_participants.head(16)['team'])
    play_in_teams = list(all_participants.tail(32)['team'])
    
    print(f"Top 16 Teams (Bye to Round of 32): {top_16_teams_byes}")
    print(f"\nTeams 17-48 (Play-in Round): {play_in_teams}")
    
    random.shuffle(play_in_teams) # Shuffle to create random matchups
    
    # --- Play-in Round (Round of 48) ---
    print("\n--- Play-in Round Matchups (16 Matches) ---")
    round_of_32_advancers = []
    for i in range(0, 32, 2):
        team_a = play_in_teams[i]
        team_b = play_in_teams[i+1]
        print(f"Match: {team_a} vs. {team_b}")
        winner = predict_match(team_a, team_b, team_features_df, latest_rankings, model, scaler)
        print(f"-> Winner: {winner}")
        round_of_32_advancers.append(winner)

    # --- Round of 32 ---
    print("\n--- Round of 32 Matchups (16 Matches) ---")
    round_of_32_teams = top_16_teams_byes + round_of_32_advancers
    random.shuffle(round_of_32_teams) # Shuffle for random matchups
    
    round_of_16_advancers = []
    for i in range(0, 32, 2):
        team_a = round_of_32_teams[i]
        team_b = round_of_32_teams[i+1]
        print(f"Match: {team_a} vs. {team_b}")
        winner = predict_match(team_a, team_b, team_features_df, latest_rankings, model, scaler)
        print(f"-> Winner: {winner}")
        round_of_16_advancers.append(winner)

    # --- Round of 16 ---
    print("\n--- Round of 16 Matchups (8 Matches) ---")
    round_of_8_advancers = []
    for i in range(0, 16, 2):
        team_a = round_of_16_advancers[i]
        team_b = round_of_16_advancers[i+1]
        print(f"Match: {team_a} vs. {team_b}")
        winner = predict_match(team_a, team_b, team_features_df, latest_rankings, model, scaler)
        print(f"-> Winner: {winner}")
        round_of_8_advancers.append(winner)

    # --- Quarter-Finals ---
    print("\n--- Quarter-Final Matchups (4 Matches) ---")
    round_of_4_advancers = []
    for i in range(0, 8, 2):
        team_a = round_of_8_advancers[i]
        team_b = round_of_8_advancers[i+1]
        print(f"Match: {team_a} vs. {team_b}")
        winner = predict_match(team_a, team_b, team_features_df, latest_rankings, model, scaler)
        print(f"-> Winner: {winner}")
        round_of_4_advancers.append(winner)

    # --- Semi-Finals ---
    print("\n--- Semi-Final Matchups (2 Matches) ---")
    finalists = []
    for i in range(0, 4, 2):
        team_a = round_of_4_advancers[i]
        team_b = round_of_4_advancers[i+1]
        print(f"Match: {team_a} vs. {team_b}")
        winner = predict_match(team_a, team_b, team_features_df, latest_rankings, model, scaler)
        print(f"-> Winner (Finalist): {winner}")
        finalists.append(winner)
        
    # --- Final ---
    print("\n--- Final Matchup ---")
    team_a = finalists[0]
    team_b = finalists[1]
    print(f"Final: {team_a} vs. {team_b}")
    winner = predict_match(team_a, team_b, team_features_df, latest_rankings, model, scaler)
    
    print("\n--- Simulation Complete ---")
    print(f"Predicted Finalists: {finalists[0]} and {finalists[1]}")
    print(f"Predicted 2026 World Cup Winner: {winner}")
    
except Exception as e:
    print(f"\nAN ERROR OCCURRED DURING SIMULATION: {e}")
    print("Please check all input files and team names.")

