import pandas as pd
from datetime import datetime


players_file = 'players.csv'
valuations_file = 'player_valuations.csv'
appearances_file = 'appearances.csv'
output_file = 'team_features.csv'

try:
    #  Load Data 
    print("Loading player and valuation data...")
    players_df = pd.read_csv(players_file)
    valuations_df = pd.read_csv(valuations_file)
    appearances_df = pd.read_csv(appearances_file, usecols=['player_id', 'minutes_played'])

    # --- Step 2: Clean Player Data & Calculate Age ---
    print("Cleaning players and calculating age...")
    players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'], errors='coerce')
    reference_date = datetime(2026, 6, 1) # Use a fixed date for 2026 prediction
    players_df['age'] = players_df['date_of_birth'].apply(
        lambda dob: (reference_date - dob).days // 365 if pd.notnull(dob) else None
    )
    players_cleaned_df = players_df[[
        'player_id', 'country_of_citizenship', 'age'
    ]].copy()
    players_cleaned_df.dropna(subset=['age'], inplace=True)
    players_cleaned_df['age'] = players_cleaned_df['age'].astype(int)

    # --- Step 3: Get Latest Market Value ---
    print("Processing market valuations...")
    valuations_df['date'] = pd.to_datetime(valuations_df['date'], errors='coerce')
    valuations_df.dropna(subset=['date', 'market_value_in_eur', 'player_id'], inplace=True)
    latest_valuations = valuations_df.sort_values(by='date').drop_duplicates(subset=['player_id'], keep='last')
    latest_valuations = latest_valuations[['player_id', 'market_value_in_eur']]
    players_cleaned_df = pd.merge(players_cleaned_df, latest_valuations, on='player_id', how='left')

    # --- Step 4: Calculate Player Experience ---
    print("Calculating player experience...")
    appearances_df.dropna(subset=['player_id', 'minutes_played'], inplace=True)
    player_experience = appearances_df.groupby('player_id')['minutes_played'].sum().reset_index()
    player_experience.rename(columns={'minutes_played': 'total_minutes_played'}, inplace=True)
    players_cleaned_df = pd.merge(players_cleaned_df, player_experience, on='player_id', how='left')
    players_cleaned_df['total_minutes_played'] = players_cleaned_df['total_minutes_played'].fillna(0)
    players_cleaned_df['total_minutes_played'] = players_cleaned_df['total_minutes_played'].astype(int)

    # --- Step 5: Aggregate Player Data to Team Level ---
    print("Aggregating features at team level...")
    players_cleaned_df['market_value_in_eur'] = players_cleaned_df['market_value_in_eur'].fillna(0)
    team_features = players_cleaned_df.groupby('country_of_citizenship').agg(
        team_avg_age=('age', 'mean'),
        team_total_market_value=('market_value_in_eur', 'sum'),
        team_avg_experience=('total_minutes_played', 'mean')
    ).reset_index()
    team_features.rename(columns={'country_of_citizenship': 'team'}, inplace=True)
    team_features['team_avg_age'] = round(team_features['team_avg_age'], 2)
    team_features['team_avg_experience'] = round(team_features['team_avg_experience'], 2)
    
    # --- Save the Team Features ---
    team_features.to_csv(output_file, index=False)
    print(f"\nSUCCESS: Team features saved to {output_file}")
    print(team_features.head())

except FileNotFoundError as e:
    print(f"\nERROR: {e}. Make sure CSV files are in the same folder.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
