import pandas as pd

try:
    # Load the team features file
    team_features_df = pd.read_csv('team_features.csv')
    
    # Get all unique team names
    unique_teams = sorted(list(team_features_df['team'].unique()))
    
    print("--- All Team Names in 'team_features.csv' ---")
    
    # Print all of them so we can find the right ones
    for team in unique_teams:
        print(team)
        
    print(f"\nFound {len(unique_teams)} total unique teams.")

except FileNotFoundError:
    print("ERROR: team_features.csv not found.")
except Exception as e:
    print(f"An error occurred: {e}")
