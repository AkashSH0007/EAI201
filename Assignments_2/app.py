import gradio as gr
import pandas as pd
import joblib
import numpy as np
import random
from PIL import Image
import os
from datetime import datetime

print("Loading all data and models...")

try:
    # --- Load Data ---
    team_features_df = pd.read_csv('team_features.csv')
    rankings_df = pd.read_csv('fifa_mens_rank.csv')
    
    # Process Rankings
    rankings_df['date'] = pd.to_datetime(rankings_df['date'], format='%Y')
    max_date = rankings_df['date'].max()
    latest_rankings_all_semesters = rankings_df[rankings_df['date'] == max_date]
    max_semester = latest_rankings_all_semesters['semester'].max()
    latest_rankings = latest_rankings_all_semesters[latest_rankings_all_semesters['semester'] == max_semester].copy()
    
    # Standardize Team Names 
    name_mapping = {
        "USA": "United States",
        "IR Iran": "Iran",
        "Korea Republic": "Korea, South", 
        "Tükiye": "Turkey",
        "Trkiye": "Turkey",
        "Türkiye": "Turkey", 
        "Czechia": "Czech Republic",
        "Cte d'Ivoire": "Cote d'Ivoire", 
        "CÃ´te d'Ivoire": "Cote d'Ivoire", 
        "Côte d'Ivoire": "Cote d'Ivoire", 
        "Bosnia and Herzegovina": "Bosnia-Herzegovina",
        "Switzerland": "Switzerland",
        "Greece": "Greece",
        "Scotland": "Scotland",
        "Egypt": "Egypt",
        "Canada": "Canada" 
    }
    # Apply the mapping to the ranking file
    latest_rankings['team'] = latest_rankings['team'].replace(name_mapping)
    
    team_features_df['team'] = team_features_df['team'].replace(name_mapping)
    print("Team names standardized.")
    # -----------------------------------------------

    latest_rankings = latest_rankings[['team', 'rank', 'total.points']]
    latest_rankings.rename(columns={'rank': 'team_rank', 'total.points': 'team_rank_points'}, inplace=True)

    # --- Load Models ---
    model = joblib.load('logistic_regression_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    print("SUCCESS: All models and data loaded.")
except Exception as e:
    print(f"FATAL ERROR: Could not load all necessary files. {e}")
    print("Please ensure .csv, .joblib files are in the same folder.")
    exit()

# --- Load Reports and Images ---
def load_report_text(filename):
    """Loads text from a markdown file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: {filename} not found. Please ensure all .md report files are in the same folder."
    except Exception as e:
        return f"Error loading file: {e}"

def load_image(filename):
    """Loads an image file."""
    try:
        return Image.open(filename)
    except FileNotFoundError:
        return None 

# --- Prediction & Simulation Logic ---
def predict_match(team1_name, team2_name):
    """Predicts a single match. Returns winner and a warning if data was missing."""
    warning = ""
    try:
        team1_features = team_features_df[team_features_df['team'] == team1_name].iloc[0]
        team1_rank = latest_rankings[latest_rankings['team'] == team1_name].iloc[0]
        team2_features = team_features_df[team_features_df['team'] == team2_name].iloc[0]
        team2_rank = latest_rankings[latest_rankings['team'] == team2_name].iloc[0]
    except IndexError:
        warning = f"**(Warning: Missing feature data for {team1_name} or {team2_name}. Defaulting to rank.)**"
        try:
            r1 = latest_rankings[latest_rankings['team'] == team1_name].iloc[0]['team_rank']
            r2 = latest_rankings[latest_rankings['team'] == team2_name].iloc[0]['team_rank']
            winner = team1_name if r1 < r2 else team2_name
            return winner, warning
        except:
             return team1_name, warning 
    
    feature_order = [
        'neutral', 'home_team_avg_age', 'home_team_total_market_value', 'home_team_avg_experience',
        'away_team_avg_age', 'away_team_total_market_value', 'away_team_avg_experience',
        'home_team_rank', 'home_team_rank_points', 'away_team_rank', 'away_team_rank_points',
        'home_team_win_rate', 'away_team_win_rate'
    ]
    feature_vector_data = {
        'neutral': 1, 'home_team_avg_age': team1_features['team_avg_age'],
        'home_team_total_market_value': team1_features['team_total_market_value'],
        'home_team_avg_experience': team1_features['team_avg_experience'],
        'away_team_avg_age': team2_features['team_avg_age'],
        'away_team_total_market_value': team2_features['team_total_market_value'],
        'away_team_avg_experience': team2_features['team_avg_experience'],
        'home_team_rank': team1_rank['team_rank'], 'home_team_rank_points': team1_rank['team_rank_points'],
        'away_team_rank': team2_rank['team_rank'], 'away_team_rank_points': team2_rank['team_rank_points'],
        'home_team_win_rate': 0.5, 'away_team_win_rate': 0.5
    }
    match_features = pd.DataFrame([feature_vector_data], columns=feature_order)
    match_features_scaled = scaler.transform(match_features)
    prediction_proba = model.predict_proba(match_features_scaled)[0]
    
    prob_draw, prob_team1_win, prob_team2_win = prediction_proba
    
    if prob_team1_win > prob_team2_win and prob_team1_win > prob_draw:
        return team1_name, warning
    elif prob_team2_win > prob_team1_win and prob_team2_win > prob_draw:
        return team2_name, warning
    else:
        winner = team1_name if team1_rank['team_rank'] < team2_rank['team_rank'] else team2_name
        return winner, warning

def run_simulation():
    """Runs the full 48-team simulation and returns a formatted string."""
    output_log = []
    
    try:
        all_participants = latest_rankings.sort_values(by='team_rank').head(48)
        top_16_teams_byes = list(all_participants.head(16)['team'])
        play_in_teams = list(all_participants.tail(32)['team'])
        
        output_log.append("## Simulating 2026 World Cup (48 Teams)\n")
        output_log.append(f"**Top 16 Teams (Bye to Round of 32):** `{', '.join(top_16_teams_byes)}`\n")
        random.shuffle(play_in_teams)
        
        # --- Play-in Round ---
        output_log.append("### Play-in Round (16 Matches)\n")
        round_of_32_advancers = []
        for i in range(0, 32, 2):
            team_a, team_b = play_in_teams[i], play_in_teams[i+1]
            winner, warn = predict_match(team_a, team_b)
            output_log.append(f"* {team_a} vs. {team_b}  ->  **{winner}** {warn}")
            round_of_32_advancers.append(winner)

        # --- Round of 32 ---
        output_log.append("\n### Round of 32 (16 Matches)\n")
        round_of_32_teams = top_16_teams_byes + round_of_32_advancers
        random.shuffle(round_of_32_teams)
        round_of_16_advancers = []
        for i in range(0, 32, 2):
            team_a, team_b = round_of_32_teams[i], round_of_32_teams[i+1]
            winner, warn = predict_match(team_a, team_b)
            output_log.append(f"* {team_a} vs. {team_b}  ->  **{winner}** {warn}")
            round_of_16_advancers.append(winner)

        # --- Round of 16 ---
        output_log.append("\n### Round of 16 (8 Matches)\n")
        round_of_8_advancers = []
        for i in range(0, 16, 2):
            team_a, team_b = round_of_16_advancers[i], round_of_16_advancers[i+1]
            winner, warn = predict_match(team_a, team_b)
            output_log.append(f"* {team_a} vs. {team_b}  ->  **{winner}** {warn}")
            round_of_8_advancers.append(winner)

        # --- Quarter-Finals ---
        output_log.append("\n### Quarter-Finals (4 Matches)\n")
        round_of_4_advancers = []
        for i in range(0, 8, 2):
            team_a, team_b = round_of_8_advancers[i], round_of_8_advancers[i+1]
            winner, warn = predict_match(team_a, team_b)
            output_log.append(f"* {team_a} vs. {team_b}  ->  **{winner}** {warn}")
            round_of_4_advancers.append(winner)

        # --- Semi-Finals ---
        output_log.append("\n### Semi-Finals (2 Matches)\n")
        finalists = []
        for i in range(0, 4, 2):
            team_a, team_b = round_of_4_advancers[i], round_of_4_advancers[i+1]
            winner, warn = predict_match(team_a, team_b)
            output_log.append(f"* {team_a} vs. {team_b}  ->  **{winner}** {warn}")
            finalists.append(winner)
            
        # --- Final ---
        output_log.append("\n### Final\n")
        team_a, team_b = finalists[0], finalists[1]
        winner, warn = predict_match(team_a, team_b)
        output_log.append(f"* {team_a} vs. {team_b}  ->  **{winner}** {warn}\n")
        
        output_log.append("---")
        output_log.append(f"## 🏆 Predicted 2026 World Cup Winner: {winner} 🏆")
        output_log.append(f"## Predicted Finalists: {finalists[0]} and {finalists[1]}")
        
        return "\n".join(output_log)

    except Exception as e:
        return f"An Error Occurred: {e}"

# --- Build the Gradio App ---
print("Building Gradio UI...")

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="blue"), title="FIFA 2026 Predictor") as app:
    gr.Markdown("# ⚽ FIFA World Cup 2026 Predictor\nAn application by Akash.")
    
    with gr.Tabs():
        # --- Tab 1: Tournament Simulator ---
        with gr.TabItem("🏆 Tournament Simulator (Task 5)"):
            gr.Markdown("Click the button to run a new, randomized 48-team knockout simulation using our trained Logistic Regression model and the latest team data.")
            sim_button = gr.Button("▶️ Run New 48-Team Simulation")
            sim_output = gr.Markdown("Simulation results will appear here...")
            sim_button.click(fn=run_simulation, inputs=None, outputs=sim_output)

        # --- Tab 2: Model Evaluation ---
        with gr.TabItem("📈 Model Evaluation (Task 3)"):
            gr.Markdown(load_report_text('week_3_report.md'))
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Logistic Regression (Our Chosen Model)")
                    gr.Image(load_image('confusion_matrix_lr.png'), label='Confusion Matrix (Logistic Regression)')
                    gr.Image(load_image('roc_curve_lr.png'), label='ROC Curve (Logistic Regression)')
                with gr.Column():
                    gr.Markdown("#### Random Forest (Overfit Model)")
                    gr.Image(load_image('confusion_matrix_rf.png'), label='Confusion Matrix (Random Forest)')
                    gr.Image(load_image('roc_curve_rf.png'), label='ROC Curve (Random Forest)')
        
        # --- Tab 3: Feature Importance ---
        with gr.TabItem("💡 Feature Importance (Task 4)"):
            gr.Markdown(load_report_text('Task_4_Interpretation.md'))
            with gr.Row():
                with gr.Column():
                    gr.Image(load_image('feature_importance_lr.png'), label='Feature Importance (Logistic Regression)')
                with gr.Column():
                    gr.Image(load_image('feature_importance_rf.png'), label='Feature Importance (Random Forest)')

        # --- Tab 4: Data & Reports ---
        with gr.TabItem("📊 Data Collection (Task 1)"):
            gr.Markdown(load_report_text('week_1_report.md'))
            
        # --- Tab 5: Model Training Report ---
        with gr.TabItem("🤖 Model Training (Task 2)"):
            gr.Markdown(load_report_text('week_2_report.md'))
            
        # --- Tab 6: Reflection Report ---
        with gr.TabItem("🧠 Reflection (Task 5)"):
            gr.Markdown(load_report_text('week_4_reflection.md'))

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio app...")
    app.launch(share=False) 

