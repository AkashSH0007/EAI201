Week 1 Report: Data Collection and Preparation

1. Introduction
This report details the activities  during Week 1  for the project focused on predicting the FIFA World Cup 2026 finalists using machine learning. The primary goals for this week were data sourcing, attempting custom web scraping, data cleaning, and initial feature engineering.
2. Data Sourcing
The project requires historical match data and relevant team/player statistics.
Primary Data Sources Used:
1.	Historical Match Results:
o	Dataset: International football results from 1872 to 2025
o	Source: Kaggle (results.csv)
o	Content: File contains match details including date, home team, away team, scores, tournament, location, and neutrality.
2.	Player and Team Data (Transfermarkt Scrape):
o	Dataset: Football Data from Transfermarkt
o	Source: Transfermarkt
o	Content: This dataset,scraped from Transfermarkt, contains  information on players (ID, name, date of birth,position), player market valuations.
3.	FIFA Rankings:
o	Dataset: FIFA Men's World Ranking
o	Source: Kaggle (fifa_mens_rank.csv)
o	Content: Provided historical FIFA rankings for national teams, including rank, team name, points, and date. This dataset was used to incorporate the required FIFA ranking feature.
3. Custom Web Scraper Development 
Websites Targeted:
•	Transfermarkt.com (Squads, Market Values)
•	FIFA.com (Official Rankings)
•	worldfootball.net (Rankings)
•	myKhel.com (Rankings)
•	FBref.com (Squads, Player Stats)
Methods Attempted:
1.	Python requests and BeautifulSoup: Initial attempts used standard libraries to fetch and parse static HTML content.
2.	Selenium with ChromeDriver: To handle JavaScript-rendered content and bypass basic bot detection, Selenium was employed both locally and within a Google Colab environment.
3.	Techniques Employed:
o	Setting realistic User-Agent headers.
o	Implementing delays (time.sleep) between requests, including randomized delays.
o	Adding retry logic to handle intermittent connection issues or temporary blocks.
o	Using different element locators (ID, Class Name, XPath) for robustness.
o	Attempting interaction with elements (e.g., cookie consent buttons).
Scraping Logic (Example - Transfermarkt):
The initial logic aimed to:
1.	Fetch the main national team ranking pages to get links to individual team pages.
2.	Iterate through team links.
3.	On each team page, locate the main player table (using class_='items').
4.	Iterate through table rows (<tr>).
5.	Extract player name, age, and market value from specific table cells (<td>) based on their position or class attributes.
6.	Handle variations in table structure (different layouts for age/birthday).
Challenges Faced:
•	HTTP 403 Forbidden Errors: The most common issue. Websites actively detected automated scripts (even with headers and delays) and blocked access. This occurred across multiple sites (myKhel.com, worldfootball.net, FIFA.com).
•	Rapidly Changing HTML Structure: Websites frequently update their layout and class names (likely to deter scraping).

Conclusion on Scraping: We scraped Player data from top 100 teams.
4. Data Cleaning
Data cleaning was performed on the datasets obtained from Kaggle and TransferMarkt using Python and the pandas library.
•	results.csv:
o	Converted the date column to datetime objects.
o	Filtered the dataset to retain only matches where the tournament column was exactly 'FIFA World Cup'.
•	players.csv:
o	Converted date_of_birth to datetime objects.
o	Calculated player age based on a reference date (June 1, 2026).
o	Selected relevant columns (player_id, name, country_of_citizenship, age, etc.).
o	Dropped rows where age calculation failed (missing/invalid date of birth).
•	player_valuations.csv:
o	Converted date to datetime objects.
o	Dropped rows with missing player_id, date, or market_value_in_eur.
o	Identified and kept only the latest valuation record for each unique player_id.
•	appearances.csv:
o	Loaded only essential columns (player_id, minutes_played).
o	Dropped rows with missing player_id or minutes_played.
•	fifa_mens_rank.csv:
o	Selected relevant columns (rank, team, date, total.points).
o	Renamed columns.
o	Converted the numeric date column (YYYY format) into a datetime object (YYYY-01-01) suitable for merging.
o	Dropped rows with missing rank or converted rank_date.
5. Feature Engineering
•	Age : Calculated directly from date_of_birth in players.csv.
•	Market Value (market_value_in_eur): Extracted as the latest valuation for each player from player_valuations.csv and merged into the player data. Missing values were later filled with 0 before team aggregation.
•	Player Experience (total_minutes_played): Calculated by grouping appearances.csv by player_id and summing the minutes_played. This sum was merged into the player data. Missing values (players with no appearance data) were filled with 0.
•	Team Aggregates: Player data was grouped by country_of_citizenship to calculate:
o	team_avg_age
o	team_total_market_value
o	team_avg_experience (based on mean total_minutes_played)
•	Merging Team Features: These aggregated team features were merged twice into the world_cup_df (once for home_team, once for away_team). Missing values arising from team name mismatches were imputed using the median value calculated across all matches.
•	Goal Difference (goal_difference): Calculated as home_score - away_score for each match.
•	FIFA Ranking (home_team_rank, away_team_rank, etc.): Merged from fifa_mens_rank.csv using pd.merge_asof to join the most recent ranking available on or before the match date. Missing values (primarily for matches before 1993) were imputed using the median rank/points.
•	Historical Win Rate (home_team_win_rate, away_team_win_rate): Calculated dynamically for each match by looking at all previous World Cup matches for each team and computing their win percentage up to that point. A function with caching was used for efficiency.
•	Target Variable (match_outcome): Created based on scores (1=HomeWin, 0=Draw, 2=AwayWin).

6. Final Dataset Description (world_cup_final_dataset.csv)
 The final cleaned dataset contains 964 rows (World Cup matches) and 23 columns.
Column Name= date, home_team, away_team, home_score, away_score, tournament, city, country, neutral, home_team_avg_age, home_team_total_market_value, home_team_avg_experience, away_team_avg_age, away_team_total_market_value, away_team_avg_experience, goal_difference, home_team_rank, home_team_rank_points, away_team_rank, away_team_rank_points,home_team_win_rate, away_team_win_rate, match_outcome.

7. Conclusion
Week 1 focused on acquiring and preparing the data. The custom web scraping completed, the required data sources were identified, cleaned, and processed. All necessary features, including team average age, market value, player experience, goal difference, FIFA ranking, and historical win rate, were successfully done by EDA. The final dataset, world_cup_final_dataset.


