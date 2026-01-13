import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# Use a consistent header for all requests
HEADERS = {'User-Agent': 
           'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

def scrape_team_players(team_name, team_url):
    """Scrapes player data for a single team with a retry mechanism."""
    
    # Try to fetch the page up to 3 times
    for attempt in range(3):
        try:
            response = requests.get(team_url, headers=HEADERS)
            response.raise_for_status() 
            break 
        except requests.exceptions.HTTPError as e:
            print(f"  FAILED attempt {attempt + 1} for {team_name}. Reason: {e}. Retrying...")
            time.sleep(random.uniform(5, 10))
    else:
        print(f"  FAILED to scrape {team_name} after all attempts. Skipping.")
        return pd.DataFrame() 

    # --- Continue scraping if request was successful ---
    soup = BeautifulSoup(response.content, 'html.parser')
    player_table = soup.find('table', class_='items')
    player_data = []
    
    if player_table:
        for row in player_table.find('tbody').find_all('tr', class_=['odd', 'even']):
            cols = row.find_all('td')
            if len(cols) > 5:
                try:
                    name = cols[1].find('a').text.strip()
                    age = cols[4].text.strip()
                    value = cols[5].text.strip()
                    
                    player_data.append({
                        'Team': team_name,
                        'Player Name': name,
                        'Position': age,
                        'Date of Birth(Age)': value
                    })
                except Exception:
                    continue # Skip a malformed row
    
    print(f"  SUCCESS: Scraped {len(player_data)} players from {team_name}.")
    return pd.DataFrame(player_data)

# --- Main Script ---

# MODIFIED: Base URL for team links
base_url = 'https://www.transfermarkt.com'

print("--- Stage 1: Fetching top 100 team links ---")
team_links = []

# MODIFIED: Loop through pages 1, 2, 3, and 4 of the world rankings
for page_num in range(1, 5):
    ranking_url = f'https://www.transfermarkt.com/statistik/weltrangliste?page={page_num}'
    print(f"Fetching links from ranking page {page_num}...")
    
    try:
        response = requests.get(ranking_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        table = soup.find('table', class_='items')
        
        if table:
            for row in table.find('tbody').find_all('tr', class_=['odd', 'even']):
                cell = row.find('td', class_='hauptlink')
                if cell and cell.find('a'):
                    team_name = cell.find('a').get('title')
                    team_relative_url = cell.find('a').get('href')
                    
                    # Ensure it's a valid team link
                    if '/verein/' in team_relative_url:
                        team_full_url = base_url + team_relative_url
                        # Add to list if not already present (prevents duplicates)
                        if not any(d['name'] == team_name for d in team_links):
                            team_links.append({'name': team_name, 'url': team_full_url})
        
        # Be polite, wait a bit between list pages
        time.sleep(random.uniform(1, 3))
        
    except Exception as e:
        print(f"Error fetching page {page_num}: {e}")

print(f"\nFound {len(team_links)} total teams to scrape.\n")

# --- Stage 2: Loop through the links and scrape each team ---
print("--- Stage 2: Scraping player data for each team ---")
all_teams_df_list = []

if not team_links:
    print("Could not find any team links. Exiting.")
else:
    # Loop through all 100 teams
    for i, team in enumerate(team_links):
        print(f"Scraping team {i+1}/{len(team_links)}: {team['name']}")
        
        team_df = scrape_team_players(team['name'], team['url'])
        if not team_df.empty:
            all_teams_df_list.append(team_df)
        
        # Be respectful, wait a random time between scraping team pages
        sleep_duration = random.uniform(2, 5) 
        print(f"  ...waiting for {sleep_duration:.2f} seconds...\n")
        time.sleep(sleep_duration)

    # --- Final Step: Combine all dataframes into one and save ---
    if all_teams_df_list:
        print("\n--- Finalizing ---")
        final_df = pd.concat(all_teams_df_list, ignore_index=True)
        
        # MODIFIED: New filename
        final_df.to_csv('top_100_squad_data.csv', index=False)
        
        print(f"Scraping complete! Combined data saved to 'top_100_squad_data.csv'.")
        print(f"Total players scraped: {len(final_df)}")
    else:
        print("No data was scraped. The final CSV was not created.")