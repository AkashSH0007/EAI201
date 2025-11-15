import pandas as pd
import numpy as np
import json

# Define the file name from Task 1
TASK_1_OUTPUT_FILE = 'task_1_integrated_data.csv'
ENGINEERED_FEATURES = ['ecosystem_type', 'predator_score']

# --- G) Required Outputs ---
print("--- Task 1: G) Required Outputs ---")

try:
    # Try to load the file created in Task 1
    merged_data = pd.read_csv(TASK_1_OUTPUT_FILE)
    print(f"Successfully loaded '{TASK_1_OUTPUT_FILE}'.")
except FileNotFoundError:
    print(f"'{TASK_1_OUTPUT_FILE}' not found. Re-running Task 1 to generate it...")
    
    # --- Re-run Task 1 ---
    # --- A) Load Datasets ---
    print("\n--- Step A: Loading Datasets ---")
    try:
        zoo_df = pd.read_csv('zoo.csv')
        class_df = pd.read_csv('class.csv')
        with open('auxiliary_metadata.json', 'r') as f:
            aux_data_list = json.load(f)
        aux_df = pd.DataFrame(aux_data_list)
        print("All files loaded.")
    except Exception as e:
        print(f"Fatal Error: Could not load initial files. {e}")
        raise

    # --- C) Standardize Field Names and Values ---
    print("\n--- Step C: Standardizing Auxiliary Data ---")
    aux_df['conservation_status'] = aux_df['conservation_status'].fillna(aux_df.get('conservation')).fillna(aux_df.get('status'))
    aux_df['habitat_type'] = aux_df['habitat'].fillna(aux_df.get('habitats'))
    aux_df['diet'] = aux_df['diet'].fillna(aux_df.get('diet_type'))
    aux_df = aux_df.drop(columns=[col for col in ['conservation', 'status', 'habitat', 'habitats', 'diet_type'] if col in aux_df.columns])
    if 'diet' in aux_df.columns:
        aux_df['diet'] = aux_df['diet'].replace('omnivor', 'omnivore')
    if 'habitat_type' in aux_df.columns:
        aux_df['habitat_type'] = aux_df['habitat_type'].str.lower().replace('fresh water', 'freshwater')
    
    # --- D) Merge Datasets ---
    print("\n--- Step D: Merging Datasets ---")
    merged_df = pd.merge(zoo_df, class_df[['Class_Number', 'Class_Type']], left_on='class_type', right_on='Class_Number', how='left')
    merged_df = merged_df.drop(columns='Class_Number')
    merged_df['merge_key'] = merged_df['animal_name'].str.lower()
    aux_df['merge_key'] = aux_df['animal_name'].str.lower()
    aux_df = aux_df.drop_duplicates(subset=['merge_key'], keep='first')
    merged_data = pd.merge(merged_df, aux_df.drop(columns='animal_name', errors='ignore'), on='merge_key', how='left')
    merged_data = merged_data.drop(columns='merge_key')
    print("Merge complete.")

    # --- E) Handle Missing Values ---
    print("\n--- Step E: Handling Missing Values ---")
    cat_cols_to_fill = ['habitat_type', 'diet', 'conservation_status']
    existing_cat_cols = [col for col in cat_cols_to_fill if col in merged_data.columns]
    if existing_cat_cols:
        merged_data[existing_cat_cols] = merged_data[existing_cat_cols].ffill()
    num_cols_with_nans = merged_data.select_dtypes(include=np.number).columns[merged_data.select_dtypes(include=np.number).isna().any()].tolist()
    if num_cols_with_nans:
        merged_data[num_cols_with_nans] = merged_data[num_cols_with_nans].bfill()
    print("Missing values handled.")
    
    # --- F) Feature Engineering ---
    print("\n--- Step F: Feature Engineering ---")
    merged_data['habitat_type'] = merged_data['habitat_type'].fillna('unknown')
    eco_conditions = [
        (merged_data['habitat_type'] == 'freshwater'),
        (merged_data['habitat_type'] == 'marine'),
        (merged_data['aquatic'] == 0)
    ]
    eco_choices = [1, 2, 0]
    merged_data['ecosystem_type'] = np.select(eco_conditions, eco_choices, default=3)
    
    merged_data['diet'] = merged_data['diet'].fillna('unknown') 
    diet_conditions = [
        (merged_data['diet'] == 'carnivore'),
        (merged_data['diet'] == 'omnivore')
    ]
    diet_choices = [3, 2]
    merged_data['predator_score'] = np.select(diet_conditions, diet_choices, default=1)
    print("Feature engineering complete.")
    
    # Save the file again
    merged_data.to_csv(TASK_1_OUTPUT_FILE, index=False)
    print(f"Task 1 re-run complete. '{TASK_1_OUTPUT_FILE}' has been generated.")


# --- Now, execute the requested print statements ---
print("\n--- Required Outputs for Task 1 ---")

# print(f"dataset shape : {self.merged_data.shape}")
print(f"dataset shape : {merged_data.shape}")

# print(f"missing values : {self.merged_dat5a.isnull().sum().sum()}")
# Note: The user's prompt had a typo 'dat5a', I am correcting it to 'merged_data'
missing_values_count = merged_data.isnull().sum().sum()
print(f"missing values : {missing_values_count}")

# print(f"duplicate rows : {self.merged_data.duplicated().sum()}")
print(f"duplicate rows : {merged_data.duplicated().sum()}")

# print("\n  first 3 rows")
print("\n  first 3 rows")
# print(self.merged_data.head(3))
print(merged_data.head(3))

# print(f"\n engineered features: {list(engineered_features_names)}")
print(f"\n engineered features: {ENGINEERED_FEATURES}")