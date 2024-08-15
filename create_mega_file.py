import pandas as pd
import os

def create_mega_file(input_folder='Saved_gen', output_file='all_restaurants.csv'):
    # List to store all dataframes
    all_data = []

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Add a column to identify the original file (cuisine type)
            df['Cuisine_Type'] = filename.replace('.csv', '')
            
            # Append to the list of dataframes
            all_data.append(df)

    # Concatenate all dataframes
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save the combined dataframe to a CSV file
        combined_df.to_csv(output_file, index=False)
        print(f"Mega file created: {output_file}")
        print(f"Total rows: {len(combined_df)}")
    else:
        print("No CSV files found in the specified directory.")

# Usage
if __name__ == "__main__":
    create_mega_file(input_folder='Saved_gen', output_file='all_restaurants.csv')