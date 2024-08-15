import pandas as pd
import os
from helpers import create_prompt, get_restaurant_info
import chardet
from tqdm import tqdm



def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def process_csv_files(input_folder='Saved', output_folder='Saved_gen'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all CSV files in the input folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    total_files = len(csv_files)

    # Initialize a counter for processed files
    processed_count = 0

    for filename in csv_files:
        file_path = os.path.join(input_folder, filename)
        encoding = detect_encoding(file_path)
        
        # Read the CSV file
        df = pd.read_csv(file_path, encoding='utf-8')
        
        if df.empty or 'Title' not in df.columns:
            print(f"Skipping {filename} as it is empty or does not contain 'Title' column.")
            continue

        print(f"\nProcessing file: {filename}")

        # Generate responses for each row with a progress bar
        def get_info(title):
            prompt = create_prompt(title)
            response = get_restaurant_info(prompt)
            return response

        # Use tqdm to show progress bar for rows
        tqdm.pandas(desc="Processing rows")
        df['Response'] = df['Title'].progress_apply(get_info)

        # Create the output filename
        output_file_name = filename.replace('.csv', '_gen.csv')
        output_file_path = os.path.join(output_folder, output_file_name)

        # Save the new DataFrame to the output folder
        # df.to_csv(output_file_path, columns=['Title', 'Response'], index=False)
        df.to_csv(output_file_path, columns=['Title', 'URL', 'Response'], index=False,encoding='utf-8')
        # print(f"Processed {filename} and saved to {output_file_path}")

        # Increment the processed file counter
        processed_count += 1
        print(f"Processed {processed_count}/{total_files} files.")

    print(f"\nAll files processed. Total files: {total_files}")

if __name__ == "__main__":
    process_csv_files()
