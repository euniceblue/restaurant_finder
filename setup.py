import os
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Azure API details from environment variables
azure_endpoint = os.getenv('AZURE_OPENAI_END_POINT')
azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')

# Validate that the environment variables are loaded correctly
if not azure_endpoint or not azure_api_key:
    raise ValueError("Please ensure AZURE_OPENAI_END_POINT and AZURE_OPENAI_API_KEY are set in your environment or .env file")

# Print the environment variables for debugging
print(f"Azure Endpoint: {azure_endpoint}")
print(f"Azure API Key: {azure_api_key}")

# Define paths for input and output folders
input_folder = 'Saved'  # Directory where your original CSV files are located
output_folder = 'Saved_gen'  # Directory where you want to save processed CSV files

print("Setup and libraries loaded successfully.")
