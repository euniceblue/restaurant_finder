import os
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Get Azure API details from environment variables
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION')
azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')

# azure_endpoint='https://bc-api-management-uksouth.azure-api.net'
# azure_api_key='078302902b6f4944ad35e770d1a0bd70'
# azure_api_version='2023-07-01-preview'
# azure_deployment='gpt-4-0613'

# Validate that the API details are loaded correctly
if not azure_endpoint or not azure_api_key or not azure_api_version or not azure_deployment:
    raise ValueError("Please ensure AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_DEPLOYMENT are set in your environment or .env file")

def create_prompt(restaurant_name):
    prompt = f"""
    Please provide detailed information about the restaurant "{restaurant_name}". It should be in London. Keep your response under 100 words.  
    Include the following details:
    - Location (Give me the region, and the postcode. e.g. South Kensington, SW7 5PR)
    - Cuisine (e.g. Italian, Chinese)
    - Type of restaurant (e.g. pub, restaurant, bar)
    - Price point (give me in the range of price, e.g. '£20-£30')
    - A summary of the reviews and any notes (maximum 50 words, such as ambience, if it is romantic, if it is good for big groups, intimate setting, casual setting)
    - Top commented and recommended dishes (Give me a list: Prawn Pasta, Tiramisu)

    **If you can't find the restaurant, try again by formatting the name differently in terms of spacing and format and try again, or search with only part of the name, 
    which is the key part of the restaurant name. All the restaurants provided exist in the world and most of them are in London. 

    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

# def get_restaurant_info(prompt):
#     client = AzureOpenAI(
#         api_key='078302902b6f4944ad35e770d1a0bd70',
#         api_version="2023-07-01-preview",
#         azure_endpoint = 'https://bc-api-management-uksouth.azure-api.net'
#     )
#     deployment_name='gpt-4-0613'

#     response = client.chat.completions.create(
#         model="gpt-4-0613",
#         messages=prompt
#     )


def get_restaurant_info(prompt):
    client = AzureOpenAI(
        api_key = azure_api_key,
        api_version = azure_api_version,
        azure_endpoint = azure_endpoint
    )
    deployment_name = azure_deployment

    response = client.chat.completions.create(
        model= azure_deployment,
        messages=prompt
    )


    # Change this line
    response_content = response.choices[0].message.content
    
    return response_content

    # print(response.choices[0].message.content)

    # response_content = response.choices[0].message["content"]
    # print(response.choices[0].message.content)
    # print(f"API response: {response_content}")
    return response_content

    # response_content = response.choices[0].message['content']
    # print(f"API response: {response_content}")
    # return response_content



#     headers = {
#         "Content-Type": "application/json",
#         "api-key": azure_api_key
#     }
#     body = {
#         "prompt": prompt,
#         "max_tokens": 1024,
#         "temperature": 0.7
#     }
#     url = f"{azure_endpoint}/openai/deployments/{azure_deployment}/completions?api-version={azure_api_version}"
#     response = requests.post(url, headers=headers, json=body)
    
#     # Print the response for debugging
#     print("API request URL:", url)
#     print("API response status code:", response.status_code)
#     print("API response content:", response.content)
    
#     response_json = response.json()
    
#     if 'choices' in response_json and len(response_json['choices']) > 0:
#         return response_json['choices'][0]['text']
#     else:
#         raise ValueError(f"Unexpected API response structure: {response_json}")

# def parse_info(info):
#     try:
#         parsed_info = eval(info)  # Not safe for production; adjust as needed
#     except:
#         parsed_info = {"error": "Invalid format"}
#     return parsed_info

# print("Helper functions defined successfully.")
