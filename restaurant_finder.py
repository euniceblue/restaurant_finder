import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Get Azure API details from environment variables
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION')
azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')

# Initialize the AzureOpenAI client globally to avoid redundant initialization
client = AzureOpenAI(
    api_key=azure_api_key,
    api_version=azure_api_version,
    azure_endpoint=azure_endpoint
)

# Utility to split data into chunks of 5 lines each
def split_data_into_chunks(df, chunk_size=5):
    return [df.iloc[i:i + chunk_size].to_string(index=False) for i in range(0, len(df), chunk_size)]

# Load the CSV file and split into chunks of 5 lines
@st.cache_data
def load_data_and_split_chunks(chunk_size=5):
    data = pd.read_csv('all_restaurants.csv')
    chunks = split_data_into_chunks(data, chunk_size)
    return data, chunks

# Function to generate embeddings
def get_embeddings(texts, model="text-embedding-ada-002"):
    """Get embeddings for a list of texts using the specified model."""
    texts = [text.replace("\n", " ") for text in texts]
    embeddings = []
    for text in texts:
        try:
            response = client.embeddings.create(input=[text], model=model)
            # Correctly access the embedding from the response
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            st.error(f"Error retrieving embedding for text: {text[:30]}... Error: {e}")
            embeddings.append(np.zeros((768,)))  # Append a zero vector in case of error
    return np.array(embeddings)

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(chunks, query, embedding_model, top_k=3):
    chunk_embeddings = get_embeddings(chunks, model=embedding_model)
    query_embedding = get_embeddings([query], model=embedding_model)[0].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_k_indices], [similarities[i] for i in top_k_indices]

# Generate a prompt with the user requirements and the relevant restaurant chunks
def generate_prompt(user_input, relevant_chunks):
    context = "\n".join(relevant_chunks)
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""
        The user has provided the following requirements for a restaurant: "{user_input}"
        Based on the following restaurant data, please recommend the top 3 matching restaurants by picking the most relevant rows and extracting the details from the provided data:

        {context}

        *Ensure to pick the most relevant rows based on the user requirements and include the corresponding Google Map URL from the data.*

        *Please ensure each recommendation is clearly separated and formatted exactly as follows, add emojis to the response:*

        1
            Name: <title>
            Cuisine: <cuisine>
            Price Range: <price_range>
            Area: <area>
            Vibe: <vibe>
            Famous for: <3 best dishes>
            Google Map URL: <url>

        2
            Name: <title>
            Cuisine: <cuisine>
            Price Range: <price_range>
            Area: <area>
            Vibe: <vibe>
            Famous for: <3 best dishes>
            Google Map URL: <url>

        3
            Name: <title>
            Cuisine: <cuisine>
            Price Range: <price_range>
            Area: <area>
            Vibe: <vibe>
            Famous for: <3 best dishes>
            Google Map URL: <url>
        """}
    ]

# Streamlined function to get restaurant recommendations using Azure OpenAI
def get_restaurant_recommendations(prompt):
    response = client.chat.completions.create(
        model=azure_deployment,
        messages=prompt
    )
    return response.choices[0].message.content.strip()

# Streamlit app
def main():
    st.title("PLANET")
    st.write("What kind of restaurant are you looking for? Please provide details such as cuisine type, price range, area, vibe, etc.")

    with st.form(key="user_input_form"):
        user_input = st.text_area("Enter your request:")
        
        # Optional slider for selecting the price range
        price_range = st.slider(
            "Select your desired price range (£):",
            min_value=0,
            max_value=100,  # Adjust max_value based on your dataset's price range
            value=(0, 100),  # Set default range to cover the entire range, making it non-selective initially
            step=1
        )

        submit_button = st.form_submit_button(label="Find Restaurants")

    if submit_button and user_input:
        with st.spinner('Searching for the best matches...'):
            data, chunks = load_data_and_split_chunks()

            # Check if the slider was adjusted by the user
            if price_range != (0, 100):
                user_input += f" with a price range between £{price_range[0]} and £{price_range[1]}"

            relevant_chunks, _ = retrieve_relevant_chunks(chunks, user_input, embedding_model="text-embedding-ada-002")
            prompt = generate_prompt(user_input, relevant_chunks)
            recommendations = get_restaurant_recommendations(prompt)

        if recommendations:
            st.subheader("Top 3 restaurants that match your requirements:")
            st.write(recommendations)
        else:
            st.write("No matching restaurants found. Please try different details.")

if __name__ == "__main__":
    main()

  
