import streamlit as st
import numpy as np
import os
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env file
load_dotenv()


# # Initialize the Streamlit app
# st.title('Knowledge Base Chatbot with RAG and LangChain')

# # Get Azure API details from environment variables
# azure_endpoint = os.getenv('AZURE_OPENAI_END_POINT')
# azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')

# # Initialize conversation memory
# memory = ConversationBufferMemory()

# # Check if the API details have been provided
# if azure_endpoint and azure_api_key:
#     api_version = '2023-07-01-preview'  # Use the correct API version

#     # Initialize the AzureOpenAI client
#     client = AzureOpenAI(
#         api_key=azure_api_key,
#         api_version=api_version,
#         azure_endpoint=azure_endpoint
#     )

#     # Define the embedding model as a constant
#     EMBEDDING_MODEL = "text-embedding-ada-002"

#     def get_embeddings(texts, model=EMBEDDING_MODEL):
#         """Get embeddings for a list of texts using the specified model."""
#         texts = [text.replace("\n", " ") for text in texts]
#         embeddings = []
#         for text in texts:
#             try:
#                 response = client.embeddings.create(input=[text], model=model)
#                 embeddings.append(response.data[0].embedding)
#             except Exception as e:
#                 st.error(f"Error retrieving embedding for text: {text[:30]}... Error: {e}")
#                 embeddings.append(np.zeros((1,)))  # Append a zero vector in case of error
#         return np.array(embeddings)

#     def split_text_into_chunks(text, chunk_size=500):
#         """Splits text into chunks of specified size."""
#         return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

#     def retrieve_relevant_chunks(chunks, query, embedding_model, top_k=5):
#         """Retrieves top_k relevant chunks based on the query using the embedding model."""
#         chunk_embeddings = get_embeddings(chunks, model=embedding_model)
#         query_embedding = get_embeddings([query], model=embedding_model)[0].reshape(1, -1)

#         similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
#         top_k_indices = np.argsort(similarities)[-top_k:][::-1]

#         return [chunks[i] for i in top_k_indices], [similarities[i] for i in top_k_indices]

#     def answer_query_rag(question, chunks, embedding_model):
#         """Answer the question using RAG approach."""
#         relevant_chunks, similarities = retrieve_relevant_chunks(chunks, question, embedding_model)
        
#         # Join the relevant chunks into a single context using a default separator
#         context = '\n\n'.join(relevant_chunks)
        
#         # Retrieve past interactions from session state
#         chat_history = st.session_state.get('chat_history', [])
#         chat_history_str = "\n".join([f"Q: {entry['question']} A: {entry['answer']}" for entry in chat_history])
        
#         # Combine context with chat history
#         prompt = f"Document: {context}\n\nChat History:\n{chat_history_str}\n\nQuestion: {question}\nAnswer:"
        
#         response = client.chat.completions.create(
#             model="gpt-4-0613",
#             messages=[
#                 {"role": "system", "content": "You are an assistant that helps to answer questions based on provided documents and previous interactions."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=150  # You can adjust this based on your needs
#         )
#         return response.choices[0].message.content, response.usage.total_tokens, response.usage.prompt_tokens, response.usage.completion_tokens, relevant_chunks, similarities

#     def process_text(text, chunk_size=500):
#         """Process the text into chunks of the selected size with gaps between them."""
#         chunks = split_text_into_chunks(text, chunk_size)
#         return chunks

#     def display_chat_history():
#         """Display the chat history with better formatting."""
#         if "chat_history" in st.session_state and st.session_state.chat_history:
#             st.subheader("Chat History")
#             for i, entry in enumerate(st.session_state.chat_history, 1):
#                 st.markdown(f"**Q{i}:** {entry['question']}")
#                 st.markdown(f"**A{i}:** {entry['answer']}")
#                 st.write("---")  # Adding a horizontal line for better separation
#         else:
#             st.write("No chat history available.")

#     def update_chat_history(question, answer):
#         """Append the new question and answer to the chat history."""
#         st.session_state.chat_history.append({"question": question, "answer": answer})

#     # Initialize session state variables
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     if "chunks" not in st.session_state:
#         st.session_state.chunks = []

#     # File uploader widget
#     uploaded_file = st.file_uploader("Choose a text file", type="txt")

#     if uploaded_file is not None:
#         if uploaded_file.name.endswith(".txt"):
#             # Read the contents of the uploaded file
#             file_contents = uploaded_file.read().decode("utf-8")

#             # Process the text based on user input
#             st.session_state.chunks = process_text(file_contents)
            
#             # Section for RAG Q&A
#             st.subheader("Chat with the Knowledge Base")

#             display_chat_history()

#             user_question_rag = st.text_input("Ask a question:")
#             if st.button("Submit"):
#                 if user_question_rag:
#                     answer_rag, total_tokens_rag, prompt_tokens_rag, completion_tokens_rag, relevant_chunks, similarities = answer_query_rag(user_question_rag, st.session_state.chunks, EMBEDDING_MODEL)
#                     update_chat_history(user_question_rag, answer_rag)
#                     # Display the latest answer
#                     st.write(f"**Answer:** {answer_rag}")
#                     # Update chat history display
#                     display_chat_history()
#         else:
#             st.error("Please upload a file with a .txt extension.")
# else:
#     st.info("Please upload a .txt file.")
