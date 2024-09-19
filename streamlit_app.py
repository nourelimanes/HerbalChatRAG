import os
from tqdm import tqdm
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
from typing import Generator

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
books_dir = os.path.join(current_dir, "books")
persistent_directory = os.path.join(db_dir, "chroma_db_herbpathy")

# Configure the Streamlit app
st.set_page_config(page_icon="💬", layout="wide", page_title="Herbal Remedies Finder")

# Define URL
url = "https://herbpathy.com/"

# Initialize global variables
vector_store = None  

# Generate embeddings for text chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 

    
# Web data loading
def load_web_data(url: str):
    loader = WebBaseLoader(url)
    web_documents  = loader.load()
    return web_documents 

# books data loading
def load_pdf_files(book_dir: str):
    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )
    book_documents = []
    for book_file in os.listdir(book_dir):
        if book_file.endswith((".pdf", ".txt")):
            loader = PyPDFLoader(os.path.join(book_dir, book_file))
            book_docs = loader.load()
            for doc in book_docs:
                # Add metadata to each document indicating its source
                doc.metadata = {"source": book_file}
                book_documents.append(doc)
    return book_documents

# Split documents into chunks
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks




# Process the external data at the start
with st.spinner("Loading and processing data ..."):
    try:
        if not os.path.exists(persistent_directory):
            # Load and process web data
            web_documents = load_web_data(url)
            web_chunks = split_text_into_chunks(web_documents)

            # Load and process PDF data
            pdf_documents = load_pdf_files(books_dir)
            pdf_chunks = split_text_into_chunks(pdf_documents)

            # Combine chunks from both sources
            all_chunks = web_chunks + pdf_chunks

            print(f"\n--- Creating vector store in {persistent_directory} ---")
            vector_store = Chroma.from_documents(tqdm(all_chunks, desc="Saving chunks"), embeddings, persist_directory=persistent_directory)
            print(f"--- Finished creating vector store in {persistent_directory} ---")
            st.success("Data loaded and processed successfully!")

        else:
            print(f"Vector store {persistent_directory} already exists. No need to initialize.")
            vector_store = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

        
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")



# Set up Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Add a centered title at the top
st.markdown(
    """
    <h1 style='text-align: center;'>Herbal Remedies Finder</h1>
    """,
    unsafe_allow_html=True
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, how are you? Is there anything you are looking for about Herbal Remedies?"}
    ]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3-70b-8192"  # Default to llama3-70b-8192

# Define model details
models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
}

# Set default model
model_option = "llama3-70b-8192"  # Fixed model

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

# Set max tokens
max_tokens = 5000  # Fixed to 5000 as per your request

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Define the system prompt to guide the AI behavior
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def query_vector_store(query: str):
    """Query the vector store for relevant documents."""
    if vector_store:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        results = retriever.invoke(query)
        return results
    else:
        return []

if prompt := st.chat_input("Enter your prompt here..."):
    # Append user's prompt to session messages
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Optionally query the vector store for relevant information
    context_documents = query_vector_store(prompt)
    context_info = "\n".join(doc.page_content for doc in context_documents)

    # Prepare system prompt with context for the AI to follow
    system_prompt = qa_system_prompt.format(context=context_info)

    # Combine previous messages into chat history for context
    chat_history = "\n".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in st.session_state.messages])

    # Combine system prompt, context, and user prompt
    combined_prompt = f"{system_prompt}\n\nChat History:\n{chat_history}\n\nUser: {prompt}"

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": combined_prompt}],
            max_tokens=max_tokens,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append({"role": "assistant", "content": combined_response})
