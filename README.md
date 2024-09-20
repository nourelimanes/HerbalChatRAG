# Herbal Remedies Finder

## Overview

**Herbal Remedies Finder** is a chatbot application designed to help users find information about herbal remedies. The app leverages a Retrieval-Augmented Generation (RAG) approach, integrating data from external sources such as books and web APIs to provide accurate and relevant responses. Built using LangChain, Groq, and Streamlit, this application offers a seamless and interactive experience for users seeking information on herbal remedies.

## Features

- **Chatbot Interface**: Users can interact with the chatbot to inquire about herbal remedies.
- **Data Integration**: Incorporates information from 15 books and a web API.
- **Guided AI Behavior**: Designed to minimize hallucinations and provide accurate responses.
- **Advanced Data Handling**: Processes and stores data efficiently using Chroma for vector storage.

## Setup and Installation

### Prerequisites

- Python 3.12
- [Poetry](https://python-poetry.org/) (for dependency management)

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-repo/herbal-remedies-finder.git
   cd herbal-remedies-finder
2. **Install Dependencies**

   ```bash
   poetry install
3. **Setup Environment Variables**

    Create a `.env` file in the root directory of the project with the following content:
    ```bash
    GROQ_API_KEY=your_groq_api_key_here
4. **Run the Application**

    ```bash
    poetry run streamlit run streamlit_app.py
