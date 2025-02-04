# Chat with Your Documents

## Overview

This project allows users to upload a document (PDF, DOCX, or TXT) and interact with it through a chatbot interface. Users can ask questions related to the content of the document, and the system will provide accurate responses using Google Gemini AI and FAISS vector database for retrieval.

## Features

- Upload and process PDF, DOCX, and TXT files
- Extract text from the uploaded document
- Chunk the text for efficient retrieval
- Store document embeddings using FAISS
- Use Google Gemini AI for intelligent responses
- Maintain conversation history
- Interactive chat interface

## Technologies Used

- **Streamlit**: For the web interface
- **LangChain**: For text processing and conversational retrieval
- **FAISS**: For vector-based search and retrieval
- **Google Generative AI**: For generating responses
- **PyPDF2**: For extracting text from PDFs
- **python-docx**: For extracting text from DOCX files

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemini AI)

### Installation

1. Clone the repository:
   ```sh
   https://github.com/hassanalisyed021/Talking-Docs.git
   cd Talking-Docs
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up your Google API Key:
   ```sh
   export GOOGLE_API_KEY="your-api-key"
   ```

### Running the Application

Start the Streamlit app by running:

```sh
streamlit run app.py
```

## Usage

1. Upload a document (PDF, DOCX, or TXT).
2. Wait for the document to be processed.
3. Enter a question in the chat input field.
4. Receive AI-generated responses based on the document content.

## Contributing

Contributions are welcome! Feel free to open issues and submit pull requests.

## Contact

For any queries, reach out via email or create an issue on the repository.

