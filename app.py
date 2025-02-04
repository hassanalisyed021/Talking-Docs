import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import tempfile
from docx import Document
import PyPDF2
import google.generativeai as genai

# Set the API key globally
os.environ["GOOGLE_API_KEY"] = "AIzaSyC9MFKn8wSB8bUDScuwRkbshIhHLm-lKqw"
genai.configure(api_key="AIzaSyC9MFKn8wSB8bUDScuwRkbshIhHLm-lKqw")

# Set page theme to dark
st.set_page_config(
    page_title="Chat with Your Documents",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    .uploadedFile {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #4A4A4A;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #1E1E1E;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.getvalue().decode("utf-8")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        sample_embeddings = embeddings.embed_query(text_chunks[0])
        dimension = len(sample_embeddings)
        
        vectorstore = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversation_chain(vectorstore):
    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        return_generated_question=True,
    )
    return conversation_chain

def handle_user_input(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation.invoke({
            "question": user_question
        })
        
        # Display only the assistant's response
        st.write("Response:", response["answer"])
        
        # Update chat history
        st.session_state.chat_history = response.get('chat_history', [])
    else:
        st.warning("Please upload a document first!")

def main():
    st.header("💬 Chat with Your Documents")
    st.subheader("Upload your document and start asking questions")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Center-aligned file uploader
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        uploaded_file = st.file_uploader(
            "Upload your document (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"]
        )

    # Process the uploaded file
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Extract text from the uploaded file
            text = extract_text_from_file(uploaded_file)
            
            # Get text chunks
            text_chunks = get_text_chunks(text)

            # Create vector store
            vectorstore = get_vectorstore(text_chunks)

            if vectorstore is not None:
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Document processed successfully! You can now ask questions about it.")

    # Chat interface
    st.markdown("---")
    user_question = st.text_input("Ask a question about your document:", key="question_input")
    if user_question:
        handle_user_input(user_question)

if __name__ == '__main__':
    main() 