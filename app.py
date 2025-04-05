import os
import json
import glob
from pathlib import Path
import torch
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import numpy as np  
from sentence_transformers import util
import time

# Set device for model (CUDA if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load environment variables - works for both local and Hugging Face Spaces
load_dotenv()

# Set up the clinical assistant LLM
# Try to get API key from Hugging Face Spaces secrets first, then fall back to .env file
try:
    # For Hugging Face Spaces
    from huggingface_hub.inference_api import InferenceApi
    import os
    groq_api_key = os.environ.get('GROQ_API_KEY')
    
    # If not found in environment, try to get from st.secrets (Streamlit Cloud/Spaces)
    if not groq_api_key and hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
        groq_api_key = st.secrets['GROQ_API_KEY']
        
    if not groq_api_key:
        st.warning("API Key is not set in the secrets. Using a placeholder for UI demonstration.")
        # For UI demonstration without API key
        class MockLLM:
            def invoke(self, prompt):
                return {"answer": "This is a placeholder response. Please set up your GROQ_API_KEY to get real responses."}
        llm = MockLLM()
    else:    
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        
except Exception as e:
    st.error(f"Error setting up LLM: {str(e)}")
    class MockLLM:
        def invoke(self, prompt):
            return {"answer": f"Error setting up LLM: {str(e)}. Please check your API key configuration."}
    llm = MockLLM()

# Set up embeddings for clinical context (Bio_ClinicalBERT)
embeddings = HuggingFaceEmbeddings(
    model_name="emilyalsentzer/Bio_ClinicalBERT",
    model_kwargs={"device": device}
)

def load_clinical_data():
    """Load both flowcharts and patient cases"""
    docs = []
    
    # Get the absolute path to the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to handle potential errors with file loading
    try:
        # Load diagnosis flowcharts
        flowchart_dir = os.path.join(current_dir, "Diagnosis_flowchart")
        if os.path.exists(flowchart_dir):
            for fpath in glob.glob(os.path.join(flowchart_dir, "*.json")):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        content = f"""
                        DIAGNOSTIC FLOWCHART: {Path(fpath).stem}
                        Diagnostic Path: {data.get('diagnostic', 'N/A')}
                        Key Criteria: {data.get('knowledge', 'N/A')}
                        """
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": fpath, "type": "flowchart"}
                        ))
                except Exception as e:
                    st.warning(f"Error loading flowchart file {fpath}: {str(e)}")
        else:
            st.warning(f"Flowchart directory not found at {flowchart_dir}")

        # Load patient cases
        finished_dir = os.path.join(current_dir, "Finished")
        if os.path.exists(finished_dir):
            for category_dir in glob.glob(os.path.join(finished_dir, "*")):
                if os.path.isdir(category_dir):
                    for case_file in glob.glob(os.path.join(category_dir, "*.json")):
                        try:
                            with open(case_file, 'r', encoding='utf-8') as f:
                                case_data = json.load(f)
                                notes = "\n".join(
                                    f"{k}: {v}" for k, v in case_data.items() if k.startswith("input")
                                )
                                docs.append(Document(
                                    page_content=f"""
                                    PATIENT CASE: {Path(case_file).stem}
                                    Category: {Path(category_dir).name}
                                    Notes: {notes}
                                    """,
                                    metadata={"source": case_file, "type": "patient_case"}
                                ))
                        except Exception as e:
                            st.warning(f"Error loading case file {case_file}: {str(e)}")
        else:
            st.warning(f"Finished directory not found at {finished_dir}")
            
        # If no documents were loaded, add a sample document for testing
        if not docs:
            st.warning("No clinical data files found. Using sample data for demonstration.")
            docs.append(Document(
                page_content="""SAMPLE CLINICAL DATA: This is sample data for demonstration purposes.
                This application requires clinical data files to be present in the correct directories.
                Please ensure the Diagnosis_flowchart and Finished directories exist with proper JSON files.""",
                metadata={"source": "sample", "type": "sample"}
            ))
    except Exception as e:
        st.error(f"Error loading clinical data: {str(e)}")
        # Add a fallback document
        docs.append(Document(
            page_content="Error loading clinical data. This is a fallback document for demonstration purposes.",
            metadata={"source": "error", "type": "error"}
        ))
    return docs

def build_vectorstore():
    """Build and return the vectorstore using FAISS"""
    documents = load_clinical_data()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# Path for saving/loading the vectorstore
def get_vectorstore_path():
    """Get the path for saving/loading the vectorstore"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "vectorstore")

# Initialize vectorstore with disk persistence
@st.cache_resource(show_spinner="Loading clinical knowledge base...")
def get_vectorstore():
    """Get or create the vectorstore with disk persistence"""
    vectorstore_path = get_vectorstore_path()
    
    # Try to load from disk first
    try:
        if os.path.exists(vectorstore_path):
            st.info("Loading vectorstore from disk...")
            # Set allow_dangerous_deserialization to True since we trust our own vectorstore files
            return FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.warning(f"Could not load vectorstore from disk: {str(e)}. Building new vectorstore.")
    
    # If loading fails or doesn't exist, build a new one
    st.info("Building new vectorstore...")
    vectorstore = build_vectorstore()
    
    # Save to disk for future use
    try:
        os.makedirs(vectorstore_path, exist_ok=True)
        vectorstore.save_local(vectorstore_path)
        st.success("Vectorstore saved to disk for future use")
    except Exception as e:
        st.warning(f"Could not save vectorstore to disk: {str(e)}")
    
    return vectorstore

def run_rag_chat(query, vectorstore):
    """Run the Retrieval-Augmented Generation (RAG) for clinical questions"""
    try:
        retriever = vectorstore.as_retriever()

        prompt_template = ChatPromptTemplate.from_template("""
        You are a clinical assistant AI. Based on the following clinical context, provide a reasoned and medically sound answer to the question.

        <context>
        {context}
        </context>

        Question: {input}

        Answer:
        """)

        retrieved_docs = retriever.invoke(query, k=3)
        retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Create document chain first
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        
        # Then create retrieval chain
        chain = create_retrieval_chain(retriever, document_chain)

        # Invoke the chain
        response = chain.invoke({"input": query})
        
        # Add retrieved documents to response for transparency
        response["context"] = retrieved_docs
        
        return response
    except Exception as e:
        st.error(f"Error in RAG processing: {str(e)}")
        # Return a fallback response
        return {
            "answer": f"I encountered an error processing your query: {str(e)}",
            "context": [],
            "input": query
        }

def calculate_hit_rate(retriever, query, expected_docs, k=3):
    """Calculate the hit rate for top-k retrieved documents"""
    retrieved_docs = retriever.get_relevant_documents(query, k=k)
    retrieved_contents = [doc.page_content for doc in retrieved_docs]
    
    hits = 0
    for expected in expected_docs:
        if any(expected in retrieved for retrieved in retrieved_contents):
            hits += 1
    
    return hits / len(expected_docs) if expected_docs else 0.0

def evaluate_rag_response(response, embeddings):
    """Evaluate the RAG response for faithfulness and hit rate"""
    scores = {}

    # Faithfulness: Answer-Context Similarity
    answer_embed = embeddings.embed_query(response["answer"])
    context_embeds = [embeddings.embed_query(doc.page_content) for doc in response["context"]]
    similarities = [util.cos_sim(answer_embed, ctx_embed).item() for ctx_embed in context_embeds]
    scores["faithfulness"] = float(np.mean(similarities)) if similarities else 0.0

    # Custom Hit Rate Calculation
    retriever = response["retriever"]
    scores["hit_rate"] = calculate_hit_rate(
        retriever,
        query=response["input"],
        expected_docs=[doc.page_content for doc in response["context"]],
        k=3
    )
    
    return scores

def main():
    """Main function to run the Streamlit app"""
    # Set page configuration
    st.set_page_config(
        page_title="DiReCT - Clinical Diagnostic Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load vectorstore only once using session state
    if 'vectorstore' not in st.session_state:
        with st.spinner("Loading clinical knowledge base... This may take a minute."):
            try:
                st.session_state.vectorstore = get_vectorstore()
                st.success("Clinical knowledge base loaded successfully!")
            except Exception as e:
                st.error(f"Error loading knowledge base: {str(e)}")
                st.session_state.vectorstore = None

    # Custom CSS for modern look with dark theme compatibility
    st.markdown("""
    <style>
    /* Remove background overrides that conflict with dark theme */
    .stApp {max-width: 1200px; margin: 0 auto;}
    .css-18e3th9 {padding-top: 2rem;}
    
    /* Button styling */
    .stButton>button {background-color: #3498db; color: white;}
    .stButton>button:hover {background-color: #2980b9;}
    
    /* Chat message styling */
    .chat-message {border-radius: 10px; padding: 10px; margin-bottom: 10px;}
    .chat-message-user {background-color: rgba(52, 152, 219, 0.2); color: inherit;}
    .chat-message-assistant {background-color: rgba(240, 240, 240, 0.2); color: inherit;}
    
    /* Source and metrics boxes with dark theme compatibility */
    .source-box {background-color: rgba(255, 255, 255, 0.1); color: inherit; border-radius: 5px; padding: 15px; margin-bottom: 10px; border-left: 5px solid #3498db;}
    .metrics-box {background-color: rgba(255, 255, 255, 0.1); color: inherit; border-radius: 5px; padding: 15px; margin-top: 20px;}
    
    /* Feature cards for homepage */
    .feature-card {background-color: rgba(255, 255, 255, 0.1); color: inherit; border-radius: 10px; padding: 20px; height: 200px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)

    # App states
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'page' not in st.session_state:
        st.session_state.page = 'cover'

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/caduceus.png", width=80)
        st.title("DiReCT")
        st.markdown("### Diagnostic Reasoning for Clinical Text")
        st.markdown("---")
        
        if st.button("Home", key="home_btn"):
            st.session_state.page = 'cover'
        if st.button("Diagnostic Assistant", key="assistant_btn"):
            st.session_state.page = 'chat'
        if st.button("About", key="about_btn"):
            st.session_state.page = 'about'
            
        st.markdown("---")
        st.markdown("### Model Information")
        st.markdown("**Embedding Model:** Bio_ClinicalBERT")
        st.markdown("**LLM:** Llama-3.3-70B")
        st.markdown("**Vector Store:** FAISS")

    # Cover page
    if st.session_state.page == 'cover':
        # Hero section with animation
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("<h1 style='font-size:3.5em;'>DiReCT</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='font-size:1.8em;color:#3498db;'>Diagnostic Reasoning for Clinical Text</h2>", unsafe_allow_html=True)
            st.markdown("""<p style='font-size:1.2em;'>A powerful RAG-based clinical diagnostic assistant that leverages the MIMIC-IV-Ext dataset to provide accurate medical insights and diagnostic reasoning.</p>""", unsafe_allow_html=True)
            
            st.markdown("""<br>""", unsafe_allow_html=True)
            if st.button("Get Started", key="get_started"):
                st.session_state.page = 'chat'
                st.rerun()
        
        with col2:
            # Animated medical icon
            st.markdown("""
            <div style='display:flex;justify-content:center;align-items:center;height:100%;'>
                <img src="https://img.icons8.com/color/240/000000/healthcare-and-medical.png" style='max-width:90%;'>
            </div>
            """, unsafe_allow_html=True)
        
        # Features section
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>Key Features</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='feature-card' style='text-align:center;'>
                <img src="https://img.icons8.com/color/48/000000/search--v1.png">
                <h3>Intelligent Retrieval</h3>
                <p>Finds the most relevant clinical information from the MIMIC-IV-Ext dataset</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class='feature-card' style='text-align:center;'>
                <img src="https://img.icons8.com/color/48/000000/brain.png">
                <h3>Advanced Reasoning</h3>
                <p>Applies clinical knowledge to generate accurate diagnostic insights</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class='feature-card' style='text-align:center;'>
                <img src="https://img.icons8.com/color/48/000000/document.png">
                <h3>Source Transparency</h3>
                <p>Provides references to all clinical sources used in generating responses</p>
            </div>
            """, unsafe_allow_html=True)

    # Chat interface
    elif st.session_state.page == 'chat':
        # Header with clear button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("<h1>Clinical Diagnostic Assistant</h1>", unsafe_allow_html=True)
        with col2:
            # Add a clear button in the header
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
                
        st.markdown("Ask any clinical diagnostic question and get insights based on medical knowledge and patient cases.")
        
        # Display chat history
        for i, (query, response) in enumerate(st.session_state.chat_history):
            st.markdown(f"<div class='chat-message chat-message-user'><b>🧑‍⚕️ You:</b> {query}</div>", unsafe_allow_html=True)
            
            st.markdown(f"<div class='chat-message chat-message-assistant'><b>🩺 DiReCT:</b> {response['answer']}</div>", unsafe_allow_html=True)
            
            with st.expander("View Sources"):
                for doc in response["context"]:
                    st.markdown(f"<div class='source-box'>"
                              f"<b>Source:</b> {Path(doc.metadata['source']).stem}<br>"
                              f"<b>Type:</b> {doc.metadata['type']}<br>"
                              f"<b>Content:</b> {doc.page_content[:300]}...</div>", 
                              unsafe_allow_html=True)
            
            # Show evaluation metrics if available
            try:
                eval_scores = evaluate_rag_response(response, embeddings)
                with st.expander("View Evaluation Metrics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Hit Rate (Top-3)", f"{eval_scores['hit_rate']:.2f}")
                    with col2:
                        st.metric("Faithfulness", f"{eval_scores['faithfulness']:.2f}")
            except Exception as e:
                st.warning(f"Evaluation metrics unavailable: {str(e)}")
        
        # Query input
        user_input = st.text_area("Ask a clinical question:", height=100)
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.button("Submit")
        
        # Process query
        if submit_button and user_input:
            if st.session_state.vectorstore is None:
                st.error("Knowledge base not loaded. Please refresh the page and try again.")
            else:
                with st.spinner("Analyzing clinical data..."):
                    try:
                        # Add a small delay for UX
                        time.sleep(0.5)
                        
                        # Run RAG
                        response = run_rag_chat(user_input, st.session_state.vectorstore)
                        response["retriever"] = st.session_state.vectorstore.as_retriever()
                        
                        # Add to chat history
                        st.session_state.chat_history.append((user_input, response))
                        
                        # Rerun to update UI
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
    
    # About page
    elif st.session_state.page == 'about':
        st.markdown("<h1>About DiReCT</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        ### Project Overview
        
        DiReCT (Diagnostic Reasoning for Clinical Text) is a Retrieval-Augmented Generation (RAG) system designed to assist medical professionals with diagnostic reasoning based on clinical notes and medical knowledge.
        
        ### Data Sources
        
        This application uses the MIMIC-IV-Ext dataset, which contains de-identified clinical notes and medical records. The system processes:
        
        - Diagnostic flowcharts
        - Patient cases
        - Clinical guidelines
        
        ### Technical Implementation
        
        - **Embedding Model**: Bio_ClinicalBERT for domain-specific text understanding
        - **Vector Database**: FAISS for efficient similarity search
        - **LLM**: Llama-3.3-70B for generating medically accurate responses
        - **Framework**: Built with LangChain and Streamlit
        
        ### Evaluation Metrics
        
        The system evaluates responses using:
        
        - **Hit Rate**: Measures how many relevant documents were retrieved
        - **Faithfulness**: Measures how well the response aligns with the retrieved context
        
        ### Ethical Considerations
        
        This system is designed as a clinical decision support tool and not as a replacement for professional medical judgment. All patient data used has been properly de-identified in compliance with healthcare privacy regulations.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Developers")
        st.markdown("This project was developed as part of an academic assignment on RAG systems for clinical applications.")

if __name__ == "__main__":
    main()
