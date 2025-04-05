# DiReCT: Diagnostic Reasoning for Clinical Text

![Clinical AI](https://img.shields.io/badge/Clinical-AI-blue)
![RAG](https://img.shields.io/badge/RAG-System-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Python](https://img.shields.io/badge/Python-3.9+-yellow)

<p align="center">
  <img src="https://img.icons8.com/color/96/000000/caduceus.png" alt="DiReCT Logo" width="120"/>
</p>

## Overview

DiReCT (Diagnostic Reasoning for Clinical Text) is a Retrieval-Augmented Generation (RAG) system designed to assist healthcare professionals with diagnostic reasoning based on clinical notes. The system leverages the MIMIC-IV-Ext dataset to provide evidence-based insights and recommendations for clinical queries.

By combining state-of-the-art language models with domain-specific clinical knowledge, DiReCT aims to enhance diagnostic accuracy and efficiency in clinical settings.

## Features

- **Intelligent Retrieval**: Finds the most relevant clinical information from the MIMIC-IV-Ext dataset using domain-specific embeddings
- **Advanced Reasoning**: Applies clinical knowledge to generate accurate diagnostic insights using large language models
- **Source Transparency**: Provides references to all clinical sources used in generating responses
- **Performance Metrics**: Evaluates responses using hit rate and faithfulness metrics
- **User-Friendly Interface**: Modern Streamlit interface with dark/light theme compatibility for easy interaction with the system
- **Persistent Vectorstore**: Optimized performance with disk-cached vectorstore that persists across deployments
- **Clear Chat Functionality**: Simple one-click option to clear conversation history

## System Architecture

DiReCT is built on a three-component architecture:

1. **Retrieval Component**: Uses Bio_ClinicalBERT embeddings and FAISS vector database to find relevant clinical documents
2. **Generation Component**: Leverages Llama-3.3-70B through Groq to generate accurate and coherent responses
3. **Evaluation Component**: Measures the quality of responses through hit rate and faithfulness metrics

## Installation

### Prerequisites

- Python 3.9+
- Groq API key (for LLM access)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RAG-for-Diagnostic-Reasoning-for-Clinical-Notes-DiReCT-.git
   cd RAG-for-Diagnostic-Reasoning-for-Clinical-Notes-DiReCT-
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv rag
   # On Windows
   .\rag\Scripts\activate
   # On Unix or MacOS
   source rag/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your API keys:
   - For local development: Create a `.env` file with your API keys:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```
   - For Hugging Face Spaces deployment: Add the API key to Spaces secrets

5. Prepare the MIMIC-IV-Ext dataset:
   - Place diagnostic flowcharts in the `Diagnosis_flowchart` directory
   - Place patient cases in the `Finished` directory

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Navigate to the provided URL (typically http://localhost:8501)

3. Use the interface to:
   - Ask clinical diagnostic questions
   - View retrieved sources
   - Examine evaluation metrics for responses

## Data Structure

The system expects data in the following structure:

```
.
├── Diagnosis_flowchart/
│   ├── flowchart1.json
│   ├── flowchart2.json
│   └── ...
└── Finished/
    ├── Category1/
    │   ├── case1.json
    │   └── ...
    ├── Category2/
    │   ├── case1.json
    │   └── ...
    └── ...
```

## Evaluation Metrics

DiReCT evaluates its responses using two key metrics:

- **Hit Rate**: Measures how many relevant documents were retrieved (higher is better)
- **Faithfulness**: Measures how well the response aligns with the retrieved context (higher is better)

## Ethical Considerations

This system is designed as a clinical decision support tool and not as a replacement for professional medical judgment. All patient data used has been properly de-identified in compliance with healthcare privacy regulations.

## Technical Details

- **Embedding Model**: Bio_ClinicalBERT for domain-specific text understanding
- **Vector Database**: FAISS for efficient similarity search with disk persistence
- **LLM**: Llama-3.3-70B for generating medically accurate responses
- **Framework**: Built with LangChain and Streamlit
- **Deployment**: Optimized for Hugging Face Spaces with secure API key management

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MIMIC-IV-Ext dataset providers
- Groq for LLM API access
- HuggingFace for providing Bio_ClinicalBERT embeddings

## Citation

If you use DiReCT in your research, please cite:

```
@software{direct_rag_2025,
  author = {Waghib Ahmad},
  title = {DiReCT: Diagnostic Reasoning for Clinical Text},
  year = {2025},
  url = {https://github.com/Waghib/RAG-for-Diagnostic-Reasoning-for-Clinical-Notes-DiReCT-}
}
```
