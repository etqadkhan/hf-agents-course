# GAIA Agent Evaluation System

A comprehensive AI agent system built for the GAIA (General AI Assistant) evaluation framework. This project implements a multi-modal AI agent capable of processing various types of questions and media files using advanced language models and specialized tools.

## 🚀 Features

### Core Capabilities
- **Multi-modal Processing**: Analyze images, transcribe audio, process videos, and handle various file formats
- **Web Research**: Search Wikipedia, general web content, and academic papers via ArXiv
- **Mathematical Operations**: Perform calculations and logical operations
- **File Analysis**: Process Excel files, Python code execution, and document analysis
- **Vector Search**: Similar question retrieval using ChromaDB vector store

### Supported File Types
- **Images**: PNG, JPG, JPEG (detailed analysis including chess positions, diagrams, charts)
- **Audio**: MP3, WAV (complete transcription with speaker context)
- **Videos**: YouTube video analysis with dialogue and visual element extraction
- **Documents**: Excel files (comprehensive data analysis with calculations)
- **Code**: Python file execution with output capture

### AI Models & Tools
- **Language Model**: Google Vertex AI Gemini 2.5 Pro
- **Embeddings**: HuggingFace sentence transformers
- **Search**: Tavily web search, Wikipedia, ArXiv
- **Vector Store**: ChromaDB for similarity search
- **Framework**: LangGraph for agent orchestration

## 🏗️ Architecture

The system is built using a modular architecture:

```
├── agent.py          # Core agent implementation with tools and graph
├── app.py            # Gradio web interface and evaluation runner
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

### Agent Components
1. **Tool Collection**: 15+ specialized tools for different tasks
2. **Graph Builder**: LangGraph-based agent workflow
3. **Vector Store**: ChromaDB for question similarity matching
4. **Evaluation System**: Automated question fetching and scoring

## 🛠️ Setup

### Prerequisites
- Python 3.11+
- Google Cloud Platform account (for Vertex AI)
- HuggingFace account (for models and embeddings)
- Tavily API key (for web search)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hf-agents-course
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file with the following variables:
   ```env
   # Google Cloud Configuration
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_LOCATION=your-location
   GEMINI_MODEL=your-model-name
   
   # HuggingFace Configuration
   HUGGINGFACE_API_TOKEN=your-hf-token
   HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
   
   # Tavily Search
   TAVILY_API_KEY=your-tavily-key
   
   # Optional: For HuggingFace Spaces
   HF_USERNAME=your-username
   SPACE_ID=your-space-id
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

## 🎯 Usage

### Web Interface
1. Launch the application using `python app.py`
2. Open your browser to the provided local URL
3. Click "Run Evaluation & Submit All Answers" to start the evaluation process
4. Monitor the progress and view results in the interface

### Agent Capabilities

The agent can handle various question types:

- **Mathematical Problems**: Basic arithmetic, complex calculations
- **Research Questions**: Web search, Wikipedia lookups, academic papers
- **Media Analysis**: Image description, audio transcription, video analysis
- **Data Processing**: Excel file analysis, Python code execution
- **Logical Operations**: Commutativity checks, text manipulation

### Example Use Cases

1. **Chess Position Analysis**: Upload a chess board image for move suggestions
2. **Audio Transcription**: Process audio files for complete text extraction
3. **Data Analysis**: Analyze Excel files with comprehensive statistics
4. **Code Execution**: Run Python scripts and capture outputs
5. **Research Tasks**: Search multiple sources for comprehensive answers

## 🔧 Configuration

### Model Settings
- **Temperature**: Set to 0 for deterministic responses
- **Max Tokens**: Configured for comprehensive answers
- **Tool Selection**: Automatic based on question type

### Vector Store
- **Embedding Model**: Configurable via environment variable
- **Persistence**: ChromaDB with local storage
- **Similarity Search**: Top-k retrieval for similar questions

## 📊 Evaluation System

The system includes an automated evaluation framework:

1. **Question Fetching**: Retrieves questions from evaluation API
2. **File Download**: Automatically downloads associated media files
3. **Agent Processing**: Runs questions through the agent pipeline
4. **Answer Submission**: Submits responses for scoring
5. **Results Display**: Shows performance metrics and detailed logs

## 🚀 Deployment

### HuggingFace Spaces
The application is configured for HuggingFace Spaces deployment:

- **SDK**: Gradio 5.25.2
- **Hardware**: GPU recommended for optimal performance
- **Environment**: Automatic dependency installation
- **OAuth**: Configured for secure access

### Local Development
For local development, ensure all environment variables are properly set and the virtual environment is activated.

## 📝 License

This project is part of an AI agents course and is intended for educational purposes.

## 🤝 Contributing

This is a course project. For questions or issues, please refer to the course materials or contact the instructor.

## 📚 Dependencies

Key dependencies include:
- `gradio`: Web interface framework
- `langchain`: LLM application framework
- `langgraph`: Agent orchestration
- `chromadb`: Vector database
- `google-cloud-aiplatform`: Vertex AI integration
- `pandas`: Data processing
- `pillow`: Image processing
- `openpyxl`: Excel file handling

For the complete list, see `requirements.txt`.

---

**Note**: This system requires proper API keys and cloud credentials to function. Ensure all environment variables are configured before running the application.