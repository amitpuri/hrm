# HRM - Hierarchical RAG Model

A Python-based hierarchical retrieval-augmented generation (RAG) system using LangChain and Chroma vector database with support for multiple AI providers.

## Project Overview

This project implements a sophisticated hierarchical RAG system that can process and query PDF documents using advanced retrieval techniques. The system uses Chroma as the vector database for document storage and retrieval, with support for both OpenAI and Azure OpenAI providers.

## Project Structure

```
hrm/
├── chroma/                    # Chroma vector database files
│   ├── *.bin                 # Database index files
│   ├── chroma.sqlite3        # SQLite database
│   └── parents.jsonl         # Document hierarchy metadata
├── data/                     # Documents for processing
│   ├── *.pdf                # Research papers and documents
│   ├── *.md                 # Markdown files
│   └── *.txt                # Text files
├── hier_rag_langchain.py    # Main hierarchical RAG implementation
├── hier_rag_langchain-query.py # Query interface implementation
├── env.example              # Environment configuration template
├── requirements.txt          # Python dependencies
├── readme.txt               # Quick start guide
└── README.md                # This file
```

## Features

- **Hierarchical Document Processing**: Organizes documents in a hierarchical structure for better retrieval
- **Multi-Format Document Support**: Processes PDF, Markdown, and text files
- **Two-Stage Retrieval**: Global summary index narrows search to candidate sections, then fine-grained index retrieves precise spans
- **Multiple AI Provider Support**: Compatible with OpenAI and Azure OpenAI
- **Chroma Vector Database**: Uses Chroma for efficient vector storage and similarity search
- **LangChain Integration**: Leverages LangChain for RAG pipeline implementation
- **Document Compression**: LLM compression + embedding de-duplication via DocumentCompressorPipeline
- **Streaming Responses**: Real-time streaming of AI responses
- **Lightweight Citations**: Clean citation system for source attribution

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hrm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

### Environment Variables

Create a `.env` file based on `env.example`:

**For OpenAI:**
```
PROVIDER=openai
OPENAI_API_KEY=sk-...
```

**For Azure OpenAI:**
```
PROVIDER=azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-large
```

## Usage

### Quick Start

1. Place your documents in the `data/` directory (supports PDF, Markdown, and text files)
2. Build the index:
```bash
python hier_rag_langchain.py --build ./data
```

3. Ask questions:
```bash
python hier_rag_langchain.py --ask "What does the spec say about token limits and rate limiting?"
```

### Advanced Usage

The system provides two main scripts:

- **`hier_rag_langchain.py`**: Main implementation with CLI interface
- **`hier_rag_langchain-query.py`**: Query interface implementation

### Document Processing

The system automatically:
- Processes documents from the `data/` directory
- Creates vector embeddings using the configured AI provider
- Builds a hierarchical structure for document organization
- Provides advanced query capabilities with two-stage retrieval

## Dependencies

Key dependencies include:
- `langchain>=0.2.11`: RAG pipeline framework
- `langchain-community>=0.2.11`: Community integrations
- `langchain-openai>=0.2.5`: OpenAI provider integration
- `langchain-chroma>=0.1.4`: Chroma vector database integration
- `chromadb>=0.5.4`: Vector database for document storage
- `tiktoken`: Token counting and text processing
- `pypdf`: PDF document processing
- `python-dotenv`: Environment variable management

## Data Management

- **Input**: PDF, Markdown, and text documents in the `data/` directory
- **Storage**: Chroma vector database in the `chroma/` directory
- **Output**: Hierarchical document structure with advanced query capabilities

## Architecture

The system implements a sophisticated two-stage retrieval approach:

1. **Global Summary Index**: Narrows search to candidate sections
2. **Fine-Grained Index**: Retrieves precise spans from those sections
3. **Document Compression**: Uses LLM compression and embedding de-duplication
4. **LCEL Answer Chain**: Clean chain with lightweight citations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, support, or contributions, please contact: openagi.news
