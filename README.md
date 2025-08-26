# HRM - Hierarchical RAG Model

A Python-based hierarchical retrieval-augmented generation (RAG) system using LangChain and Chroma vector database.

## Project Overview

This project implements a hierarchical RAG system that can process and query PDF documents using advanced retrieval techniques. The system uses Chroma as the vector database for document storage and retrieval.

## Project Structure

```
hrm/
├── chroma/                    # Chroma vector database files
│   ├── *.bin                 # Database index files
│   ├── chroma.sqlite3        # SQLite database
│   └── parents.jsonl         # Document hierarchy metadata
├── data/                     # PDF documents for processing
│   └── *.pdf                # Research papers and documents
├── hier_rag_langchain.py    # Main hierarchical RAG implementation
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Features

- **Hierarchical Document Processing**: Organizes documents in a hierarchical structure for better retrieval
- **PDF Document Support**: Processes and indexes PDF research papers
- **Chroma Vector Database**: Uses Chroma for efficient vector storage and similarity search
- **LangChain Integration**: Leverages LangChain for RAG pipeline implementation

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

## Usage

### Basic Setup

1. Place your PDF documents in the `data/` directory
2. Run the main script:
```bash
python hier_rag_langchain.py
```

### Configuration

The system automatically:
- Processes PDF documents from the `data/` directory
- Creates vector embeddings using Chroma
- Builds a hierarchical structure for document organization
- Provides query capabilities for document retrieval

## Dependencies

Key dependencies include:
- `langchain`: RAG pipeline framework
- `chromadb`: Vector database for document storage
- `pypdf`: PDF document processing
- Additional dependencies listed in `requirements.txt`

## Data Management

- **Input**: PDF documents in the `data/` directory
- **Storage**: Chroma vector database in the `chroma/` directory
- **Output**: Hierarchical document structure with query capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license information here]

## Contact

[Add contact information here]
