# EasyChromaDirectories

(ECD) is A lightweight, Mac-optimized implementation of ChromaDB designed for multimodal document embeddings. Built specifically for fast RAG (Retrieval-Augmented Generation) pipelines, this tool seamlessly handles text, images, and mixed-media documents with minimal setup.

> ⚠️ **Note**: Dependencies and requirements packaging are under active development.

## Features

- 🚀 **Optimized for Apple Silicon**: Specifically tested and optimized for M-series chips
- 📊 **Multimodal Embeddings**: Unified handling of text, images, and mixed content
- 💨 **Lightweight RAG Pipeline**: Minimal setup for production-ready document retrieval
- 🔍 **Smart Search**: Cross-modal natural language queries
- 🏷️ **Rich Metadata**: Comprehensive document management
- 🔄 **Duplicate Prevention**: Content-based hash checking
- ⚡ **Fast Processing**: Optimized for quick document ingestion and retrieval

## System Requirements

### Essential Prerequisites
- macOS with Apple Silicon (Tested on M4)
- Xcode Command Line Tools
- Homebrew
- Python 3.8+

### Core Dependencies
- ChromaDB
- SentenceTransformers (Hugging Face)
- Additional dependencies in `requirements.txt`

### Installation Steps

1. Install system requirements:
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Clone and set up the project:
```bash
git clone https://github.com/yourusername/EasyChromaDirectories.git
cd EasyChromaDirectories
pip install -r requirements.txt
```

> 🚧 **Development Notice**: Package management and dependency resolution are being actively improved. Some manual setup might be required.

## Usage

### Python API

#### Initialize Collection

```python
from easychromadb import DocumentEncoder

encoder = DocumentEncoder(collection_name="assets")
```

#### Process Documents

```python
# Process a single document
encoder.process_file("path/to/document.txt")

# Process an entire directory
encoder.process_directory("path/to/documents/")
```

#### Query Documents

```python
results = encoder.query("your search query here")
for result in results:
    print(f"Document: {result.name}")
    print(f"Similarity: {result.score}")
```

### Command Line Interface (CLI)

No Python experience required! Use these simple commands to manage your documents:

#### Process Documents
```bash
# Process a directory of documents
python Chromav4_Encode_documents.py your_directory/

# Example:
python Chromav4_Encode_documents.py assets_ChromaDB_Vec/
```

#### List Documents
```bash
# List all documents in the collection
python Chromav4_Encode_documents.py your_directory/ --list

# Example output:
# Collection: assets
# Total Documents: 6
# +-----+----------------+--------+---------------+
# |   # | ID             | Type   | Name          |
# +=====+================+========+===============+
# |   1 | txt_0_2288d1ca | TEXT   | doc1.txt      |
# |   2 | txt_1_c2ecec13 | TEXT   | doc2.txt      |
# ...
```

#### Search Documents
```bash
# Search with a query and specify number of results
python Chromav4_Encode_documents.py your_directory/ --query "your search query" --n_results 2

# Example:
python Chromav4_Encode_documents.py assets_ChromaDB_Vec/ --query "Why is the sky blue?" --n_results 2
```

#### Advanced Search Features
```bash
# Partial word matching
python Chromav4_Encode_documents.py your_directory/ --query "Who's the _____ uncle" --n_results 1

# Image and text combined search
python Chromav4_Encode_documents.py your_directory/ --query "Find similar images and text about nature"
```

The CLI will automatically:
- Skip duplicate documents
- Process both text and images
- Show similarity scores
- Display document metadata
- Handle partial word matching
- Support natural language queries

## Tests

The project includes comprehensive tests covering:

- ✅ Encoder initialization
- ✅ Document processing
- ✅ Directory scanning
- ✅ Query functionality
- ✅ Metadata management
- ✅ Duplicate detection
- ✅ Error handling

Run tests using:
```bash
pytest test_Chromav4_Encode_documents.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation as needed
- Maintain backward compatibility
- Keep commits atomic and well-described

## Next Steps

### PDF Processing with Apple OCR

- Native integration with macOS Vision framework
- Optimized for Apple Silicon
- Automatic text extraction from PDFs
- Support for multi-page documents
- Quality assessment of extracted text

### Automatic File Renaming

- Content-based intelligent file naming
- Custom naming patterns and rules
- Metadata-driven naming schemas
- Duplicate handling and versioning

### CSV Transaction Matching

- CSV parsing and normalization
- Receipt-to-transaction matching
- Support for multiple CSV formats
- Fuzzy matching algorithms
- Confidence scoring for matches

### Roadmap

- [ ] Streamlined dependency management
- [ ] Pre-built wheels for Apple Silicon
- [ ] Automated installation scripts
- [ ] Performance optimization for larger datasets
- [ ] Extended multimodal support

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ChromaDB team for the excellent vector database
- Hugging Face for SentenceTransformers
- Contributors and testers

---

Built with ❤️ for the document processing community