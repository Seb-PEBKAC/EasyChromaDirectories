import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import hashlib
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import shutil
from datetime import datetime
from tabulate import tabulate

class DocumentEncoder:
    def __init__(self, assets_dir: str, load_model: bool = True, rename_dir: bool = True):
        """Initialize the document encoder with ChromaDB settings.
        
        Args:
            assets_dir: Path to the directory containing documents to process
            load_model: Whether to load the embedding model
            rename_dir: Whether to rename the assets directory
        """
        self.assets_dir = Path(assets_dir)
        if not self.assets_dir.exists():
            raise ValueError(f"Directory not found: {assets_dir}")
        
        # Rename directory if not already renamed
        if rename_dir and not self.assets_dir.name.endswith("_ChromaDB_Vec"):
            new_name = f"{self.assets_dir.name}_ChromaDB_Vec"
            new_path = self.assets_dir.parent / new_name
            self.assets_dir.rename(new_path)
            self.assets_dir = new_path
            print(f"Renamed directory to: {new_name}")

        self.db_path = self.assets_dir / "chroma_db"
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        collection_name = self.assets_dir.name.replace("_ChromaDB_Vec", "")
        
        # Only load the embedding function if needed
        if load_model:
            self.embedding_function = MyEmbeddingFunction()
            self.collection = self.client.get_or_create_collection(
                collection_name,
                embedding_function=self.embedding_function
            )
        else:
            self.collection = self.client.get_or_create_collection(collection_name)
        
        print(f"Initialized encoder for collection: {collection_name}")
        print(f"ChromaDB path: {self.db_path}")

    def list_documents(self):
        """List documents in the collection."""
        print(f"\nCollection: {self.collection.name}")
        total_docs = self.collection.count()
        print(f"Total Documents: {total_docs}\n")

        # Fetch all documents
        results = self.collection.get()
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        ids = results.get('ids', [])

        # Prepare table data
        table_data = []
        for idx, (doc_id, metadata) in enumerate(zip(ids, metadatas), start=1):
            doc_type = metadata.get('type', 'N/A').upper()
            source = metadata.get('source', 'N/A')
            name = metadata.get('name', 'N/A')
            date_encoded = metadata.get('date_encoded', 'N/A')
            table_data.append([idx, doc_id, doc_type, source, name, date_encoded])

        headers = ['#', 'ID', 'Type', 'Source', 'Name', 'Date Encoded']
        table = tabulate(table_data, headers=headers, tablefmt='grid')
        print("Document List:")
        print(table)

    def process_directory(self):
        """Process all documents in the directory."""
        print("Processing directory...")

        # Collect text document paths
        text_documents = list(self.assets_dir.glob("*.txt"))
        print(f"Found {len(text_documents)} text documents.")

        # Process text documents
        self.process_text_documents(text_documents)

        # Collect image document paths
        image_documents = list(self.assets_dir.glob("*.png")) + list(self.assets_dir.glob("*.jpeg"))
        print(f"Found {len(image_documents)} image documents.")

        # Process image documents
        self.process_image_documents(image_documents)

    def process_text_documents(self, text_files: List[Path]):
        """Process text documents and add them to the collection."""
        print("Processing text documents...")
        new_texts = []
        metadatas = []
        ids = []

        for idx, file_path in enumerate(text_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Generate hash for the document content
            doc_hash = self._get_document_hash(content)
            # Check if the document already exists
            if not self._document_exists(doc_hash):
                new_texts.append(content)
                # Include file name and path in metadata
                metadata = {
                    'type': 'text',
                    'source': str(file_path),
                    'name': file_path.name,
                    'date_encoded': datetime.now().isoformat(),
                    'hash': doc_hash
                }
                metadatas.append(metadata)
                # Generate a unique ID
                ids.append(f"txt_{idx}_{doc_hash[:8]}")
            else:
                print(f"Document {file_path.name} already exists. Skipping.")

        if new_texts:
            embeddings = self.embedding_function(new_texts)
            self.collection.add(
                documents=new_texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            print(f"Added {len(new_texts)} new text documents.")
        else:
            print("No new text documents to add.")

    def process_image_documents(self, image_paths: List[Path]) -> None:
        """Process and add image documents to the collection if they don't exist."""
        print("Processing image documents...")
        new_images = []
        new_ids = []
        new_metadata = []

        for idx, path in enumerate(image_paths):
            if not path.exists():
                print(f"Warning: Image file not found at {path}")
                continue

            img_hash = self._get_image_hash(path)
            if not self._document_exists(img_hash):
                new_images.append({"image_path": str(path)})
                new_ids.append(f"img_{idx}_{img_hash[:8]}")
                new_metadata.append({
                    "type": "image",
                    "source": str(path),
                    "name": path.name,
                    "date_encoded": datetime.now().isoformat(),
                    "hash": img_hash
                })

        if new_images:
            embeddings = self.embedding_function(new_images)
            self.collection.add(
                ids=new_ids,
                metadatas=new_metadata,
                embeddings=embeddings,
                documents=None  # Exclude documents for images
            )
            print(f"Added {len(new_images)} new images.")
        else:
            print("No new images to add.")

    def query_similar(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the collection for similar documents."""
        query_embedding = self.embedding_function([query_text])
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["metadatas", "distances", "documents"]
        )
        return results

    def _get_document_hash(self, content: str) -> str:
        """Generate a hash for a document to check for duplicates."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_image_hash(self, image_path: str) -> str:
        """Generate a hash for an image file to check for duplicates."""
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _document_exists(self, doc_hash: str) -> bool:
        """Check if a document with the given hash exists in the collection."""
        results = self.collection.get(
            where={"hash": doc_hash},
            limit=1
        )
        return len(results.get('ids', [])) > 0

    def format_query_results(self, results):
        """Format query results in a human-readable way."""
        print("\n=== Query Results ===\n")
        for i, (docs, metas, dists) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ), start=1):
            print(f"Result {i}:")
            print("-" * 50)
            for doc, meta, dist in zip(docs, metas, dists):
                doc_type = meta.get('type', 'N/A').upper()
                name = meta.get('name', 'N/A')
                source = meta.get('source', 'N/A')
                date_encoded = meta.get('date_encoded', 'N/A')
                similarity = dist
                print(f"Type: {doc_type}")
                print(f"Name: {name}")
                print(f"Source: {source}")
                print(f"Date Encoded: {date_encoded}")
                print(f"Similarity Score: {similarity:.4f}")
                if doc_type == 'TEXT':
                    print(f"Content:\n{doc}\n")
                elif doc_type == 'IMAGE':
                    print(f"Image Path: {doc}\n")
            print("-" * 50 + "\n")

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="infgrad/jasper_en_vision_language_v1", use_gpu=False):
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device="cpu" if not use_gpu else "cuda",
            model_kwargs={
                "torch_dtype": torch.bfloat16 if use_gpu else torch.float32,
                "attn_implementation": "sdpa"
            },
            config_kwargs={"is_text_encoder": False, "vector_dim": 12288},
        )
        self.model.max_seq_length = 1024

    def __call__(self, inputs: Documents) -> Embeddings:
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(inputs[0], str):
            return self.model.encode(inputs)
        elif isinstance(inputs[0], dict) and "image_path" in inputs[0]:
            formatted_input = [[{"type": "image_path", "content": item["image_path"]}] for item in inputs]
            return self.model.encode(formatted_input)
        else:
            raise ValueError("Unsupported document type for embedding")

def main():
    parser = argparse.ArgumentParser(description="Process and manage documents with ChromaDB.")
    parser.add_argument("assets_dir", help="Path to the directory containing documents to process")
    parser.add_argument("--list", help="List documents in the collection", action="store_true")
    parser.add_argument("--query", help="Query text to search for similar documents", type=str)
    parser.add_argument("--n_results", help="Number of results to return", type=int, default=5)
    args = parser.parse_args()

    # Determine if the embedding model needs to be loaded
    load_model = bool(args.query) or not args.list

    encoder = DocumentEncoder(args.assets_dir, load_model=load_model)

    if args.list:
        encoder.list_documents()
    elif args.query:
        results = encoder.query_similar(args.query, n_results=args.n_results)
        encoder.format_query_results(results)
    else:
        encoder.process_directory()

if __name__ == "__main__":
    exit(main())