import pytest
from unittest.mock import Mock, patch, mock_open
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from Chromav4_Encode_documents import DocumentEncoder, MyEmbeddingFunction
from PIL import Image

# Test data from TryChromav3.py
DOC1 = """
Blue light is scattered in all directions by the tiny molecules of air in Earth's atmosphere. 
Blue is scattered more than other colors because it travels as shorter, smaller waves. This is why we see a blue sky most of the time. 
Closer to the horizon, the sky fades to a lighter blue or white.
"""

DOC2 = """
When choosing colors, you can consider the following factors:
Color theory: Understand how colors work together and how they can evoke different reactions. 
Color psychology: Consider how colors affect emotions, behaviors, and responses. 
Brand identity: Colors can convey meaning and information about a brand. 
Mood: Consider the mood you want to create. For example, brighter colors can feel cheerful, while cooler colors can be calming.
Space: Consider the size of the space and the amount of natural light it receives. Dark colors can make a room feel smaller, while light colors can make it feel larger.
Color wheel: Use the color wheel to identify primary, secondary, and tertiary colors. 
Color combinations: Decide how to best complement your preferred color with others. 
Color palette: Limit your color palette to a main color and one or two additional colors. 
60-30-10 rule: Use a primary color 60% of the time, a secondary color 30% of the time, and an accent color 10% of the time
"""

@pytest.fixture(scope="session")
def embedding_model():
    """Create a single instance of MyEmbeddingFunction to be reused across all tests."""
    from Chromav4_Encode_documents import MyEmbeddingFunction
    model = MyEmbeddingFunction()
    return model

@pytest.fixture
def mock_collection():
    collection = Mock()
    # Setup default return value for get method
    collection.get.return_value = {'ids': []}
    collection.name = "test_collection"
    return collection

@pytest.fixture
def mock_client(mock_collection):
    client = Mock()
    client.get_or_create_collection.return_value = mock_collection
    return client

@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary test directory with sample files."""
    # Create main directory
    test_dir = tmp_path / "test_documents"
    test_dir.mkdir()
    
    # Create assets directory
    assets_dir = test_dir / "assets"
    assets_dir.mkdir()
    
    # Create test files
    text_file = test_dir / "test.txt"
    text_file.write_text(DOC1)
    
    # Create nested directory with more files
    nested_dir = test_dir / "nested"
    nested_dir.mkdir()
    nested_text = nested_dir / "nested.txt"
    nested_text.write_text(DOC2)
    
    # Create valid image files in assets directory
    img1_path = assets_dir / "img1.png"
    img2_path = assets_dir / "img2.png"

    # Create simple images
    img1 = Image.new('RGB', (10, 10), color='red')
    img2 = Image.new('RGB', (10, 10), color='blue')

    # Save images to the paths
    img1.save(img1_path)
    img2.save(img2_path)
    
    return test_dir

@pytest.fixture
def mock_embedding_function():
    """Create a mock embedding function that returns fixed embeddings."""
    mock_func = Mock()
    mock_func.return_value = np.random.rand(1, 768)  # Smaller dimension for testing
    return mock_func

@pytest.fixture
def mock_transformer():
    """Create a mock SentenceTransformer."""
    with patch('Chromav4_Encode_documents.SentenceTransformer') as mock:
        mock.return_value.encode.return_value = np.random.rand(1, 768)
        yield mock

class TestDocumentEncoder:
    def test_init(self, test_dir, embedding_model, mock_client):
        """Test initialization of DocumentEncoder"""
        with patch('Chromav4_Encode_documents.chromadb.PersistentClient', return_value=mock_client), \
             patch('Chromav4_Encode_documents.MyEmbeddingFunction', return_value=embedding_model):
            from Chromav4_Encode_documents import DocumentEncoder
            encoder = DocumentEncoder(str(test_dir), rename_dir=True)
            expected_path = test_dir.parent / f"{test_dir.name}_ChromaDB_Vec"
            assert encoder.assets_dir == expected_path

    def test_process_directory(self, test_dir, embedding_model, mock_client):
        """Test directory processing"""
        with patch('Chromav4_Encode_documents.chromadb.PersistentClient', return_value=mock_client), \
             patch('Chromav4_Encode_documents.MyEmbeddingFunction', return_value=embedding_model):
            encoder = DocumentEncoder(str(test_dir), rename_dir=False)
            encoder.process_text_documents = Mock()
            encoder.process_image_documents = Mock()
            encoder.process_directory()
            assert encoder.process_text_documents.called
            assert encoder.process_image_documents.called

    def test_process_text_documents(self, test_dir, embedding_model, mock_client):
        """Test text document processing"""
        with patch('Chromav4_Encode_documents.chromadb.PersistentClient', return_value=mock_client), \
             patch('Chromav4_Encode_documents.MyEmbeddingFunction', return_value=embedding_model):
            encoder = DocumentEncoder(str(test_dir), rename_dir=False)
            encoder.process_text_documents([DOC1])
            assert mock_client.get_or_create_collection.return_value.add.called

    def test_process_image_documents(self, test_dir, embedding_model, mock_client):
        """Test image document processing"""
        with patch('Chromav4_Encode_documents.chromadb.PersistentClient', return_value=mock_client), \
             patch('Chromav4_Encode_documents.MyEmbeddingFunction', return_value=embedding_model):
            encoder = DocumentEncoder(str(test_dir), rename_dir=False)
            image_paths = [str(test_dir / "assets" / "img1.png")]
            encoder.process_image_documents(image_paths)
            assert mock_client.get_or_create_collection.return_value.add.called

    def test_list_documents(self, test_dir, mock_client):
        """Test document listing"""
        with patch('Chromav4_Encode_documents.chromadb.PersistentClient', return_value=mock_client):
            encoder = DocumentEncoder(str(test_dir), load_model=False, rename_dir=False)
            # Mock collection.get() to return sample data
            mock_client.get_or_create_collection.return_value.get.return_value = {
                'documents': ['Sample text content', 'Another text content'],
                'metadatas': [
                    {'type': 'text', 'source': '/path/to/test.txt', 'name': 'test.txt', 'date_encoded': '2023-10-15T12:34:56'},
                    {'type': 'text', 'source': '/path/to/nested/nested.txt', 'name': 'nested.txt', 'date_encoded': '2023-10-15T12:35:01'}
                ],
                'ids': ['txt_0_abc12345', 'txt_1_def67890']
            }
            encoder.list_documents()
            # You can add assertions here to check the output if needed

class TestMyEmbeddingFunction:
    def test_init(self, embedding_model):
        """Test initialization of MyEmbeddingFunction"""
        assert embedding_model.model.max_seq_length == 1024

    def test_call_with_text(self, embedding_model):
        """Test embedding function with text input"""
        result = embedding_model([DOC1])
        assert isinstance(result, list)
        assert isinstance(result[0], np.ndarray)
        assert result[0].shape == (12288,)

    def test_call_with_image(self, embedding_model):
        """Test embedding function with image input"""
        result = embedding_model([{"image_path": "assets/img1.png"}])
        assert isinstance(result, list)
        assert isinstance(result[0], np.ndarray)
        assert result[0].shape == (12288,)

    def test_call_with_invalid_input(self, embedding_model):
        """Test embedding function with invalid input"""
        with pytest.raises(ValueError):
            embedding_model([123])

def test_command_line_interface(test_dir):
    """Test the command line interface"""
    from Chromav4_Encode_documents import main
    import sys
    
    # Test normal processing
    sys.argv = ['script', str(test_dir)]
    with patch('Chromav4_Encode_documents.DocumentEncoder') as mock_encoder:
        main()
        assert mock_encoder.called
        mock_encoder.assert_called_with(str(test_dir))
        mock_encoder.return_value.process_directory.assert_called_once()
    
    # Test list flag
    sys.argv = ['script', str(test_dir), '--list']
    with patch('Chromav4_Encode_documents.DocumentEncoder') as mock_encoder:
        main()
        assert mock_encoder.called
        mock_encoder.assert_called_with(str(test_dir))
        mock_encoder.return_value.list_documents.assert_called_once()

def test_format_query_results(capsys, test_dir, embedding_model, mock_client):
    """Test query results formatting"""
    with patch('Chromav4_Encode_documents.chromadb.PersistentClient', return_value=mock_client), \
         patch('Chromav4_Encode_documents.MyEmbeddingFunction', return_value=embedding_model):
        encoder = DocumentEncoder(str(test_dir), rename_dir=False)
        test_results = {
            'documents': [['doc1', 'img1']],
            'metadatas': [[{'type': 'text', 'source': 'test_source'}, {'type': 'image', 'source': 'test_image'}]],
            'distances': [[0.5, 0.7]]
        }
        encoder.format_query_results(test_results)
        captured = capsys.readouterr()
        assert "Query Results" in captured.out
        assert "Result Set 1" in captured.out
        assert "Text Document" in captured.out
        assert "Image Document" in captured.out 