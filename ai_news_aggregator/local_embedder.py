"""Local model embedder using Ollama."""
import numpy as np
from typing import List, Union
import ollama
from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    def __init__(self, model_name: str = 'nomic-embed-text'):
        """
        Initialize the local embedder.
        
        Args:
            model_name: Name of the Ollama model to use for embeddings
        """
        self.model_name = model_name
        self.model_type = 'ollama'  # Can be 'ollama' or 'sentence-transformers'
        
        # Check if we should use Ollama or fall back to sentence-transformers
        if model_name == 'nomic-embed-text':
            try:
                # Test if Ollama is available
                ollama.pull(model_name)
                self.model_type = 'ollama'
            except Exception as e:
                print(f"Warning: Could not use Ollama, falling back to sentence-transformers. Error: {e}")
                self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1')
                self.model_type = 'sentence-transformers'
        else:
            # For other models, use sentence-transformers
            self.model = SentenceTransformer(model_name)
            self.model_type = 'sentence-transformers'
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the input text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy.ndarray: Embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if self.model_type == 'ollama':
            # Use Ollama for embeddings
            embeddings = []
            for text in texts:
                response = ollama.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                embeddings.append(response['embedding'])
            return np.array(embeddings)
        else:
            # Use sentence-transformers
            return self.model.encode(texts, convert_to_numpy=True)
    
    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Alias for embed() to maintain compatibility with SentenceTransformer"""
        return self.embed(texts)
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Alias for embed() to maintain compatibility with SentenceTransformer"""
        return self.embed(texts)
