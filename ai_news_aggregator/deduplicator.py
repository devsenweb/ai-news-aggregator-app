"""Module for detecting and removing duplicate or near-duplicate articles."""
from typing import List, Dict, Any, Set, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .local_embedder import LocalEmbedder

@dataclass
class Article:
    """Represents a news article with its metadata and content."""
    id: str
    headline: str
    url: str
    content: str
    publication_date: datetime
    source: str
    embedding: np.ndarray = None

class Deduplicator:
    """Detects and removes duplicate or near-duplicate articles."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize the deduplicator.
        
        Args:
            similarity_threshold: Threshold for considering articles as duplicates
        """
        self.similarity_threshold = similarity_threshold
        # Initialize the local embedder
        self.model = LocalEmbedder('nomic-embed-text')
        self.logger = logging.getLogger(__name__)
    
    def deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles from a list.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of unique articles
        """
        if not articles:
            return []
            
        # Convert to Article objects and generate embeddings
        article_objects = []
        for i, article in enumerate(articles):
            try:
                text = f"{article['headline']} {article.get('summary', article.get('content', ''))}"
                embedding = self.model.encode([text], convert_to_tensor=False)[0]
                
                article_obj = Article(
                    id=str(i),
                    headline=article['headline'],
                    url=article['url'],
                    content=article.get('content', ''),
                    publication_date=article.get('publication_date', datetime.utcnow()),
                    source=article.get('source', 'unknown'),
                    embedding=embedding
                )
                article_objects.append(article_obj)
            except Exception as e:
                self.logger.error(f"Error processing article {i}: {e}", exc_info=True)
                continue
        
        if not article_objects:
            return []
        
        # Group articles by time windows (same day)
        time_window = timedelta(days=1)
        article_objects.sort(key=lambda x: x.publication_date)
        
        unique_articles = []
        current_window = []
        current_window_end = article_objects[0].publication_date + time_window
        
        # Process articles in time windows
        for article in article_objects:
            if article.publication_date > current_window_end:
                unique_articles.extend(self._process_window(current_window))
                current_window = [article]
                current_window_end = article.publication_date + time_window
            else:
                current_window.append(article)
        
        # Process the last window
        if current_window:
            unique_articles.extend(self._process_window(current_window))
        
        # Convert back to original format
        return [{
            'headline': a.headline,
            'url': a.url,
            'source': a.source,
            'publication_date': a.publication_date,
            'content': a.content,
            'id': a.id
        } for a in unique_articles]
    
    def _process_window(self, articles: List[Article]) -> List[Article]:
        """Process a time window of articles to find duplicates.
        
        Args:
            articles: List of articles in the same time window
            
        Returns:
            List of unique articles
        """
        if not articles:
            return []
            
        # If only one article in window, keep it
        if len(articles) == 1:
            return articles
            
        # Calculate similarity matrix
        embeddings = np.array([a.embedding for a in articles])
        similarities = cosine_similarity(embeddings)
        
        # Find duplicates
        duplicate_indices = set()
        unique_articles = []
        
        for i in range(len(articles)):
            if i in duplicate_indices:
                continue
                
            # Add current article to unique list
            unique_articles.append(articles[i])
            
            # Find duplicates of the current article
            for j in range(i + 1, len(articles)):
                if j in duplicate_indices:
                    continue
                    
                if similarities[i, j] > self.similarity_threshold:
                    duplicate_indices.add(j)
                    self.logger.debug(
                        f"Found duplicate articles: '{articles[i].headline}' and '{articles[j].headline}' "
                        f"(similarity: {similarities[i, j]:.2f})"
                    )
        
        return unique_articles
