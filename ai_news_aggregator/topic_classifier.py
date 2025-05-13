"""Module for classifying news articles into topics."""
from typing import List, Dict, Any, Tuple
from datetime import datetime
import logging
from .local_embedder import LocalEmbedder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TopicClassifier:
    """Classifies news articles into topics using semantic similarity with predefined categories."""
    
    # Predefined categories for better topic grouping
    CATEGORIES = {
        'politics': [
            'election', 'trump', 'biden', 'government', 'congress', 'senate', 'president', 'democrat',
            'republican', 'policy', 'administration', 'white house', 'parliament', 'pm', 'prime minister'
        ],
        'business': [
            'market', 'stocks', 'economy', 'business', 'trade', 'tariff', 'inflation', 'fed', 'bank',
            'finance', 'economic', 'dollar', 'euro', 'stock market', 'bitcoin', 'crypto'
        ],
        'technology': [
            'tech', 'ai', 'artificial intelligence', 'google', 'apple', 'microsoft', 'amazon', 'facebook',
            'meta', 'tesla', 'spacex', 'elon musk', 'startup', 'app', 'software', 'hardware', 'gadget',
            'smartphone', 'computer', 'internet'
        ],
        'sports': [
            'sports', 'football', 'soccer', 'nfl', 'nba', 'mlb', 'nhl', 'tennis', 'golf', 'olympics',
            'champions league', 'premier league', 'world cup', 'ncaa', 'basketball', 'baseball', 'hockey'
        ],
        'entertainment': [
            'movie', 'film', 'hollywood', 'actor', 'actress', 'oscar', 'emmy', 'netflix', 'disney',
            'music', 'album', 'song', 'billboard', 'award', 'grammy', 'celebrity', 'tv show', 'series'
        ],
        'world': [
            'russia', 'ukraine', 'china', 'india', 'pakistan', 'europe', 'asia', 'middle east', 'united nations',
            'nato', 'eu', 'brexit', 'un', 'war', 'conflict', 'treaty', 'alliance'
        ]
    }
    
    def __init__(self, similarity_threshold: float = 0.75, max_topics: int = 5):
        """Initialize the topic classifier.
        
        Args:
            similarity_threshold: Threshold for considering articles as part of the same topic
            max_topics: Maximum number of topics to maintain per category
        """
        self.similarity_threshold = similarity_threshold
        self.max_topics_per_category = max_topics
        # Initialize the local embedder
        self.model = LocalEmbedder('nomic-embed-text')
        # Dictionary to store topics by category
        self.topics = {category: [] for category in self.CATEGORIES.keys()}
        self.logger = logging.getLogger(__name__)
    
    def _categorize_article(self, article: Dict[str, Any]) -> str:
        """Categorize an article into one of the predefined categories."""
        text = f"{article['headline']} {article.get('summary', '')}".lower()
        
        # Calculate category scores
        category_scores = {}
        for category, keywords in self.CATEGORIES.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or 'general' if no match
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    def _get_or_create_topic(self, article: Dict[str, Any], category: str) -> str:
        """Get or create a topic in the given category."""
        # Generate embedding for the article
        text = f"{article['headline']} {article.get('summary', '')}"
        article_embedding = self.model.encode([text], convert_to_tensor=False)
        current_time = datetime.utcnow()
        
        # Get topics for this category
        category_topics = self.topics[category]
        
        # If no topics in category yet, create a new one
        if not category_topics:
            new_topic = article['headline']
            self.topics[category].append((new_topic, article_embedding, current_time))
            return new_topic
        
        # Calculate similarity with existing topics in the same category
        topic_embeddings = np.array([topic[1] for topic in category_topics])
        similarities = cosine_similarity(
            article_embedding.reshape(1, -1),
            topic_embeddings.reshape(len(category_topics), -1)
        )[0]
        
        # Find the most similar topic in this category
        max_sim_idx = np.argmax(similarities)
        max_similarity = similarities[max_sim_idx]
        
        # If similarity is above threshold, update the topic
        if max_similarity > self.similarity_threshold:
            topic_title = category_topics[max_sim_idx][0]
            # Update the topic's last_updated time
            self.topics[category][max_sim_idx] = (topic_title, article_embedding, current_time)
            return topic_title
        
        # If we reach max topics for this category, replace the oldest topic
        if len(category_topics) >= self.max_topics_per_category:
            # Find the oldest topic
            oldest_idx = np.argmin([topic[2] for topic in category_topics])
            self.topics[category][oldest_idx] = (article['headline'], article_embedding, current_time)
            return article['headline']
        
        # Otherwise, add a new topic to this category
        new_topic = article['headline']
        self.topics[category].append((new_topic, article_embedding, current_time))
        return new_topic
    
    def classify_article(self, article: Dict[str, Any]) -> Tuple[str, str]:
        """Classify an article into a topic and return both category and topic.
        
        Args:
            article: Article data including 'headline' and 'summary'
            
        Returns:
            tuple: (category, topic_name)
        """
        try:
            text = f"{article.get('headline', '')} {article.get('summary', '')}"
            
            # Check for category matches first
            for category, keywords in self.CATEGORIES.items():
                if any(keyword.lower() in text.lower() for keyword in keywords):
                    # Get or create a topic in this category
                    topic = self._get_or_create_topic(article, category)
                    return (category, topic)
            
            # If no category match, use 'general' category
            topic = self._get_or_create_topic(article, 'general')
            return ('general', topic)
            
        except Exception as e:
            self.logger.error(f"Error classifying article: {e}", exc_info=True)
            return ('general', 'general')
            self.topics[category][max_sim_idx] = (topic_title, category_topics[max_sim_idx][1], current_time)
            return f"{category.upper()}: {topic_title}", False
        
        # If we've reached max topics in this category, replace the oldest
        if len(category_topics) >= self.max_topics_per_category:
            oldest_idx = min(range(len(category_topics)), key=lambda i: category_topics[i][2])
            removed_topic = category_topics[oldest_idx][0]
            self.logger.info(f"Reached max topics in {category}. Replacing oldest topic: {removed_topic}")
            new_topic = article['headline']
            self.topics[category][oldest_idx] = (new_topic, article_embedding, current_time)
            return f"{category.upper()}: {new_topic}", True
        
        # Otherwise, create a new topic in this category
        new_topic = article['headline']
        self.topics[category].append((new_topic, article_embedding, current_time))
        return f"{category.upper()}: {new_topic}", True
    
    def get_all_topics(self) -> List[Tuple[str, str, datetime]]:
        """Get a list of all topics with their categories and update times.
        
        Returns:
            List of (category, topic_title, last_updated) tuples, sorted by last_updated (newest first)
        """
        all_topics = []
        for category, topics in self.topics.items():
            for topic in topics:
                all_topics.append((category, topic[0], topic[2]))
        
        # Sort by last_updated (newest first)
        return sorted(all_topics, key=lambda x: x[2], reverse=True)
