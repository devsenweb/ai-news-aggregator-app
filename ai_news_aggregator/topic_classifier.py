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
    # Commented out other categories for testing
    CATEGORIES = {
        'politics': [
            'election', 'trump', 'biden', 'government', 'congress', 'senate', 'president', 'democrat',
            'republican', 'policy', 'administration', 'white house', 'parliament', 'pm', 'prime minister',
            'election', 'vote', 'senator', 'representative', 'supreme court', 'congressional', 'political'
        ],
        'sports': [
            'sports', 'football', 'soccer', 'nfl', 'nba', 'mlb', 'nhl', 'tennis', 'golf', 'olympics',
            'champions league', 'premier league', 'world cup', 'ncaa', 'basketball', 'baseball', 'hockey',
            'game', 'match', 'tournament', 'championship', 'playoff', 'super bowl', 'world series', 'stanley cup'
        ]
        # 'business': [
        #     'market', 'stocks', 'economy', 'business', 'trade', 'tariff', 'inflation', 'fed', 'bank',
        #     'finance', 'economic', 'dollar', 'euro', 'stock market', 'bitcoin', 'crypto'
        # ],
        # 'technology': [
        #     'tech', 'ai', 'artificial intelligence', 'google', 'apple', 'microsoft', 'amazon', 'facebook',
        #     'meta', 'tesla', 'spacex', 'elon musk', 'startup', 'app', 'software', 'hardware', 'gadget',
        #     'smartphone', 'computer', 'internet'
        # ],
        # 'entertainment': [
        #     'movie', 'film', 'hollywood', 'actor', 'actress', 'oscar', 'emmy', 'netflix', 'disney',
        #     'music', 'album', 'song', 'billboard', 'award', 'grammy', 'celebrity', 'tv show', 'series'
        # ],
        # 'world': [
        #     'russia', 'ukraine', 'china', 'india', 'pakistan', 'europe', 'asia', 'middle east', 'united nations',
        #     'nato', 'eu', 'brexit', 'un', 'war', 'conflict', 'treaty', 'alliance'
        # ]
    }
    
    def __init__(self, similarity_threshold: float = 0.75, max_topics: int = 5):
        """Initialize the topic classifier.
        
        Args:
            similarity_threshold: Threshold for considering articles as part of the same topic
            max_topics: Maximum number of topics to maintain per category (set to 5 for testing)
        """
        self.similarity_threshold = similarity_threshold
        self.max_topics_per_category = 5  # Hardcoded to 5 as requested
        
        # Initialize the local embedder
        self.model = LocalEmbedder('nomic-embed-text')
        
        # Dictionary to store topics by category
        self.topics = {category: [] for category in self.CATEGORIES.keys()}
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized TopicClassifier with categories: {list(self.CATEGORIES.keys())}")
        self.logger.info(f"Max topics per category: {self.max_topics_per_category}")
    
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
    
    def _extract_topic_name(self, article: Dict[str, Any], category: str) -> str:
        """Extract a meaningful topic name from the article content.
        
        Args:
            article: Article data including 'headline' and 'summary'
            category: The category the article belongs to
            
        Returns:
            str: A meaningful topic name
        """
        import re
        from collections import Counter
        
        # Get text content
        headline = article.get('headline', '')
        summary = article.get('summary', '')
        text = f"{headline} {summary}"
        
        # Common words to exclude
        common_words = {
            'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'a', 'an', 
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'that', 'this', 'with', 'as', 'by', 'it', 'its', 'over', 'from', 'says'
        }
        
        # Try to extract key phrases from the headline first
        headline_words = [word.lower() for word in headline.split() if len(word) > 3 and word.lower() not in common_words]
        
        # If we have good words in the headline, use them
        if len(headline_words) >= 2:
            # Try to find proper nouns (capitalized words) first
            proper_nouns = [word for word in headline.split() if word[0].isupper() and word.lower() not in common_words]
            if len(proper_nouns) >= 2:
                return ' '.join(proper_nouns[:2])
            return ' '.join(headline_words[:2]).title()
        
        # If headline extraction fails, try the summary
        # Look for phrases in quotes or proper nouns
        quoted_phrases = re.findall(r'"(.*?)"', summary)
        if quoted_phrases:
            return quoted_phrases[0]
            
        # Look for proper noun phrases in the first sentence of the summary
        first_sentence = summary.split('.')[0]
        proper_nouns = [word for word in first_sentence.split() if word[0].isupper() and word.lower() not in common_words]
        if len(proper_nouns) >= 2:
            return ' '.join(proper_nouns[:2])
            
        # Fallback: Use the first 3-5 meaningful words from the headline
        meaningful_words = [word for word in headline.split() if word.lower() not in common_words][:3]
        if meaningful_words:
            return ' '.join(meaningful_words)
            
        # Last resort: First few words of the headline
        return ' '.join(headline.split()[:4])
    
    def _get_or_create_topic(self, article: Dict[str, Any], category: str) -> str:
        """Get or create a topic in the given category."""
        # Generate embedding for the article
        text = f"{article['headline']} {article.get('summary', '')}"
        article_embedding = self.model.encode([text], convert_to_tensor=False)
        current_time = datetime.utcnow()
        
        # Get topics for this category
        category_topics = self.topics[category]
        
        # Generate a topic name
        topic_name = self._extract_topic_name(article, category)
        
        # If no topics in category yet, create a new one
        if not category_topics:
            self.topics[category].append((topic_name, article_embedding, current_time))
            return topic_name
        
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
            # Keep the existing topic name to maintain consistency
            topic_title = category_topics[max_sim_idx][0]
            # Update the topic's last_updated time and embedding
            self.topics[category][max_sim_idx] = (topic_title, article_embedding, current_time)
            return topic_title
        
        # If we reach max topics for this category, replace the oldest topic
        if len(category_topics) >= self.max_topics_per_category:
            # Find the oldest topic
            oldest_idx = np.argmin([topic[2] for topic in category_topics])
            self.topics[category][oldest_idx] = (topic_name, article_embedding, current_time)
            return topic_name
        
        # Otherwise, add a new topic to this category
        self.topics[category].append((topic_name, article_embedding, current_time))
        return topic_name
    
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
            
            # If no category match, use the best matching category from the predefined ones
            # instead of 'general' to avoid the error
            category = self._categorize_article(article)
            if category == 'general':
                # If still 'general', use the first category as fallback
                category = list(self.CATEGORIES.keys())[0]
                self.logger.info(f"Using fallback category '{category}' for article")
            
            topic = self._get_or_create_topic(article, category)
            return (category, topic)
            
        except Exception as e:
            self.logger.error(f"Error classifying article: {e}", exc_info=True)
            # Return the first available category as fallback
            fallback_category = list(self.CATEGORIES.keys())[0]
            return (fallback_category, 'general')
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
