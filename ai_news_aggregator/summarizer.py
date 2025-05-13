"""Module for generating concise summaries of news articles."""
from typing import Dict, Any, List
import logging
from transformers import pipeline
from newspaper import Article

class ArticleSummarizer:
    """Generates concise summaries of news articles."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", max_length: int = 150):
        """Initialize the summarizer.
        
        Args:
            model_name: Name of the summarization model to use
            max_length: Maximum length of the generated summary
        """
        self.model_name = model_name
        self.max_length = max_length
        self.summarizer = None
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Lazy initialization of the summarization model."""
        if self.summarizer is None:
            self.logger.info(f"Loading summarization model: {self.model_name}")
            self.summarizer = pipeline("summarization", model=self.model_name)
    
    def _get_dynamic_length(self, text: str) -> int:
        """Calculate dynamic max length based on input text length."""
        word_count = len(text.split())
        # For very short texts, return a fraction of the length
        if word_count < 50:
            return max(10, word_count // 2)
        # For medium texts, return a smaller fraction
        elif word_count < 200:
            return max(30, word_count // 3)
        # For longer texts, cap at max_length
        return min(self.max_length, max(50, word_count // 4))

    def summarize_article(self, article: Dict[str, Any]) -> str:
        """Generate a summary for a single article.
        
        Args:
            article: Dictionary containing article data
            
        Returns:
            Generated summary text
        """
        self.initialize()
        
        try:
            # Extract text content
            text = article.get('content', '').strip()
            if not text and 'url' in article:
                # If no content but URL is available, try to download the article
                try:
                    article_obj = Article(article['url'])
                    article_obj.download()
                    article_obj.parse()
                    text = article_obj.text
                except Exception as e:
                    self.logger.warning(f"Failed to download article from {article['url']}: {e}")
                    return article.get('headline', '')[:self.max_length] + "..."
            
            if not text:
                return article.get('headline', '')[:self.max_length] + "..."
            
            # Calculate dynamic length based on input text
            dynamic_max_length = self._get_dynamic_length(text)
            
            # If text is short, summarize directly
            if len(text.split()) < 300:  # About 1-2 paragraphs
                summary = self.summarizer(
                    text,
                    max_length=dynamic_max_length,
                    min_length=max(10, dynamic_max_length // 2),
                    do_sample=False
                )
                return summary[0]['summary_text'].strip()
            
            # For longer texts, use extractive summarization first
            from collections import defaultdict
            from heapq import nlargest
            from string import punctuation
            from nltk.tokenize import sent_tokenize, word_tokenize
            from nltk.corpus import stopwords
            
            try:
                # Tokenize into sentences
                sentences = sent_tokenize(text)
                
                # Remove stopwords and punctuation
                stop_words = set(stopwords.words('english') + list(punctuation))
                
                # Calculate word frequencies
                word_frequencies = defaultdict(int)
                for sentence in sentences:
                    for word in word_tokenize(sentence.lower()):
                        if word not in stop_words:
                            word_frequencies[word] += 1
                
                # Calculate sentence scores based on word frequencies
                sentence_scores = defaultdict(int)
                for i, sentence in enumerate(sentences):
                    for word in word_tokenize(sentence.lower()):
                        if word in word_frequencies:
                            sentence_scores[i] += word_frequencies[word]
                
                # Get top sentences
                selected_sentences_indices = nlargest(
                    min(5, len(sentences) // 2),  # Take at most 5 or half the sentences
                    sentence_scores,
                    key=sentence_scores.get
                )
                
                # Sort selected sentences by their original order
                selected_sentences = [sentences[i] for i in sorted(selected_sentences_indices)]
                extractive_summary = ' '.join(selected_sentences)
                
                # If the extractive summary is already concise, return it
                if len(extractive_summary.split()) <= dynamic_max_length * 1.5:
                    return extractive_summary
                
                # Otherwise, use abstractive summarization on the extractive summary
                summary = self.summarizer(
                    extractive_summary,
                    max_length=dynamic_max_length,
                    min_length=max(30, dynamic_max_length // 2),
                    do_sample=False
                )
                return summary[0]['summary_text'].strip()
                
            except Exception as e:
                self.logger.warning(f"Extractive summarization failed, falling back to full text: {e}")
                # Fallback to chunking if extractive summarization fails
                max_chunk_length = 1024
                chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
                
                summaries = []
                for chunk in chunks:
                    chunk_max_length = self._get_dynamic_length(chunk)
                    summary = self.summarizer(
                        chunk,
                        max_length=chunk_max_length,
                        min_length=max(10, chunk_max_length // 3),
                        do_sample=False
                    )
                    summaries.append(summary[0]['summary_text'])
                
                combined_summary = " ".join(summaries)
                if len(combined_summary.split()) > dynamic_max_length * 1.5:
                    final_summary = self.summarizer(
                        combined_summary,
                        max_length=dynamic_max_length,
                        min_length=max(20, dynamic_max_length // 2),
                        do_sample=False
                    )
                    return final_summary[0]['summary_text'].strip()
                return combined_summary.strip()
            
        except Exception as e:
            self.logger.error(f"Error summarizing article: {e}", exc_info=True)
            # Fallback to headline or first 150 characters of content
            return article.get('headline', article.get('content', '')[:self.max_length] + "...")
    
    def summarize_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate summaries for a list of articles.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of articles with added 'summary' field
        """
        self.initialize()
        
        for article in articles:
            if 'summary' not in article or not article['summary']:
                article['summary'] = self.summarize_article(article)
        
        return articles
