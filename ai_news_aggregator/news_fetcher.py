"""Module for fetching and processing news articles from various sources."""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import feedparser
from newspaper import Article, ArticleException
from dateutil import parser as date_parser

class NewsFetcher:
    """Fetches and processes news articles from various sources."""
    
    def __init__(self, rss_feeds: List[str]):
        """Initialize the news fetcher with a list of RSS feeds.
        
        Args:
            rss_feeds: List of RSS feed URLs to fetch articles from
        """
        self.rss_feeds = rss_feeds
        self.logger = logging.getLogger(__name__)
    
    def fetch_articles(self) -> List[Dict[str, Any]]:
        """Fetch articles from all configured RSS feeds.
        
        Returns:
            List of article dictionaries with metadata and content
        """
        articles = []
        
        for feed_url in self.rss_feeds:
            try:
                self.logger.info(f"Fetching articles from feed: {feed_url}")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    try:
                        article = self._process_entry(entry, feed.feed.get('title', 'Unknown Source'))
                        if article:
                            articles.append(article)
                    except Exception as e:
                        self.logger.error(f"Error processing entry from {feed_url}: {e}", exc_info=True)
                        continue
                        
            except Exception as e:
                self.logger.error(f"Error fetching feed {feed_url}: {e}", exc_info=True)
                continue
                
        return articles
    
    def _process_entry(self, entry: Dict, source_name: str) -> Optional[Dict]:
        """Process a single RSS entry into a structured article.
        
        Args:
            entry: RSS entry dictionary
            source_name: Name of the source feed
            
        Returns:
            Processed article dictionary or None if processing fails
        """
        try:
            # Extract basic metadata
            url = entry.get('link', '')
            if not url:
                return None
                
            # Download and parse the full article
            article = Article(url)
            article.download()
            article.parse()
            
            # Parse the publication date
            pub_date = self._parse_date(entry.get('published', ''))
            
            # Extract domain from URL
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace('www.', '').split('.')[0]
            
            # Prepare article data with all fields
            article_data = {
                'headline': article.title or entry.get('title', 'No title'),
                'url': url,
                'source': source_name,
                'source_name': domain.capitalize(),  # News portal name
                'original_url': url,  # Original article URL
                'publication_date': pub_date,
                'content': article.text,
                'authors': article.authors,
                'top_image': article.top_image,
                'summary': article.meta_description or '',
                'keywords': article.meta_keywords,
                'rss_entry': entry,  # Keep original entry for reference
                'image_url': article.top_image  # Add image_url for consistency
            }
            
            # Log the article data for debugging
            import logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            logger.info(f"Article data prepared: {article_data.keys()}")
            
            return article_data
            
        except ArticleException as e:
            self.logger.warning(f"Failed to parse article {entry.get('link')}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error processing article: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        """Parse a date string into a datetime object.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            Parsed datetime or current time if parsing fails
        """
        try:
            return date_parser.parse(date_str)
        except (ValueError, AttributeError, OverflowError):
            return datetime.utcnow()
