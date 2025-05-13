"""Command-line interface for the AI News Aggregator."""
import os
import logging
import click
from dotenv import load_dotenv
from typing import List, Dict, Any
from datetime import datetime

from .news_fetcher import NewsFetcher
from .topic_classifier import TopicClassifier
from .deduplicator import Deduplicator
from .summarizer import ArticleSummarizer
from .firebase_service import FirebaseService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
@click.version_option()
def cli():
    """AI News Aggregator - Organize news articles into topic-based timelines."""
    pass

@cli.command()
@click.option('--firebase-credentials', help='Path to Firebase credentials JSON file', required=True)
@click.option('--firebase-db-url', help='Firebase database URL', required=True)
@click.option('--limit', type=int, default=5, help='Number of top topics to display')
def list_topics(firebase_credentials: str, firebase_db_url: str, limit: int):
    """List the top topics from the database."""
    try:
        firebase_service = FirebaseService(firebase_credentials, firebase_db_url)
        
        # Get all topics
        topics_ref = firebase_service.db.collection('topics')
        topics = topics_ref.order_by('latestUpdate', direction='DESCENDING').limit(limit).stream()
        
        click.echo(f"\nTop {limit} Topics:\n" + "="*50)
        for i, topic in enumerate(topics, 1):
            topic_data = topic.to_dict()
            click.echo(f"{i}. {topic_data.get('title', 'Untitled')}")
            click.echo(f"   Latest Update: {topic_data.get('latestUpdate', 'N/A')}")
            
            # Get article count
            events_ref = topics_ref.document(topic.id).collection('events')
            article_count = len(list(events_ref.limit(100).stream()))
            click.echo(f"   Articles: {article_count}")
            
            # Get most recent article
            recent_article = events_ref.order_by('timestamp', direction='DESCENDING').limit(1).stream()
            for article in recent_article:
                article_data = article.to_dict()
                click.echo(f"   Latest: {article_data.get('headline', 'N/A')[:60]}...")
            
            click.echo("-"*50)
            
    except Exception as e:
        logger.error(f"Error listing topics: {e}", exc_info=True)
        raise click.ClickException(str(e))

@cli.command()
@click.option('--rss-feeds', help='Comma-separated list of RSS feed URLs', required=True)
@click.option('--firebase-credentials', help='Path to Firebase credentials JSON file', required=True)
@click.option('--firebase-db-url', help='Firebase database URL', required=True)
@click.option('--max-articles', type=int, default=50, help='Maximum number of articles to process')
@click.option('--similarity-threshold', type=float, default=0.75, help='Similarity threshold for topic classification (0.0 to 1.0)')
@click.option('--dedupe-threshold', type=float, default=0.85, help='Similarity threshold for deduplication (0.0 to 1.0)')
@click.option('--summary-length', type=int, default=150, help='Maximum length of article summaries')
@click.option('--dry-run', is_flag=True, help='Process articles but do not upload to Firebase')
def run(
    rss_feeds: str,
    firebase_credentials: str,
    firebase_db_url: str,
    max_articles: int,
    similarity_threshold: float,
    dedupe_threshold: float,
    summary_length: int,
    dry_run: bool
):
    """Fetch, process, and organize news articles into topic-based timelines."""
    try:
        # Initialize components
        logger.info("Initializing news aggregator...")
        
        # Initialize Firebase service
        firebase_service = FirebaseService(firebase_credentials, firebase_db_url)
        
        # Initialize other components
        news_fetcher = NewsFetcher(rss_feeds.split(','))
        topic_classifier = TopicClassifier(
            similarity_threshold=similarity_threshold,
            max_topics=5  # Limit to 5 topics
        )
        deduplicator = Deduplicator(similarity_threshold=dedupe_threshold)
        summarizer = ArticleSummarizer(max_length=summary_length)
        
        # Step 1: Fetch articles
        logger.info("Fetching articles from RSS feeds...")
        articles = news_fetcher.fetch_articles()
        logger.info(f"Fetched {len(articles)} articles")
        
        if not articles:
            logger.warning("No articles were fetched. Exiting.")
            return
        
        # Step 2: Deduplicate articles
        logger.info("Removing duplicate articles...")
        unique_articles = deduplicator.deduplicate_articles(articles)
        logger.info(f"After deduplication: {len(unique_articles)} unique articles")
        
        # Limit number of articles to process
        if len(unique_articles) > max_articles:
            logger.info(f"Limiting to {max_articles} most recent articles")
            unique_articles = sorted(
                unique_articles,
                key=lambda x: x.get('publication_date', datetime.min),
                reverse=True
            )[:max_articles]
        
        # Step 3: Generate summaries
        logger.info("Generating article summaries...")
        articles_with_summaries = summarizer.summarize_articles(unique_articles)
        
        # Step 4: Classify articles into topics
        logger.info("Classifying articles into topics...")
        topic_assignments = {}
        
        for article in articles_with_summaries:
            topic, is_new = topic_classifier.classify_article(article)
            if topic not in topic_assignments:
                topic_assignments[topic] = []
            topic_assignments[topic].append(article)
        
        logger.info(f"Organized articles into {len(topic_assignments)} topics")
        
        if dry_run:
            logger.info("Dry run complete. Would have processed the following:")
            for topic, articles in topic_assignments.items():
                logger.info(f"Topic: {topic} ({len(articles)} articles)")
                for article in articles:
                    logger.info(f"  - {article['headline']} ({article['source']})")
            return
        
        # Step 5: Process each article
        logger.info("Processing articles...")
        topic_counts = {}
        
        for article in unique_articles:
            try:
                # Generate summary if not already present
                if 'summary' not in article:
                    summary = summarizer.summarize_article(article)
                    article['summary'] = summary
                
                # Classify article into a category and topic
                category, topic = topic_classifier.classify_article(article)
                
                if not dry_run:
                    # Add article to topic in Firebase
                    firebase_service.add_article_to_topic(category, topic, article)
                
                # Update topic counts
                topic_key = f"{category.upper()}: {topic}"
                topic_counts[topic_key] = topic_counts.get(topic_key, 0) + 1
                
                logger.debug(f"Added to {category.upper()} topic '{topic}': {article['headline']}")
                
            except Exception as e:
                logger.error(f"Error processing article: {e}", exc_info=True)
                continue
        
        logger.info("News aggregation complete!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise click.ClickException(str(e)) from e

if __name__ == '__main__':
    load_dotenv()
    cli()
