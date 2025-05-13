"""Firebase service for interacting with Firestore."""
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import Client as FirestoreClient
import uuid
import logging

class FirebaseService:
    """Service for interacting with Firebase Firestore."""

    def __init__(self, credentials_path: str, database_url: str):
        """Initialize Firebase Admin SDK.
        
        Args:
            credentials_path: Path to the Firebase service account JSON file
            database_url: Firebase database URL
        """
        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': database_url
            })
        
        self.db: FirestoreClient = firestore.client()
        self.logger = logging.getLogger(__name__)
    
    def add_topic(self, topic_data: Dict) -> str:
        """Add a new topic to Firestore.
        
        Args:
            topic_data: Dictionary containing topic data
            
        Returns:
            str: The ID of the created topic
        """
        topic_data['createdAt'] = firestore.SERVER_TIMESTAMP
        topic_data['latestUpdate'] = firestore.SERVER_TIMESTAMP
        
        topic_ref = self.db.collection('topics').document()
        topic_ref.set(topic_data)
        return topic_ref.id
    
    def add_event_to_topic(self, topic_id: str, event_data: Dict) -> str:
        """Add an event to a topic's events subcollection.
        
        Args:
            topic_id: ID of the topic to add the event to
            event_data: Dictionary containing event data
            
        Returns:
            str: The ID of the created event
        """
        event_data['timestamp'] = firestore.SERVER_TIMESTAMP
        
        event_ref = self.db.collection('topics').document(topic_id).collection('events').document()
        event_ref.set(event_data)
        
        # Update the topic's latestUpdate timestamp
        self.db.collection('topics').document(topic_id).update({
            'latestUpdate': firestore.SERVER_TIMESTAMP
        })
        
        return event_ref.id
    
    def topic_exists(self, topic_title: str) -> Optional[str]:
        """Check if a topic with the given title already exists.
        
        Args:
            topic_title: Title of the topic to check
            
        Returns:
            Optional[str]: The topic ID if it exists, None otherwise
        """
        topics_ref = self.db.collection('topics')
        query = topics_ref.where('title', '==', topic_title).limit(1).stream()
        
        for doc in query:
            return doc.id
        return None
    
    def get_topic_events(self, topic_id: str) -> List[Dict]:
        """Get all events for a topic, ordered by timestamp.
        
        Args:
            topic_id: ID of the topic
            
        Returns:
            List[Dict]: List of event dictionaries
        """
        events_ref = self.db.collection('topics').document(topic_id).collection('events')
        events = events_ref.order_by('timestamp', direction='ASCENDING').stream()
        
        return [{'id': event.id, **event.to_dict()} for event in events]
    
    def add_article_to_topic(self, category: str, topic_title: str, article: Dict[str, Any]) -> None:
        """Add an article to a topic in Firestore, with category support.
        
        Args:
            category: The category of the topic
            topic_title: The title of the topic
            article: Article data to add
        """
        try:
            # Clean up any category prefixes in the title
            topic_title = topic_title.strip()
            if ': ' in topic_title:
                topic_title = topic_title.split(': ', 1)[1]
            
            # Get or create the topic document
            topics_ref = self.db.collection('topics')
            
            # Query for existing topic with this title and category
            query = topics_ref.where('title', '==', topic_title).where('category', '==', category).limit(1)
            existing_topics = query.stream()
            
            topic_id = None
            topic_data = None
            
            # Check if topic exists
            for topic in existing_topics:
                topic_id = topic.id
                topic_data = topic.to_dict()
                break
                
            current_time = datetime.utcnow()
            
            if topic_id:
                # Update existing topic
                topic_ref = topics_ref.document(topic_id)
                topic_ref.update({
                    'latestUpdate': current_time,
                    'articleCount': firestore.Increment(1)
                })
            else:
                # Create new topic with category
                new_topic_ref = topics_ref.document()
                topic_id = new_topic_ref.id
                new_topic_ref.set({
                    'title': topic_title,
                    'category': category,
                    'createdAt': current_time,
                    'latestUpdate': current_time,
                    'articleCount': 1
                })
            
            # Prepare article data with all required fields
            article_data = {
                'headline': article.get('title', ''),
                'summary': article.get('summary', ''),
                'source': article.get('source', ''),
                'url': article.get('url', ''),
                # Ensure these fields are included
                'source_name': article.get('source_name', article.get('source', '')),  # Use source_name or fallback to source
                'original_url': article.get('original_url', article.get('url', '')),  # Use original_url or fallback to url
                'publication_date': article.get('publication_date', article.get('published', current_time)),
                'timestamp': current_time,
                'image_url': article.get('image_url', ''),
                'category': category,
                # Include any additional fields from the article
                **{k: v for k, v in article.items() if k not in ['title', 'summary', 'source', 'url', 'published', 'image_url']}
            }
            
            # Add article to the topic's events subcollection
            events_ref = topics_ref.document(topic_id).collection('events')
            
            # Create a safe document ID from URL
            import urllib.parse
            article_url = article.get('url', '')
            article_id = urllib.parse.quote_plus(article_url) if article_url else str(uuid.uuid4())
            
            # Log the article data being saved
            self.logger.info("Article data being saved to Firestore:")
            for key, value in article_data.items():
                self.logger.info(f"  {key}: {value[:100] if isinstance(value, str) else value}")
            
            # Add the safe document ID
            doc_ref = events_ref.document(article_id)
            doc_ref.set(article_data, merge=True)
            self.logger.info(f"Added article to {category.upper()} topic '{topic_title}': {article.get('title')}")
            self.logger.info(f"Document reference: {doc_ref.path}")
            
        except Exception as e:
            self.logger.error(f"Error adding article to topic: {e}", exc_info=True)
            raise
