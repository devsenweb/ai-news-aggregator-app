# AI News Aggregator

A Python-based news aggregation system that fetches, classifies, and organizes news articles into topic-based timelines, storing them in Firebase Firestore.

## Features

- Fetches news articles from multiple RSS feeds
- Classifies articles into topics using semantic similarity
- Removes duplicate or near-duplicate articles
- Generates concise summaries of articles
- Organizes articles into chronological timelines by topic
- Stores results in Firebase Firestore with a structured schema
- Command-line interface for easy execution

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-news-aggregator.git
   cd ai-news-aggregator
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update the values in `.env` with your Firebase credentials and other settings

## Firebase Setup

1. Create a new Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Firestore Database
3. Go to Project Settings > Service Accounts
4. Generate a new private key and save it as `firebase-credentials.json` in the project root
5. Update the `FIREBASE_CREDENTIALS_PATH` in `.env` to point to this file
6. Get your Firebase database URL from Project Settings > General > Your Apps > Firebase SDK snippet

## Usage

### Basic Usage

```bash
python -m ai_news_aggregator.cli run \
  --rss-feeds "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml,http://feeds.bbci.co.uk/news/rss.xml" \
  --firebase-credentials path/to/your/firebase-credentials.json \
  --firebase-db-url "https://your-project-id.firebaseio.com"
```

### Command Line Options

```
Options:
  --rss-feeds TEXT          Comma-separated list of RSS feed URLs  [required]
  --firebase-credentials PATH
                           Path to Firebase credentials JSON file  [required]
  --firebase-db-url TEXT    Firebase database URL  [required]
  --max-articles INTEGER    Maximum number of articles to process  [default: 50]
  --similarity-threshold FLOAT
                           Similarity threshold for topic classification (0.0 to 1.0)  [default: 0.75]
  --dedupe-threshold FLOAT  Similarity threshold for deduplication (0.0 to 1.0)  [default: 0.85]
  --summary-length INTEGER  Maximum length of article summaries  [default: 150]
  --dry-run                 Process articles but do not upload to Firebase  [default: False]
  --help                   Show this message and exit.
```

### Dry Run

To test the aggregator without uploading to Firebase:

```bash
python -m ai_news_aggregator.cli run \
  --rss-feeds "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml" \
  --firebase-credentials path/to/your/firebase-credentials.json \
  --firebase-db-url "https://your-project-id.firebaseio.com" \
  --dry-run
```

## Project Structure

```
ai-news-aggregator/
├── ai_news_aggregator/
│   ├── __init__.py
│   ├── cli.py               # Command-line interface
│   ├── deduplicator.py      # Article deduplication logic
│   ├── firebase_service.py  # Firebase Firestore interactions
│   ├── news_fetcher.py      # RSS feed fetching and parsing
│   ├── summarizer.py        # Article summarization
│   └── topic_classifier.py  # Topic classification
├── tests/                   # Unit tests
├── .env.example            # Example environment variables
├── .gitignore
├── README.md
└── requirements.txt        # Python dependencies
```

## Configuration

Edit the `.env` file to configure the application:

- `FIREBASE_CREDENTIALS_PATH`: Path to your Firebase service account JSON file
- `FIREBASE_DATABASE_URL`: Your Firebase database URL
- `RSS_FEEDS`: Comma-separated list of RSS feed URLs
- `SIMILARITY_THRESHOLD`: Threshold for topic classification (0.0 to 1.0)
- `SUMMARY_MAX_LENGTH`: Maximum length of generated summaries

## License

MIT License - see the [LICENSE](LICENSE) file for details.
