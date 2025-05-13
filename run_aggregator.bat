@echo off
python -m ai_news_aggregator.cli run ^
  --rss-feeds "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml,http://feeds.bbci.co.uk/news/rss.xml,https://www.theverge.com/rss/index.xml,http://feeds.feedburner.com/TechCrunch/,https://www.wired.com/feed/rss,http://rss.cnn.com/rss/cnn_topstories.rss,http://feeds.reuters.com/reuters/topNews,https://www.theguardian.com/world/rss" ^
  --firebase-credentials "ai-news-aggregation.json" ^
  --firebase-db-url "https://ai-news-aggregation.firebaseio.com"
pause
