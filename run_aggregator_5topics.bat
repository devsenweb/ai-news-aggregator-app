@echo off
echo Running AI News Aggregator with 5-topic limit...
python -m ai_news_aggregator.cli run ^
  --rss-feeds "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml,http://feeds.bbci.co.uk/news/rss.xml" ^
  --firebase-credentials "ai-news-aggregation.json" ^
  --firebase-db-url "https://ai-news-aggregation.firebaseio.com" ^
  --max-articles 20
pause
