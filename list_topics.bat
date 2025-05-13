@echo off
echo Fetching top 5 topics...
python -m ai_news_aggregator.cli list-topics ^
  --firebase-credentials "ai-news-aggregation.json" ^
  --firebase-db-url "https://ai-news-aggregation.firebaseio.com" ^
  --limit 5
pause
