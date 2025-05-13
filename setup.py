from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-news-aggregator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered news aggregator that organizes articles into topic-based timelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-news-aggregator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content :: News/Diary",
    ],
    python_requires='>=3.8',
    install_requires=[
        'firebase-admin>=6.0.0',
        'python-dotenv>=0.19.0',
        'feedparser>=6.0.8',
        'newspaper3k>=0.2.8',
        'nltk>=3.6.7',
        'scikit-learn>=1.0.2',
        'sentence-transformers>=2.2.2',
        'python-dateutil>=2.8.2',
        'tqdm>=4.62.3',
        'click>=8.0.3',
        'transformers>=4.30.0',
        'torch>=1.9.0',  # Required for sentence-transformers
    ],
    entry_points={
        'console_scripts': [
            'ai-news-aggregator=ai_news_aggregator.cli:cli',
        ],
    },
)
