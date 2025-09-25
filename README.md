# Reddit Data Collection System - DSCI 560 Lab 4

A complete Reddit scraper that fetches posts from r/tech, extracts keywords and topics, handles image OCR, and stores data in MySQL database.

## Features

- **Reddit API Integration**: Fetches posts from r/tech subreddit
- **Data Processing**: Text preprocessing, username masking, domain extraction
- **Keyword Extraction**: TF-IDF based keyword extraction with fallback
- **Topic Classification**: AI/ML, Hardware, Software, Mobile, Gaming, Security, etc.
- **Image OCR**: Extracts text from images using pytesseract
- **Database Storage**: MySQL database with optimized schema
- **Periodic Scraping**: Automated data collection at specified intervals

## Database Schema

```sql
posts (
    id VARCHAR(20) PRIMARY KEY,
    title TEXT NOT NULL,
    author VARCHAR(255),
    created_utc DATETIME,
    score INT,
    num_comments INT,
    upvote_ratio DECIMAL(3,2),
    url TEXT,
    domain VARCHAR(255),
    keywords TEXT,
    topics TEXT,
    image_text TEXT
)
```

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Reddit API**
   - Create Reddit app at https://www.reddit.com/prefs/apps
   - Update `config.py` with your credentials

3. **Setup Database**
   ```bash
   python setup_database.py
   ```

## Usage

### One-time Scraping
```bash
python reddit_scraper.py <number_of_posts>
```
Example: `python reddit_scraper.py 100`

### Periodic Scraping
```bash
python main.py <interval_in_minutes>
```
Examples:
- `python main.py 5` - Every 5 minutes
- `python main.py 30` - Every 30 minutes
- `python main.py 60` - Every hour

Press Ctrl+C to stop periodic scraping.

## Files

- `reddit_scraper.py` - Main scraper with TF-IDF keyword extraction
- `main.py` - Periodic scraper for automated collection
- `config.py` - API credentials and database settings
- `database_setup.sql` - MySQL schema
- `setup_database.py` - Database initialization
- `requirements.txt` - Python dependencies

## Data Processing

- **Keywords**: TF-IDF extraction from title + image text with enhanced filtering
- **Topics**: 8 categories (AI/ML, Hardware, Software, Mobile, Gaming, Security, Internet, Health)
- **Privacy**: Usernames masked as `user_XXXX` format
- **Domain**: Source domain extracted (youtube.com, github.com, self.tech)
- **Images**: OCR text extraction from imgur, i.redd.it using pytesseract

## Example Queries

```sql
-- Posts by domain
SELECT domain, COUNT(*) FROM posts GROUP BY domain;

-- High-scoring posts
SELECT title, score, keywords FROM posts WHERE score > 100;

-- Posts by topic
SELECT * FROM posts WHERE topics LIKE '%AI%';
```