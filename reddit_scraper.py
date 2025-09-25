import praw
import mysql.connector
import time
import re
from datetime import datetime
import requests
from PIL import Image
import pytesseract
import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from config import REDDIT_CONFIG, DB_CONFIG, MAX_POSTS_PER_REQUEST, API_TIMEOUT, REQUEST_DELAY

class RedditScraper:
    def __init__(self):
        self.reddit = praw.Reddit(**REDDIT_CONFIG)
        self.db_connection = None
        self.setup_nltk()
    
    def setup_nltk(self):
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            print(f"NLTK setup warning: {e}")
            pass
    
    def connect_db(self):
        self.db_connection = mysql.connector.connect(**DB_CONFIG)
        return self.db_connection.cursor()
    
    def preprocess_text(self, text):
        if not text:
            return ""
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords_tfidf(self, text):
        if not text or not text.strip():
            return ""
        
        try:
            words = text.split()
            if len(words) < 5:
                return self.extract_keywords_simple(text)
            
            # Create mini-corpus
            sentences = []
            for delimiter in ['.', ':', '-', '|', '–']:
                if delimiter in text:
                    parts = text.split(delimiter)
                    sentences.extend([p.strip() for p in parts if p.strip()])
            
            if len(sentences) < 2:
                words = text.split()
                for i in range(0, len(words), 3):
                    chunk = ' '.join(words[i:i+4])
                    if chunk.strip():
                        sentences.append(chunk)
            
            if len(sentences) < 2:
                return self.extract_keywords_simple(text)
            
            # Enhanced stop words
            custom_stop = ['new', 'first', 'like', 'go', 'get', 'make', 'use', 'way', 'time', 'year', 'day', 'work', 'world', 'life', 'right', 'good', 'high', 'small', 'large', 'long', 'great', 'little', 'own', 'old', 'different', 'big', 'public', 'bad', 'same', 'able']
            
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words=list(set(list(TfidfVectorizer(stop_words='english').get_stop_words()) + custom_stop)),
                ngram_range=(1, 2),
                min_df=1,
                token_pattern=r'\b[a-zA-Z]{3,}\b'
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Filter and deduplicate keywords
            keyword_scores = [(feature_names[i], mean_scores[i]) for i in range(len(feature_names)) if mean_scores[i] > 0]
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            final_keywords = []
            seen_roots = set()
            
            for keyword, score in keyword_scores:
                # Skip if contains common words or is too generic
                if any(word in keyword for word in ['like', 'new', 'first', 'go', 'get']):
                    continue
                
                # Check for root word duplicates
                root = keyword.split()[0] if ' ' in keyword else keyword
                if root in seen_roots:
                    continue
                
                seen_roots.add(root)
                final_keywords.append(keyword)
                
                if len(final_keywords) >= 8:
                    break
            
            return ', '.join(final_keywords) if final_keywords else self.extract_keywords_simple(text)
            
        except Exception as e:
            print(f"TF-IDF error: {e}")
            return self.extract_keywords_simple(text)
    
    def extract_keywords_simple(self, text):
        if not text:
            return ""
        
        try:
            # Try NLTK first
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
        except:
            # Fallback to simple split
            tokens = text.lower().split()
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        keywords = []
        for word in tokens:
            clean_word = re.sub(r'[^a-zA-Z]', '', word)  # Remove non-letters
            if (clean_word and 
                len(clean_word) > 2 and 
                clean_word not in stop_words and 
                clean_word not in keywords):
                keywords.append(clean_word)
        
        return ', '.join(keywords[:10]) if keywords else ""
    
    def extract_image_text(self, url):
        try:
            if not url:
                return ""
            
            # Handle various image hosting sites and formats
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
            image_hosts = ['imgur.com', 'i.redd.it', 'preview.redd.it', 'i.imgur.com']
            
            # Check if URL contains image or is from image hosting site
            is_image = (any(ext in url.lower() for ext in image_extensions) or 
                       any(host in url.lower() for host in image_hosts))
            
            if not is_image:
                return ""
            

            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, timeout=15, headers=headers)
            
            if response.status_code != 200:
                return ""
            
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Use pytesseract with better configuration
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            extracted_text = self.preprocess_text(text)

            
            return extracted_text
            
        except Exception as e:
            return ""
    
    def mask_username(self, username):
        if not username or username == '[deleted]':
            return 'anonymous'
        return f"user_{hash(username) % 10000}"
    
    def fetch_posts_batch(self, subreddit_name, limit):
        posts = []
        subreddit = self.reddit.subreddit(subreddit_name)
        
        try:
            for post in subreddit.hot(limit=min(limit, MAX_POSTS_PER_REQUEST)):
                # Skip promoted/advertisement posts
                if post.stickied or post.distinguished:
                    continue
                
                title = self.preprocess_text(post.title)
                
                # Extract image text if URL contains image
                image_text = self.extract_image_text(post.url)
                
                # For keyword extraction, use title + image_text
                text_for_keywords = title
                if image_text:
                    text_for_keywords += f" {image_text}"
                
                keywords = self.extract_keywords_tfidf(text_for_keywords.strip()) if text_for_keywords.strip() else ""
                topics = self.extract_topics(text_for_keywords.strip()) if text_for_keywords.strip() else ""
                

                
                post_data = {
                    'id': post.id,
                    'title': title,
                    'author': self.mask_username(str(post.author)),
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'upvote_ratio': post.upvote_ratio,
                    'url': post.url,
                    'domain': post.domain,
                    'keywords': keywords,
                    'topics': topics,
                    'image_text': image_text
                }
                posts.append(post_data)
                
        except Exception as e:
            print(f"Error fetching posts: {e}")
        
        return posts
    

    
    def extract_topics(self, text):
        if not text or not text.strip():
            return ""
        
        # Enhanced topic modeling using keyword density and semantic grouping
        topic_keywords = {
            'AI/ML': {
                'primary': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'algorithm'],
                'secondary': ['chatgpt', 'openai', 'robot', 'automation', 'ai', 'ml', 'tensorflow', 'pytorch']
            },
            'Software': {
                'primary': ['software development', 'programming', 'application', 'framework'],
                'secondary': ['code', 'app', 'library', 'api', 'github', 'python', 'javascript', 'developer']
            },
            'Hardware': {
                'primary': ['processor', 'semiconductor', 'chip', 'hardware'],
                'secondary': ['cpu', 'gpu', 'memory', 'storage', 'intel', 'amd', 'nvidia']
            },
            'Mobile': {
                'primary': ['smartphone', 'mobile phone', 'android', 'ios'],
                'secondary': ['iphone', 'samsung', 'google pixel', 'mobile app', 'tablet']
            },
            'Gaming': {
                'primary': ['video game', 'gaming console', 'game development'],
                'secondary': ['steam', 'xbox', 'playstation', 'nintendo', 'esports', 'gaming']
            },
            'Security': {
                'primary': ['cybersecurity', 'data breach', 'encryption', 'privacy'],
                'secondary': ['hack', 'malware', 'security', 'cyber', 'vulnerability']
            },
            'Internet': {
                'primary': ['internet', 'web browser', 'network', 'website'],
                'secondary': ['online', 'wifi', 'broadband', 'connectivity', 'web']
            },
            'Health/Medical': {
                'primary': ['medical technology', 'health tech', 'biotechnology'],
                'secondary': ['medicine', 'treatment', 'therapy', 'disease', 'healthcare']
            }
        }
        
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keyword_groups in topic_keywords.items():
            score = 0
            
            # Primary keywords get higher weight
            for keyword in keyword_groups['primary']:
                if keyword in text_lower:
                    score += 3
            
            # Secondary keywords get lower weight
            for keyword in keyword_groups['secondary']:
                if keyword in text_lower:
                    score += 1
            
            if score > 0:
                topic_scores[topic] = score
        
        # Return topics sorted by relevance score
        if topic_scores:
            sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
            return ', '.join([topic for topic, score in sorted_topics[:3]])  # Top 3 topics
        
        return 'General'
    
    def save_posts(self, posts):
        cursor = self.connect_db()
        
        insert_query = """
        INSERT IGNORE INTO posts 
        (id, title, author, created_utc, score, num_comments, upvote_ratio, url, domain, keywords, topics, image_text)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        for post in posts:
            cursor.execute(insert_query, (
                post['id'], post['title'], post['author'],
                post['created_utc'], post['score'], post['num_comments'],
                post['upvote_ratio'], post['url'], post['domain'],
                post['keywords'], post['topics'], post['image_text']
            ))
        
        self.db_connection.commit()
        cursor.close()
        self.db_connection.close()
        
        return len(posts)
    
    def scrape_posts(self, num_posts, subreddit='tech'):
        print(f"Starting to scrape {num_posts} posts from r/{subreddit}")
        
        total_fetched = 0
        all_posts = []
        request_start_time = time.time()
        max_runtime = 380  # 380 seconds to stay under 400s limit
        
        while total_fetched < num_posts:
            # Check timeout - stop if approaching 400 seconds
            elapsed_total = time.time() - request_start_time
            if elapsed_total > max_runtime:
                print(f"Timeout approaching ({elapsed_total:.1f}s), stopping at {total_fetched} posts")
                break
            
            remaining = num_posts - total_fetched
            # Dynamic batch sizing based on remaining time and posts
            time_remaining = max_runtime - elapsed_total
            
            if time_remaining < 30:  # Less than 30s left
                batch_size = min(remaining, 25)
            elif time_remaining < 60:  # Less than 60s left
                batch_size = min(remaining, 50)
            else:
                # Normal batch size, respect API limits
                batch_size = min(remaining, min(MAX_POSTS_PER_REQUEST, 100))
            
            print(f"Batch {batch_size} posts... ({total_fetched}/{num_posts}, {time_remaining:.0f}s left)")
            
            batch_start_time = time.time()
            
            try:
                posts = self.fetch_posts_batch(subreddit, batch_size)
                
                if not posts:
                    print("No more posts available")
                    break
                
                all_posts.extend(posts)
                total_fetched += len(posts)
                
                batch_elapsed = time.time() - batch_start_time
                print(f"Fetched {len(posts)} posts in {batch_elapsed:.1f}s")
                
                # Save periodically for large requests to avoid data loss
                if len(all_posts) >= 500:
                    print(f"Saving {len(all_posts)} posts to database...")
                    saved_count = self.save_posts(all_posts)
                    print(f"Saved {saved_count} posts")
                    all_posts = []  # Clear saved posts
                
                # Adaptive delay based on batch time and remaining time
                if time_remaining > 60 and batch_elapsed < REQUEST_DELAY:
                    sleep_time = min(REQUEST_DELAY - batch_elapsed, time_remaining - 30)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
            except Exception as e:
                print(f"Batch error: {e}, continuing...")
                time.sleep(1)  # Brief pause before retry
        
        # Save any remaining posts
        if all_posts:
            print(f"Saving final {len(all_posts)} posts to database...")
            saved_count = self.save_posts(all_posts)
            print(f"Saved {saved_count} posts")
        
        total_time = time.time() - request_start_time
        print(f"Completed: {total_fetched} posts in {total_time:.1f} seconds")
        
        return total_fetched

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python reddit_scraper.py <number_of_posts>")
        sys.exit(1)
    
    try:
        num_posts = int(sys.argv[1])
        if num_posts <= 0:
            raise ValueError("Number of posts must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    scraper = RedditScraper()
    scraper.scrape_posts(num_posts)

if __name__ == "__main__":
    main()