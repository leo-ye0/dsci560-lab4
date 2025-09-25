import os

# Reddit API Configuration
REDDIT_CONFIG = {
    # 'client_id': 'YOUR_CLIENT_ID',
    # 'client_secret': 'YOUR_CLIENT_SECRET',
    # 'user_agent': 'dsci560-lab4-reddit-scraper/1.0'
    'client_id': 'zksjf6woKZY5rbcBpXAvwQ',
    'client_secret': 'sqR0v9baoi2J7y5voG8qnNJErqW6HA',
    'user_agent': 'leoyeah'
}

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'admin',
    'password': 'password',
    'database': 'reddit_data'
}

# API Limits
MAX_POSTS_PER_REQUEST = 100
API_TIMEOUT = 60
REQUEST_DELAY = 2