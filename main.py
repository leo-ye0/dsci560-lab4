import sys
import time
import signal
from datetime import datetime
from reddit_scraper import RedditScraper

class PeriodicScraper:
    def __init__(self, interval_minutes):
        self.interval_minutes = interval_minutes
        self.interval_seconds = interval_minutes * 60
        self.scraper = RedditScraper()
        self.running = True
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        print("\n[INFO] Received interrupt signal. Stopping periodic scraper...")
        self.running = False
    
    def log_message(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def run_scraping_cycle(self):
        try:
            self.log_message("Starting data collection cycle")
            
            # Fetch new posts (default 50 posts per cycle)
            self.log_message("Fetching data from Reddit...")
            posts_fetched = self.scraper.scrape_posts(50)
            
            if posts_fetched > 0:
                self.log_message(f"Processing completed. {posts_fetched} posts processed and stored")
                self.log_message("Database updated successfully")
            else:
                self.log_message("No new posts found", "WARNING")
                
        except Exception as e:
            self.log_message(f"Error during scraping cycle: {e}", "ERROR")
            self.log_message("Continuing with next cycle...", "INFO")
    
    def start(self):
        self.log_message(f"Starting periodic Reddit scraper (interval: {self.interval_minutes} minutes)")
        self.log_message("Press Ctrl+C to stop")
        
        cycle_count = 0
        
        while self.running:
            cycle_count += 1
            self.log_message(f"--- Cycle {cycle_count} ---")
            
            self.run_scraping_cycle()
            
            if self.running:
                self.log_message(f"Waiting {self.interval_minutes} minutes until next cycle...")
                
                # Sleep in smaller intervals to allow for graceful shutdown
                for _ in range(self.interval_seconds):
                    if not self.running:
                        break
                    time.sleep(1)
        
        self.log_message("Periodic scraper stopped")

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <interval_in_minutes>")
        print("Example: python main.py 5")
        sys.exit(1)
    
    try:
        interval = int(sys.argv[1])
        if interval <= 0:
            raise ValueError("Interval must be positive")
    except ValueError as e:
        print(f"Error: Invalid interval - {e}")
        sys.exit(1)
    
    scraper = PeriodicScraper(interval)
    scraper.start()

if __name__ == "__main__":
    main()