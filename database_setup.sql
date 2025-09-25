CREATE DATABASE IF NOT EXISTS reddit_data;
USE reddit_data;

SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS comments;
DROP TABLE IF EXISTS posts;
SET FOREIGN_KEY_CHECKS = 1;

CREATE TABLE posts (
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
);

CREATE INDEX idx_posts_created ON posts(created_utc);
CREATE INDEX idx_posts_score ON posts(score);