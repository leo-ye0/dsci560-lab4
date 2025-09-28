import mysql.connector
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF info/warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # hide GPU info
import warnings
warnings.filterwarnings("ignore")
import pickle
import numpy as np
import tensorflow_hub as hub
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Load data
def load_posts(limit=500):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Johnnyware@123",
        database="reddit_data"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM posts LIMIT %s", (limit,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# USE Embedding 
def embed_use(texts):
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embeddings = model(texts)
    return np.array(embeddings), model

# Clustering
def cluster_embeddings(embeddings, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans

# Extract keywords from each cluster
def extract_cluster_keywords(texts, labels, topn=5):
    cluster_keywords = {}
    for cid in set(labels):
        cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == cid]
        if not cluster_texts:
            continue
        vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
        X = vectorizer.fit_transform(cluster_texts)
        feature_names = vectorizer.get_feature_names_out()
        scores = np.asarray(X.mean(axis=0)).flatten()
        top_indices = scores.argsort()[-topn:][::-1]
        cluster_keywords[cid] = [feature_names[i] for i in top_indices]
    return cluster_keywords

# Visualize clusters
def visualize_clusters(embeddings, labels, kmeans, cluster_keywords):
    reduced = PCA(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap="tab10", alpha=0.7)

    # 2D cluster
    centers_2d = PCA(n_components=2).fit_transform(kmeans.cluster_centers_)
    plt.scatter(centers_2d[:,0], centers_2d[:,1], 
                c="black", s=200, marker="X", label="Centroids")

    # centroid keywords
    for i, (x, y) in enumerate(centers_2d):
        plt.text(x, y+0.05, str(i), fontsize=12, weight="bold", color="black",
                 ha="center", va="bottom")

    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title("Clusters of Reddit Posts (USE)")
    plt.savefig("clusters_use.png")
    plt.show()

# save models
def save_models(kmeans, embeddings, texts, save_path="models/cluster_model.pkl"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    with open(save_path, "wb") as f:
        pickle.dump((kmeans, embeddings, texts), f)
    print(f"[INFO] Saved clustering model to {save_path}")

# Main workflow
def main():
    rows = load_posts()
    ids, texts = zip(*rows)

    print("Embedding with USE...")
    embeddings, model = embed_use(list(texts))

    print("Clustering...")
    k = 5
    labels, kmeans = cluster_embeddings(embeddings, k)

    # representative posts in each cluster
    print("\nRepresentative Posts:")
    for cid in range(k):
        cluster_indices = [i for i in range(len(texts)) if labels[i] == cid]
        if not cluster_indices:
            continue
        center = kmeans.cluster_centers_[cid]
        distances = [np.linalg.norm(embeddings[i]-center) for i in cluster_indices]
        rep_index = cluster_indices[np.argmin(distances)]
        print(f"Cluster {cid} representative: {texts[rep_index]}")

    # Keywords for each cluster
    print("\nCluster Keywords:")
    keywords = extract_cluster_keywords(texts, labels, topn=5)
    for cid, words in keywords.items():
        print(f"Cluster {cid} keywords: {', '.join(words)}")

    # Verify clustering results
    print("\nVerifying clusters with sample messages:")
    for cid in range(k):
        cluster_indices = [i for i in range(len(texts)) if labels[i] == cid]
        print(f"\nCluster {cid} ({len(cluster_indices)} posts) keywords: {', '.join(keywords[cid])}")
        for idx in cluster_indices[:5]: 
            print(f"- {texts[idx]}")

    visualize_clusters(embeddings, labels, kmeans, keywords)
    save_models(kmeans, embeddings, texts)

if __name__ == "__main__":
    main()
    
