import json
import os
import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Téléchargements nécessaires
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

# --- PARAMÈTRES GLOBAUX ---
STOP_WORDS = set(stopwords.words('english')) | set(string.punctuation)
STOP_WORDS.update(['employee', 'organization', 'work', 'job', 'company', "'s"])
STEMMER = nltk.stem.SnowballStemmer('english')
SIA = SentimentIntensityAnalyzer()

# --- FONCTIONS UTILITAIRES ---

def load_content(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier '{file_path}' est introuvable.")
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_content(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [STEMMER.stem(token) for token in tokens if token not in STOP_WORDS and len(token) > 2]
    return tokens

def tokenize_corpus(corpus):
    all_tokens = []
    for content in corpus.values():
        all_tokens.extend(preprocess_text(content))
    return all_tokens

# --- ANALYSE DE SENTIMENT ---

def analyze_word_sentiment(word1, word2, corpus):
    positive, negative, neutral = 0, 0, 0
    for content in corpus.values():
        sentences = sent_tokenize(content)
        for sentence in sentences:
            tokens = preprocess_text(sentence)
            if word1 in tokens and word2 in tokens:
                sentiment = SIA.polarity_scores(sentence)
                if sentiment['compound'] > 0.05:
                    positive += 1
                elif sentiment['compound'] < -0.05:
                    negative += 1
                else:
                    neutral += 1
    total = positive + negative + neutral
    return {
        'Words': (word1, word2),
        'Positive': positive,
        'Negative': negative,
        'Neutral': neutral,
        'Total': total,
        'Positive Ratio': f"{(positive / total * 100):.0f}%" if total else '0%',
        'Negative Ratio': f"{(negative / total * 100):.0f}%" if total else '0%'
    }

# --- BIGRAMMES ---

def get_top_bigrams(corpus, freq_filter=5, top_n=10):
    all_tokens = tokenize_corpus(corpus)
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(all_tokens)
    finder.apply_freq_filter(freq_filter)
    return finder.nbest(bigram_measures.likelihood_ratio, top_n)

# --- ANALYSE DES TOKENS ---

def show_top_tokens(corpus, top_n=20):
    all_tokens = tokenize_corpus(corpus)
    freq_dist = nltk.FreqDist(all_tokens)
    return freq_dist.most_common(top_n)

# --- WORDCLOUD ---

def generate_wordcloud(corpus):
    all_tokens = tokenize_corpus(corpus)
    text = ' '.join(all_tokens)
    wordcloud = WordCloud(
        background_color='white',
        stopwords=STOP_WORDS,
        max_words=30,
        min_font_size=10
    ).generate(text)
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# --- CLUSTERISATION ---

def cluster_documents(corpus, num_clusters=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    documents = list(corpus.values())
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
    X_tfidf = vectorizer.fit_transform(documents)
    kmeans = KMeans(n_clusters=num_clusters, max_iter=100, n_init=1)
    kmeans.fit(X_tfidf)
    clusters = {}
    for i, doc in enumerate(documents):
        clusters[f"Document_{i}"] = {'cluster': int(kmeans.labels_[i]), 'content': doc}
    return clusters

# --- EXEMPLES D'UTILISATION ---
if __name__ == '__main__':
    file_path = 'content1.json'
    corpus = load_content(file_path)
    
    # Analyse de sentiment
    sentiment_result = analyze_word_sentiment('profit', 'growth', corpus)
    print(sentiment_result)
    
    # Bigrammes
    bigrams = get_top_bigrams(corpus)
    print("Top Bigrams:", bigrams)
    
    # Tokens fréquents
    top_tokens = show_top_tokens(corpus)
    print("Top Tokens:", top_tokens)
    
    # WordCloud
    generate_wordcloud(corpus)
    
    # Clusterisation
    clusters = cluster_documents(corpus)
    save_content(clusters, 'document_clusters.json')
    print("Clustering terminé et sauvegardé dans 'document_clusters.json'")
