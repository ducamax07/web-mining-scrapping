import wikipedia
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
import json
import os
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
# Configurer Wikipédia et NLTK
wikipedia.set_lang("en")
nltk.download('punkt')
nltk.download('stopwords')

# Préparer les stopwords et le stemmer (radicalisation)
stop_words = list(set(stopwords.words('english'))) + ["'s"]
stem = nltk.stem.SnowballStemmer("english")


#---- Nettoyage des données----

# Fonction pour extraire les tokens 
def extract_tokens(text):
    # Mettre tout en minuscules
    text = text.lower()
    # Supprimer les caractères non-alphabétiques
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Tokenisation
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]
    # Supprimer les stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    tokens = [stem.stem(token) for token in tokens]
    return tokens

# Fonction pour charger le contenu d'un fichier JSON
def load_content(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    return []

# Fonction pour nettoyer les tokens et les sauvegarder dans un nouveau fichier JSON
def clean_and_save_tokens(input_file, output_file):
    content = load_content(input_file)
    cleaned_data = {}

    for page_title, page_content in content.items():
        # Extraire les tokens nettoyés
        tokens = extract_tokens(page_content)
        # Sauvegarder les tokens par page
        cleaned_data[page_title] = tokens
    
    # Sauvegarder dans un nouveau fichier JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
        print(f"Tokens nettoyés sauvegardés dans : {output_file}")

# Frequences des mots dans le nouveau fichier JSON
# Fonction pour afficher les 20 tokens les plus fréquents
def show_top_20_tokens(file_path):
    content = load_content(file_path)
    all_tokens = []

    for tokens in content.values():
        all_tokens.extend(tokens)

    freq_dist = nltk.FreqDist(all_tokens)
    most_common = freq_dist.most_common(20)

    print("Les 20 tokens les plus fréquents sont :")
    for token, frequency in most_common:
        print(f"{token}: {frequency}")

def generate_wordcloud(file_path):
    content = load_content(file_path)
    all_tokens = []

    for tokens in content.values():
        all_tokens.extend(tokens)

    text = ' '.join(all_tokens)
    
    wordcloud = WordCloud(background_color ='white', 
                stopwords = stop_words, max_words=30,
                min_font_size = 10).generate(text)

    plt.imshow(wordcloud) 
    plt.axis("off")
    plt.show()

"""
#----classification des textes en clusters----
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
# Sauvegarder les clusters obtenus
def save_clusters(file_path, clusters):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(clusters, file, ensure_ascii=False, indent=4)
file_path = 'content1.json'
content = load_content(file_path)

# Étape 2 : Préparer les données (chaque document est une valeur dans le JSON)
documents = list(content.values())

# Étape 3 : Calculer la matrice TF-IDF pour les documents
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words="english")
X_tfidf = vectorizer.fit_transform(documents)
print(f"Matrice TF-IDF : {X_tfidf.shape}")


# Étape 4 : Appliquer KMeans pour classifier les documents
k = 3  # Choisir le nombre de clusters
kmeans = KMeans(n_clusters=k, max_iter=100, n_init=1)
kmeans.fit(X_tfidf)

# Étape 5 : Obtenir les clusters
document_clusters = {}
for i, doc in enumerate(documents):
    cluster = int(kmeans.labels_[i])  # Convertir numpy.int32 en int
    document_clusters[f"Document_{i}"] = {"cluster": cluster, "content": doc}

# Sauvegarder les résultats
output_file = "document_clusters.json"
save_clusters(output_file, document_clusters)
print(f"Les documents ont été classifiés en {k} clusters. Résultats sauvegardés dans {output_file}.")

#afficher les mots les plus fréquents dans chaque cluster (trouvé dans documentation sklearn)
print("Top termes par cluster :\n")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]  # Trier les termes par importance dans chaque cluster
terms = vectorizer.get_feature_names_out()
for i in range(3):
    print(f"Cluster {i}:")
    for ind in order_centroids[i, :20]: 
        print(f" {terms[ind]}")
    print("\n")

#afficher le nombre de documents par cluster
def show_cluster_distribution(clusters):
    cluster_counts = {}
    for doc in clusters.values():
        cluster = doc["cluster"]
        if cluster in cluster_counts:
            cluster_counts[cluster] += 1
        else:
            cluster_counts[cluster] = 1

    print("Nombre de documents par cluster :")
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} documents")
show_cluster_distribution(document_clusters)
"""

nltk.download('punkt')
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Télécharger les ressources nécessaires

# Charger et nettoyer le corpus
def load_and_clean_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = json.load(file)
    all_text = " ".join(content.values())
    all_text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", all_text)  # Nettoyer le texte
    return all_text

# Trouver les phrases contenant un mot spécifique
def get_sentences_with_word(text, word):
    sentences = sent_tokenize(text)  # Découper en phrases
    word = word.lower()
    relevant_sentences = [sentence for sentence in sentences if word in sentence.lower()]
    return relevant_sentences

# Analyser le sentiment des phrases
def analyze_word_sentiment(sentences):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0, "compound": 0}
    for sentence in sentences:
        scores = sia.polarity_scores(sentence)
        sentiment_scores["positive"] += scores["pos"]
        sentiment_scores["negative"] += scores["neg"]
        sentiment_scores["neutral"] += scores["neu"]
        sentiment_scores["compound"] += scores["compound"]

    # Moyenne des scores
    num_sentences = len(sentences)
    if num_sentences > 0:
        sentiment_scores = {key: value / num_sentences for key, value in sentiment_scores.items()}

    # Déterminer le sentiment principal
    if sentiment_scores["positive"] > sentiment_scores["negative"] and sentiment_scores["positive"] > sentiment_scores["neutral"]:
        sentiment = "Positif"
    elif sentiment_scores["negative"] > sentiment_scores["positive"] and sentiment_scores["negative"] > sentiment_scores["neutral"]:
        sentiment = "Négatif"
    else:
        sentiment = "Neutre"

    return sentiment_scores, sentiment

# Fonction principale
def word_sentiment_analysis(file_path, word):
    # Charger et nettoyer le corpus
    text = load_and_clean_corpus(file_path)

    # Trouver les phrases contenant le mot
    sentences_with_word = get_sentences_with_word(text, word)
    if not sentences_with_word:
        return f"Le mot '{word}' n'apparaît pas dans le corpus."

    # Analyser le sentiment des phrases
    sentiment_scores, sentiment = analyze_word_sentiment(sentences_with_word)

    return {
        "word": word,
        "sentences_with_word": sentences_with_word,
        "sentiment_scores": sentiment_scores,
        "overall_sentiment": sentiment
    }

# Exemple d'utilisation
file_path = "content1.json"  # Remplacez par le chemin de votre fichier
word = "profit"  # Le mot à analyser
result = word_sentiment_analysis(file_path, word)

print(f"Analyse de sentiment pour le mot '{result['word']}':")
print(f"Scores moyens : {result['sentiment_scores']}")
print(f"Sentiment global : {result['overall_sentiment']}")
print("\nExemples de phrases contenant le mot :")
for sentence in result["sentences_with_word"][:5]:  # Afficher jusqu'à 5 phrases
    print(f"- {sentence}")
