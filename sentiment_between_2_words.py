import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
import re
import string
stop_words = list(set(stopwords.words('english'))) + ["'s"]
stem = nltk.stem.SnowballStemmer("english")

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
# Charger le corpus
file_path = "content1.json"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")
with open(file_path, "r", encoding="utf-8") as file:
    corpus = json.load(file)

# Télécharger les ressources nécessaires
nltk.download('punkt')
nltk.download('vader_lexicon')
file_path = "content1.json"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")
with open(file_path, "r", encoding="utf-8") as file:
    corpus = json.load(file)



# Initialiser l'analyseur de sentiment
sia = SentimentIntensityAnalyzer()

# Fonction pour analyser les sentiments associés à un mot dans le corpus
def analyze_word_sentiment(word1, word2, corpus):
    positive, negative, neutral = 0, 0, 0
    
    for doc_id, content in corpus.items():
        sentences = sent_tokenize(content)  # Découper en phrases
        for sentence in sentences:
            
            if word1  in extract_tokens(sentence) and word2 in extract_tokens(sentence):  # Vérifier si le mot est dans la phrase
                print(sentence)
                sentiment = sia.polarity_scores(sentence)  # Analyse de sentiment
                print(sentiment)
                print("-------------------------------------")
                
                
                if sentiment['compound'] > 0.05:
                    positive += 1
                elif sentiment['compound'] < -0.05:
                    negative += 1
                else:
                    neutral += 1

    total = positive + negative + neutral
    if total == 0:
        return f"Le mot '{word1,word2}' n'apparaît pas dans le corpus."

    return {
        "Mot": (word1, word2),
        "Positif": positive,
        "Négatif": negative,
        "Neutre": neutral,
        "Total": total,
        "Ratio positif": f"{(positive / total) * 100:.0f}%",
        "Ratio négatif": f"{(negative / total) * 100:.0f}%"
    }

"""# Exemple d'utilisation
mot = "busi"
resultat = analyze_word_sentiment(mot, "divers",corpus)
print("Analyse de sentiment pour le mot choisi :")
print(json.dumps(resultat, indent=4, ensure_ascii=False))"""