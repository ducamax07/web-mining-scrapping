from nltk.collocations import *
from nltk.tokenize import word_tokenize
import json
import re
import string
from nltk.corpus import stopwords
from nltk.collocations import *
import nltk
# Préparer les stopwords
stop_words = set(stopwords.words('english')) | set(string.punctuation)
custom_stopwords = set(['employee', 'organization', 'work', 'job', 'company'])  # Stopwords spécifiques au corpus
stop_words.update(custom_stopwords)

# Fonction pour nettoyer et tokeniser le texte
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Garder uniquement les lettres et espaces
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if len(token) > 2]  # Exclure les mots très courts
    return tokens

# Charger le corpus
file_path = "content1.json"  # Remplacez par le chemin de votre fichier
with open(file_path, "r", encoding="utf-8") as file:
    corpus = json.load(file)

# Préparer tous les mots du corpus
all_tokens = []
for content in corpus.values():
    all_tokens.extend(preprocess_text(content))

# Trouver les bigrammes
bigram_measures = nltk.collocations.BigramAssocMeasures()
bigram_finder = BigramCollocationFinder.from_words(all_tokens)

# Appliquer un filtre de fréquence minimale
bigram_finder.apply_freq_filter(5)  # Garder les bigrammes apparaissant au moins 5 fois

# Utiliser une métrique différente
best_bigrams = bigram_finder.nbest(bigram_measures.likelihood_ratio, 10)

# Afficher les résultats
print("Top 10 Bigrammes (Likelihood Ratio) :")
print(best_bigrams)
