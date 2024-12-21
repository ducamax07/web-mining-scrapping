import nltk
import json
import string
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Charger les stopwords et les initialisations
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english')) | {"'s"}

# Charger le contenu du fichier JSON
def load_content(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    else:
        raise FileNotFoundError(f"Le fichier '{file_path}' est introuvable.")

# Fonction pour afficher les 20 tokens les plus fréquents
def show_top_20_tokens(file_path):
    content = load_content(file_path)
    all_tokens = []

    for text in content.values():
        # Tokeniser et nettoyer les textes
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        all_tokens.extend(tokens)

    # Calculer la fréquence des tokens
    freq_dist = nltk.FreqDist(all_tokens)
    most_common = freq_dist.most_common(20)

    print("Les 20 tokens les plus fréquents sont :")
    for token, frequency in most_common:
        print(f"{token}: {frequency}")

# Fonction pour générer un WordCloud
def generate_wordcloud(file_path):
    content = load_content(file_path)
    all_tokens = []

    for text in content.values():
        # Tokeniser et nettoyer les textes
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        all_tokens.extend(tokens)

    # Générer un WordCloud
    text = ' '.join(all_tokens)
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stop_words,
        max_words=30,
        min_font_size=10
    ).generate(text)

    # Afficher le WordCloud
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Exemple d'utilisation
file_path = "content1.json"
show_top_20_tokens(file_path)
generate_wordcloud(file_path)
