import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
import string

# Configuration
STOP_WORDS = set(stopwords.words('english')) | set(string.punctuation)
STEMMER = SnowballStemmer('english')

# Fonction pour nettoyer et tokeniser le texte
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [STEMMER.stem(token) for token in tokens if token.isalnum() and token not in STOP_WORDS]

# Fonction principale pour trouver les cooccurrences
def find_cooccurrences(corpus_path, keyword, min_freq=2, window_size=5):
    with open(corpus_path, 'r', encoding='utf-8') as file:
        corpus = json.load(file)
    
    # Prétraitement des textes et agrégation des tokens
    all_tokens = []
    for text in corpus.values():
        all_tokens.extend(preprocess_text(text))
    
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(all_tokens, window_size=window_size)
    finder.apply_freq_filter(min_freq)
    
    # Filtrer pour ne conserver que les bigrammes contenant le mot-clé
    keyword_stem = STEMMER.stem(keyword)
    # Filtrer les bigrammes pertinents contenant le mot-clé stemmatisé
    relevant_bigrams = {}
    for bigram, freq in finder.ngram_fd.items():
        if keyword_stem in bigram:
            distances = []
            # Parcourir les occurrences pour calculer les distances
            for i in range(len(all_tokens) - 1):
                if all_tokens[i] == bigram[0] and bigram[1] in all_tokens[i + 1:i + window_size + 1]:
                    j = all_tokens.index(bigram[1], i + 1, i + window_size + 1)
                    distances.append(abs(j - i))
            # Ajouter la fréquence et la distance moyenne
            relevant_bigrams[bigram] = {
                "frequency": freq,
                "mean_distance": sum(distances) / len(distances) if distances else 0
            }
    # Trier les bigrammes par fréquence

    sorted_bigrams = sorted(relevant_bigrams.items(), key=lambda x: x[1]['frequency'], reverse=True)

    # Afficher les résultats
    print(f"Cooccurrences pour le mot-clé '{keyword}':\n")
    print(f"{'Cooccurrence':<20}\t{'Frequency':<10}\t{'Mean Distance':<15}")
    print(f"{'--------------------------------------------------------------'}")
    for bigram, data in sorted_bigrams:
        print(f"{bigram[0]}\t{bigram[1]:<17}\t{data['frequency']:<10}\t{data['mean_distance']:<15.2f}")

    return sorted_bigrams



if __name__ == "__main__":
    corpus_path = 'content3.json'  # Remplacez par votre fichier JSON
    keyword = "profit"         # Mot-clé à analyser
    find_cooccurrences(corpus_path, keyword, min_freq=10, window_size=1)
