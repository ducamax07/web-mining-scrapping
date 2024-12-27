import json
import csv
import os
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string

def generate_token_table(json_path, output_csv):
    STOP_WORDS = set(stopwords.words('english')) | set(string.punctuation)
    STEMMER = SnowballStemmer('english')
    with open(json_path, 'r', encoding='utf-8') as file:
        content = json.load(file)
    token_counts = {}  # Dictionnaire pour les occurrences
    document_counts = defaultdict(int)  # Nombre de documents contenant chaque token
    word_map = defaultdict(set)  # Mots originaux associés aux tokens stemmés


    for text in content.values():
        tokens = word_tokenize(text.lower())
        stemmed_tokens = [STEMMER.stem(token) for token in tokens if token.isalnum() and token not in STOP_WORDS]
        #Compter les occurrences globales
        for token in stemmed_tokens:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1

        #Compter les documents contenant chaque token (uniquement une fois par document)
        unique_tokens = set(stemmed_tokens)
        for token in unique_tokens:
            document_counts[token] += 1

        #Mapper les mots originaux aux tokens stemmés
        for word in tokens:
            if word.isalnum() and word not in STOP_WORDS:
                stemmed = STEMMER.stem(word)
                word_map[stemmed].add(word)

    # Construire et sauvegarder le tableau en une étape
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Écrire l'en-tête
        writer.writerow(["Token (stemmatisé)", "Mots associés", "Occurrences", "Articles concernés"])
        
        # Écrire les données
        for token, freq in token_counts.items():
            writer.writerow([
                token,
                ', '.join(word_map[token]),  # Convertir les mots associés en chaîne de caractères
                freq,
                document_counts[token]
            ])
    print(f"Tableau sauvegardé dans {output_csv}")


with open("content3.json", "r") as file:
    data = json.load(file)
    print(len(data.values())) 