import wikipedia
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
import json
import os
import re

# Configurer Wikipédia et NLTK
wikipedia.set_lang("en")
nltk.download('punkt')
nltk.download('stopwords')

# Préparer les stopwords et le stemmer
stop_words = list(set(stopwords.words('english'))) + ["'s"]
stem = nltk.stem.SnowballStemmer("english")

# Fonctions utilitaires
def extract_tokens(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stem.stem(token) for token in tokens]
    return tokens


def get_top_tokens(content, n=25):  # Modification ici à 25 tokens
    tokens = extract_tokens(content)
    token_counts = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    sorted_tokens = sorted(token_counts.items(), key=lambda item: item[1], reverse=True)
    return set(item[0] for item in sorted_tokens[:n])


def save_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    return {}


def save_explored_pages(file_path, explored_pages):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(list(explored_pages), file, ensure_ascii=False, indent=4)


def load_explored_pages(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return set(json.load(file))
    return set()


def bfs_scrape(start_page, max_depth=3, 
               content_file='contentA.json', 
               link_file='linksA.json', 
               queue_file='queueA.json', 
               explored_file='exploredA.json'):

    visited = set()
    queue = [(start_page, 0)]  # Recommencer depuis le début
    explored_pages = load_explored_pages(explored_file)

    content_storage = {}
    link_storage = {}

    main_page = wikipedia.WikipediaPage(start_page)
    top_tokens = get_top_tokens(main_page.content, n=25)  # Modification ici

    while queue:
        current_page, depth = queue.pop(0)

        if depth > max_depth:
            continue

        # Ignorer les pages déjà explorées sans les vérifier
        if current_page in explored_pages:
            print(f"Page déjà explorée : {current_page}")
            continue

        if current_page in visited:
            continue
        visited.add(current_page)

        try:
            print(f"Scraping: {current_page} (Depth {depth})")
            page = wikipedia.WikipediaPage(current_page)

            # Sauvegarder directement les données car validité ignorée
            content_storage[current_page] = page.content
            for link in page.links:
                if link not in visited and link not in explored_pages:
                    queue.append((link, depth + 1))

            # Ajouter la page comme explorée
            explored_pages.add(current_page)
            save_explored_pages(explored_file, explored_pages)

        except wikipedia.exceptions.DisambiguationError:
            print(f"DisambiguationError sur {current_page}. Ignoré.")
        except wikipedia.exceptions.PageError:
            print(f"PageError : {current_page} n'existe pas.")
        except Exception as e:
            print(f"Erreur inattendue sur {current_page}: {e}")

    # Sauvegarder les données collectées
    save_data(content_file, content_storage)
    save_data(link_file, link_storage)
    print(f"Scraping terminé. Visité {len(visited)} pages.")


# Lancer le scraping
import time

if __name__ == "__main__":
    while True:
        try:
            bfs_scrape("Diversity (business)", max_depth=3)
            print("Scraping terminé avec succès.")
            break
        except Exception as e:
            print(f"Erreur : {e}")
            print("Redémarrage dans 10 secondes...")
            time.sleep(10)
        except KeyboardInterrupt:
            print("Arrêt manuel.")
            break