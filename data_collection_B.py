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


def get_top_tokens(content, n=25):
    tokens = extract_tokens(content)
    token_counts = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    sorted_tokens = sorted(token_counts.items(), key=lambda item: item[1], reverse=True)
    return set(item[0] for item in sorted_tokens[:n])


def clear_current_link(file_path, current_page):
    if os.path.exists(file_path):
        data = load_data(file_path)
        if current_page in data:
            del data[current_page]
            save_data(file_path, data)



def calculate_dynamic_threshold(page_summary, base_threshold=6):
    text_length = len(page_summary.split())
    return base_threshold + (text_length // 150)


def is_relevant_based_on_top_tokens(top_tokens, linked_summary, base_threshold=6):
    linked_tokens = extract_tokens(linked_summary)
    unique_linked_tokens = set([token for token in linked_tokens if token in top_tokens])
    dynamic_threshold = calculate_dynamic_threshold(linked_summary, base_threshold)
    return len(unique_linked_tokens) >= dynamic_threshold


# Gestion des fichiers JSON

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
               content_file='contentB.json', 
               link_file='linksB.json', 
               queue_file='queueB.json', 
               current_link_file='current_linkB.json', 
               explored_file='exploredB.json'):

    visited = set()
    queue = load_data(queue_file) if os.path.exists(queue_file) else [(start_page, 0)]
    explored_pages = load_explored_pages(explored_file)  # Charger les pages explorées

    content_storage = load_data(content_file)
    link_storage = load_data(link_file)

    # Obtenir les top tokens de la page de départ
    main_page = wikipedia.WikipediaPage(start_page)
    top_tokens = get_top_tokens(main_page.content, n=25)

    while queue:
        current_page, depth = queue.pop(0)
        save_data(queue_file, queue)

        if depth > max_depth:
            continue

        # Ignorer si déjà exploré
        if current_page in explored_pages:
            print(f"Page déjà explorée : {current_page}")
            continue

        if current_page in visited:
            continue
        visited.add(current_page)

        try:
            print(f"Scraping: {current_page} (Depth {depth})")
            page = wikipedia.WikipediaPage(current_page)

            if is_relevant_based_on_top_tokens(top_tokens, page.summary):
                save_data(content_file, content_storage)
                
                for link in page.links:
                    if link not in visited and link not in explored_pages:
                        try:
                            linked_page = wikipedia.WikipediaPage(link)
                            if is_relevant_based_on_top_tokens(top_tokens, linked_page.summary):
                                print(f"Page validée : {link}")
                                save_data(link_file, link_storage)
                                content_storage[link] = linked_page.content
                                link_storage.setdefault(current_page, []).append(link)
                                queue.append((link, depth + 1))
                                save_data(queue_file, queue)
                        except wikipedia.exceptions.DisambiguationError:
                            print(f"DisambiguationError sur {link}. Ignoré.")
                        except wikipedia.exceptions.PageError:
                            print(f"PageError : {link} n'existe pas.")
                        except Exception as e:
                            print(f"Erreur inattendue sur {link}: {e}")

                # Nettoyer le current_link après avoir traité la page
                clear_current_link(current_link_file, current_page)
                explored_pages.add(current_page)
                save_explored_pages(explored_file, explored_pages)

        except wikipedia.exceptions.DisambiguationError:
            print(f"DisambiguationError sur {current_page}. Ignoré.")
        except wikipedia.exceptions.PageError:
            print(f"PageError : {current_page} n'existe pas.")
        except Exception as e:
            print(f"Erreur inattendue sur {current_page}: {e}")

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