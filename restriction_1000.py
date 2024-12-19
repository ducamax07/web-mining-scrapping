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

# Préparer les stopwords et le stemmer (radicalisation)
stop_words = list(set(stopwords.words('english'))) + ["'s"]
stem = nltk.stem.SnowballStemmer("english")


# Fonction pour extraire les tokens 
def extract_tokens(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  #fonction régulière
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stem.stem(token) for token in tokens]
    return tokens

# Fonction pour obtenir les n tokens les plus fréquents
def get_top_tokens(content, n=20):
    tokens = extract_tokens(content)
    token_counts = {}
    for token in tokens:
        if token in token_counts:
            token_counts[token] += 1
        else:
            token_counts[token] = 1
    sorted_tokens = sorted(token_counts.items(), key=lambda item: item[1], reverse=True)
    top_tokens = [item[0] for item in sorted_tokens[:n]]
    return set(top_tokens)

def calculate_dynamic_threshold(page_summary, base_threshold=5):
    
    # Calcule un seuil dynamique de pertinence en fonction de la longueur du résumé.
    
    text_length = len(page_summary.split())
    
    # Ajuster le seuil en fonction de la longueur du texte (plus de texte, plus de tokens nécessaires)
    dynamic_threshold = base_threshold + (text_length // 100)  # Augmenter de 1 pour chaque 100 mots
    return dynamic_threshold

def is_relevant_based_on_top_tokens(top_tokens, linked_summary, page_name, base_threshold=5):
    # Extraire les tokens et rendre linked_tokens unique
    linked_tokens = extract_tokens(linked_summary)  # Assure unicité des tokens extraits
    unique_linked_tokens = []
    for token in linked_tokens:
        if token in top_tokens:
            if token not in unique_linked_tokens:
                unique_linked_tokens.append(token)
    dynamic_threshold = calculate_dynamic_threshold(linked_summary, base_threshold)
    
    # Comparer avec le seuil
    if len(unique_linked_tokens) >= dynamic_threshold:
        print(f"Page validée : {page_name}")
        return True
    else:
        #print(f"Page non validée : {page_name}")
        return False

# Sauvegarder les données dans un fichier JSON
def save_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    return {}

# Fonction pour stocker les contenus des pages
def store_content(page_title, page_content, content_storage, content_file):
    if page_title not in content_storage:
        content_storage[page_title] = page_content
        save_data(content_file, content_storage)

# Fonction pour stocker les liens entre les pages
def store_links(source_page, linked_page, link_storage, link_file):
    if source_page not in link_storage:
        link_storage[source_page] = []
    if linked_page not in link_storage[source_page]:
        link_storage[source_page].append(linked_page)
        save_data(link_file, link_storage)  

# Fonction pour sauvegarder la queue
def save_queue(file_path, queue):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(queue, file, ensure_ascii=False, indent=4)

# Fonction pour charger la queue
def load_queue(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    return []

# Sauvegarder le lien en cours de traitement
def save_current_link(file_path, current_page, current_link):
    last = {current_page: current_link}
    save_data(file_path, last)
# Charger le lien en cours de traitement
def load_current_link(file_path, current_page):
    if os.path.exists(file_path):
        last = load_data(file_path)
        return last.get(current_page)
    return None
# Supprimer le lien en cours de traitement après achèvement
def clear_current_link(file_path, current_page):
    if os.path.exists(file_path):
        last = load_data(file_path)
        if current_page in last:
            del last[current_page]
            save_data(file_path, last)

# Fonction principale pour le scraping BFS
def bfs_scrape(start_page, max_depth=3, content_file='content.json', link_file='links.json', queue_file='queue.json', current_link_file='current_link.json'):
    visited = set()  # Set des pages visitées
    queue = load_queue(queue_file)  # Charger la queue sauvegardée
    visited_count = 0

    # Si la queue est vide, initialiser avec la page de départ
    if not queue:
        queue = [(start_page, 0)]
        save_queue(queue_file, queue)  # Sauvegarder l'état initial

    content_storage = load_data(content_file)
    link_storage = load_data(link_file)

    # Obtenir les top tokens de la page de départ
    main_page = wikipedia.WikipediaPage(start_page)
    top_tokens = get_top_tokens(main_page.content, n=20)
    print(f"Top Tokens for {start_page}: {top_tokens}\n")

    while queue:
        # Retirer la première page de la queue
        current_page, depth = queue.pop(0)
        save_queue(queue_file, queue)  # Sauvegarder la queue mise à jour

        # Vérifier si la profondeur maximale est atteinte
        if depth > max_depth:
            print(f"Profondeur max atteinte pour {current_page}.")
            continue

        # Marquer la page comme visitée
        if current_page in visited:
            print(f"{current_page} déjà visité.")
            continue
        visited.add(current_page)

        try:
            # Charger la page actuelle
            print(f"Scraping: {current_page} (Depth {depth})")
            page = wikipedia.WikipediaPage(current_page)
            visited_count += 1

            # Charger le lien en cours ou initialiser la boucle
            start_link = load_current_link(current_link_file, current_page)
            links_to_process = page.links
            if start_link:
                # Reprendre après le dernier lien traité
                links_to_process = links_to_process[links_to_process.index(start_link) + 1:]

            # Vérifier si la page est pertinente
            if is_relevant_based_on_top_tokens(top_tokens, page.summary, current_page, base_threshold=5):
                # Stocker le contenu de la page
                store_content(current_page, page.content, content_storage, content_file)

                # Parcourir les liens pertinents et les stocker
                for link in links_to_process:
                    save_current_link(current_link_file, current_page, link)  # Sauvegarder le lien courant
                    if link not in visited:
                        try:
                            linked_page = wikipedia.WikipediaPage(link)
                            
                            # Vérifier si le lien est pertinent
                            if is_relevant_based_on_top_tokens(top_tokens, linked_page.summary, link, base_threshold=5):
                                # Stocker le lien pertinent
                                store_links(current_page, link, link_storage, link_file)
                                # Stocker directement le contenu de la page liée
                                store_content(link, linked_page.content, content_storage, content_file)
                                # Ajouter à la file d'attente
                                queue.append((link, depth + 1))
                                save_queue(queue_file, queue)

                        except wikipedia.exceptions.DisambiguationError:
                            print(f"DisambiguationError: {link} has multiple meanings.")
                        except wikipedia.exceptions.PageError:
                            print(f"PageError: {link} does not exist.")
                        except wikipedia.exceptions.WikipediaException as e:
                            print(f"WikipediaException: {link} caused an error. Details: {e}")

                clear_current_link(current_link_file, current_page)  # Nettoyer après la boucle

        except wikipedia.exceptions.DisambiguationError:
            print(f"DisambiguationError: {current_page} has multiple meanings.")
        except wikipedia.exceptions.PageError:
            print(f"PageError: {current_page} does not exist.")
        except wikipedia.exceptions.WikipediaException as e:
            print(f"WikipediaException: {current_page} caused an error. Details: {e}")

    print(f"Scraping Finished! Visited {visited_count} pages.")


# Lancer le scraping
bfs_scrape("Diversity (business)", max_depth=1, content_file='content2.json', link_file='links2.json', queue_file='queue2.json', current_link_file='current_link2.json')