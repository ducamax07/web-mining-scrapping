import json
import os

def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    return {}

def save_explored_pages(file_path, explored_pages):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(list(explored_pages), file, ensure_ascii=False, indent=4)

link_file = 'linksA.json'
explored_file = 'exploredA.json'

links = load_data(link_file)
explored_pages = set(links.keys())
save_explored_pages(explored_file, explored_pages)

print(f"Explored pages saved: {len(explored_pages)} pages.")
