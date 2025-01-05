# MLSMM2153 Web Mining - Diversité et Inclusion

## Présentation du projet

Ce projet consiste à élaborer une étude sur un sujet précis à partir d'une méthode de Web Scraping. Elle utilise des analyses méthodologiques comprenant du text mining et du link analysis.

Dans le cadre de ce travail, nous l'avons utilisé pour étudier les impacts de la diversité et de l'inclusion sur une entreprise, bien que ce projet puisse être utilisé pour tout type de sujet.

Il est important de correctement suivre la procédure prescrite suivante afin d'utiliser nos outils de façon optimale. À noter que le code est initialement formaté pour le traitement de données en anglais.

## Présentation du répertoire

En ouvrant le repository dans votre éditeur de texte, vous découvrirez différents fichiers.

Parmi ceux-ci, les 3 fichiers à manipuler sont les codes Python. Ceux-ci sont respectivement nommés "data_collection_B.py", "text_mining_B.ipynb" et "links_analysis_B.ipynb". Ils sont à utiliser dans un environnement Python, le module Jupyter Notebook est à installer.Les fichiers "contentB.json" et "linksB.json" sont directement issus de notre scraping sur le sujet Diversité et Inclusion.
Les répertoires "output", "outputgml", "outputLA", et "cluster_links" sont également inclus.

## Procédure à suivre

### 1. Prérequis

Avant de démarrer, veuillez vous assurer d'avoir installé les bibliothèques suivantes :'pip install wikipedia-api, nltk, matplotlib, networkx, pandas, numpy, scikit-learn, seaborn, jupyter, ipywidgets'

Les packages à importer sont à chaque fois stipulés au début de chaque code respectif. Ils doivent être installés avant toute procédure.

### 2. Utilisez data_collection_B.py

Ce code est au format Python, il s'exécute donc entièrement d'un coup. Celui-ci a pour objectif de collecter les données sur la base d'une page Wikipédia spécifique. Les pages sélectionnées sont analysées sur la base d'une comparaison entre leur résumé et la page initiale.

Toute modification concernant les méthodes de sélection est à implémenter directement dans les paramètres de la fonction bfs_scrape, que ce soit les fichiers de sortie, la profondeur de recherche ou la base du nombre de mots dans l'analyse de similarité.

Les outputs sont renvoyés dans les fichiers "contentB.json" et "linksB.json".

### 3. Utilisez text_mining_B.ipynb

Ce code est rédigé via Jupyter Notebook, chaque bout de code se lance donc de manière indépendante. Ce code se divise en deux grandes parties, une contenant les fonctions (à lancer en entier), l'autre concernant leurs appels.

Avant de lancer un appel de fonction spécifique, assurez-vous d'avoir lancé les fonctions concernant le chargement des fichiers et l'entraînement des modèles.

Les outputs apparaîtront dans le dossier "output" à l'exception des outputs au format gml qui seront classés dans le dossier "outputgml".

### 4. Utiliser links_analysis_B.ipynb

Ce code est rédigé via Jupyter Notebook, chaque bout de code se lance donc de manière indépendante. Ce code se divise en deux grandes parties, une contenant les fonctions (à lancer en entier), l'autre concernant leurs appels.

Celui-ci fonctionne à partir du fichier "outputgml/graph.gml", il est donc important de l'avoir créé dans la partie précédente et de le charger dans cette partie-ci en premier.
Les outputs apparaîtront dans le dossier "outputLA" et sont formatés afin d'être utilisés avec Gephi.

## Crédits

Ce projet a été rédigé sur l'éditeur de texte Visual Studio Code.
Ce projet a été réalisé dans le cadre d'un cours à l'UCLouvain FUCaM Mons.
