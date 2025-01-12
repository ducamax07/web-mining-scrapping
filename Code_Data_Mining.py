import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
data = pd.read_csv('/Users/quentindubart/Library/Mobile Documents/com~apple~CloudDocs/Data_mining_projet/Données/Normalisation/SOFIFAZ___.csv', sep=';', decimal=',',encoding="utf-8")
#decoupe en categories exemple
bins = [0, 20, 25, 30, float('inf')]  
labels = ['<20', '20-25', '25-30', '>30']  # Noms des tranches
data['Age_bins'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

#Histogramme d'une colonne
def plot_hist(data, x):
    plt.hist(data[x], bins=10, edgecolor='black')
    plt.xlabel(x)
    plt.ylabel('Nombre de joueurs')
    plt.title('Histogramme de ' + x)
    plt.show()

#Boite à moustaches
def plot_boxplot(data, x, y):
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=data,
        x=x, 
        y=y,  
        palette="Blues", 
        showfliers=True,
        width=0.6
    )
    plt.title(f'Distribution de {y} par {x}', fontsize=14, weight='bold')
    plt.xlabel(x, fontsize=12, weight='bold')
    plt.ylabel(y, fontsize=12, weight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Ajouter une grille discrète
    plt.tight_layout()
    plt.show()

#Graphique en camembert
def plot_pie(data, x,title):
    plt.figure(figsize=(4, 4))  
    plt.title(title, fontsize=12) 
    data[x].value_counts().plot.pie(
        autopct='%1.1f%%',  # Format des pourcentages
        startangle=90, 
        colors=plt.cm.Blues(np.linspace(0.3, 0.9, data[x].nunique())),  # Palette de tons bleus
        textprops={'fontsize': 10} 
    )
    plt.ylabel('') 
    plt.show()

#fichier csv avec les infos de chaque colonne (min, max, moyenne, médiane, écart-type,quartiles)
def save_column_stats_to_excel(data, filename):
    # Sélectionner uniquement les colonnes numériques
    numeric_data = data.select_dtypes(include=['number'])
    # deselctionner la colonne ID
    numeric_data = numeric_data.drop(columns=['ID'])
    
    # Calcul des statistiques descriptives
    stats = numeric_data.describe().T  # Transposer pour avoir une ligne par variable
    stats['median'] = numeric_data.median()  # Ajouter la médiane
    stats['quartile_1'] = numeric_data.quantile(0.25)  # Ajouter le 1er quartile
    stats['quartile_3'] = numeric_data.quantile(0.75)  # Ajouter le 3e quartile
    # Renommer les colonnes pour des noms en français
    stats = stats.rename(columns={
        'mean': 'moyenne',
        'std': 'écart-type',
        'min': 'min',
        'max': 'max'
    })
    # Réorganiser les colonnes 
    stats = stats[['min', 'max', 'moyenne', 'median', 'écart-type', 'quartile_1', 'quartile_3']]
    
    # Sauvegarder les statistiques dans un fichier Excel
    stats.to_excel(filename, index=True)
    print(f"Les statistiques descriptives ont été sauvegardées dans : {filename}")

# Fonction pour afficher un tableau de corrélations avec Spearman
def spearman_correlation_table(data, filename=None):
    # Sélectionner uniquement les colonnes numériques
    numeric_data = data.select_dtypes(include=['number'])
    #retirer la colonne ID
    numeric_data = numeric_data.drop(columns=['ID'])
    # Calculer la matrice de corrélation de Spearman
    spearman_corr_matrix = numeric_data.corr(method='spearman')
    # Afficher la matrice sous forme de tableau
    print("Matrice des corrélations de Spearman :")
    print(spearman_corr_matrix)
    if filename:
        spearman_corr_matrix.to_csv(filename, index=True) 
    return spearman_corr_matrix
def plot_spearman_correlation_heatmap(corr_matrix):
    plt.figure(figsize=(10, 8))
    # Créer un masque pour cacher la moitié supérieure
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,  
        annot=True, 
        cmap='Blues',  # Palette de couleurs
        fmt=".2f",  # Format des valeurs numériques
        linewidths=0.5,  # Espacement entre les cases
        cbar_kws={"shrink": 0.8}  # Ajuster la barre des couleurs
    )

    plt.title("Matrice des corrélations de Spearman (Triangulaire Inférieure)", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()
plot_spearman_correlation_heatmap(spearman_corr_matrix)
