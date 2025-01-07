import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
data = pd.read_csv('SOFIFA.csv', sep=';', decimal=',')



#afficher le graphique nuage de points entre deux colonnes
def plot_correlation(data, x, y):
    plt.scatter(data[x], data[y], color="Blue",s=3, alpha=0.8) #s=taille des points, alpha=transparence
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Corrélation entre ' + x + ' et ' + y)
    plt.show()
plot_correlation(data, 'Age', 'Value')

#Histogramme d'une colonne
def plot_hist(data, x):
    plt.hist(data[x], bins=10, edgecolor='black') #bins = nombre de barres, edgecolor pour les lignes noires
    plt.xlabel(x)
    plt.ylabel('Nombre de joueurs')
    plt.title('Histogramme de ' + x)
    plt.show()

# Fonction pour créer un boxplot avec une échelle Y ajustée
def plot_boxplot_with_age_bins(data, x, y):
    # Créer des tranches d'âge
    bins = [0, 20, 25, 30, float('inf')]  # Définir les intervalles de tranches
    labels = ['<20', '20-25', '25-30', '>30']  # Noms des tranches
    data['Age_bins'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

    # Créer le boxplot
    plt.figure(figsize=(10, 6))  # Taille du graphique
    sns.boxplot(
        data=data,
        x='Age_bins',  # Tranches d'âge comme axe X
        y=x,  # Variable Y (Value)
        palette="Blues",  # Palette de couleurs
        showfliers=True,  # Afficher les outliers
        width=0.6  # Largeur des boxplots
    )
    plt.title(f'Distribution de {x} par {y} (Tranches d\'âge)', fontsize=14, weight='bold')
    plt.xlabel(y, fontsize=12, weight='bold')
    plt.ylabel(x, fontsize=12, weight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Ajouter une grille discrète
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
# Remplacez 'data' par votre DataFrame et 'Value' comme valeur à analyser
plot_boxplot_with_age_bins(data, 'Value', 'Age')
#Graphique en camembert
def plot_pie(data, x,title):
    plt.figure(figsize=(4, 4))  # Taille de la figure
    plt.title(title, fontsize=12)  # Titre
    data[x].value_counts().plot.pie(
        autopct='%1.1f%%',  # Format des pourcentages
        startangle=90,  # Angle de départ
        colors=plt.cm.Blues(np.linspace(0.3, 0.9, data[x].nunique())),  # Palette de tons bleus
        textprops={'fontsize': 10}  # Taille des textes
    )
    plt.ylabel('')  # Supprimer l'étiquette de l'axe Y
    plt.show()


#afficher la régréssion linéaire entre colonnes Value et Age
def plot_regression(data, x, y):
    # Calculer les coefficients de la régression linéaire
    slope, intercept = np.polyfit(data[x], data[y], 1)
    # Afficher les points de données
    plt.scatter(data[x], data[y], label='Données')
    # Afficher la ligne de régression
    plt.plot(data[x], slope * data[x] + intercept, color='red', label='Régression linéaire')   
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Régression linéaire entre ' + x + ' et ' + y)
    plt.legend()
    plt.show()
    # Afficher les coefficients
    print(f'Coefficient de régression (pente): {slope}')
    print(f'Ordonnée à l\'origine: {intercept}')

#fichier csv avec les infos de chaque colonne (min, max, moyenne, médiane, écart-type,quartiles)
def save_column_stats_to_excel(data, filename):
    # Sélectionner uniquement les colonnes numériques
    numeric_data = data.select_dtypes(include=['number'])
    
    # Vérifier qu'il y a des colonnes numériques
    if numeric_data.empty:
        print("Aucune colonne numérique dans les données.")
        return
    # deselctionner la colonne ID, Joine
    numeric_data = numeric_data.drop(columns=['ID', 'Joined'])
    
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
    
    # Réorganiser les colonnes pour un ordre logique
    stats = stats[['min', 'max', 'moyenne', 'median', 'écart-type', 'quartile_1', 'quartile_3']]
    
    # Sauvegarder les statistiques dans un fichier Excel
    stats.to_excel(filename, index=True)
    print(f"Les statistiques descriptives ont été sauvegardées dans : {filename}")



"""# Fonction pour afficher un tableau de corrélations avec Spearman
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
    
    # Enregistrer la matrice dans un fichier Excel ou CSV (optionnel)
    if filename:
        spearman_corr_matrix.to_csv(filename, index=True)
        print(f"Matrice des corrélations de Spearman sauvegardée dans : {filename}")
    
    return spearman_corr_matrix
spearman_corr_matrix = spearman_correlation_table(data, filename="/Users/quentindubart/Library/Mobile Documents/com~apple~CloudDocs/Data_mining_projet/Données/spearman_correlation_matrix.csv")

def plot_spearman_correlation_heatmap(corr_matrix):
    plt.figure(figsize=(10, 8))

    # Créer un masque pour cacher la moitié supérieure
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Heatmap avec le masque appliqué
    sns.heatmap(
        corr_matrix,
        mask=mask,  # Masquer la moitié supérieure
        annot=True,  # Afficher les valeurs
        cmap='Blues',  # Palette de couleurs
        fmt=".2f",  # Format des valeurs numériques
        linewidths=0.5,  # Espacement entre les cases
        cbar_kws={"shrink": 0.8}  # Ajuster la barre des couleurs
    )

    plt.title("Matrice des corrélations de Spearman (Triangulaire Inférieure)", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()
plot_spearman_correlation_heatmap(spearman_corr_matrix)
"""