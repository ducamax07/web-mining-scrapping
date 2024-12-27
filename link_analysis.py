from networkx import hits
import numpy as np
import networkx as nx
# Charger le fichier GML et créer la matrice d'adjacence
def create_numpy_adjacency_matrix(gml_file):
    graph = nx.read_gml(gml_file)
    adjacency_matrix = nx.to_numpy_array(graph)
    return adjacency_matrix, list(graph.nodes())

# Fonction pour calculer les voisins communs
def voisins_communs(matrix):
    # Initialiser une matrice pour stocker les voisins communs
    voisins = np.zeros(matrix.shape)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            voisins[i, j] = np.sum(np.multiply(matrix[i, :], matrix[j, :]))
    return voisins

# Fonction pour calculer l'indice d'attachement préférentiel entre tous les paires de nœuds
def preferential_attachment(matrix):
    degrees = np.sum(matrix, axis=1)
    pref_attach = np.zeros(matrix.shape) 
    # Parcourir les paires de nœuds
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            # L'attachement préférentiel entre deux nœuds est le produit de leurs degrés
            pref_attach[i, j] = degrees[i] * degrees[j]
    return pref_attach 

# Fonction pour calculer le coefficient cosinus entre tous les paires de nœuds
def cosine_similarity(matrix):
    cosine_sim = np.zeros(matrix.shape)
    # Parcourir les paires de nœuds
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            # Calculer le produit scalaire des deux vecteurs correspondant aux nœuds i et j
            dot_product = np.dot(matrix[i, :], matrix[j, :])
            # Calculer les normes des vecteurs pour normalisation
            norm_i = np.linalg.norm(matrix[i, :])
            norm_j = np.linalg.norm(matrix[j, :])         
            # Calculer la similarité cosinus (éviter les divisions par zéro)
            if norm_i > 0 and norm_j > 0:
                cosine_sim[i, j] = dot_product / (norm_i * norm_j)
            else:
                cosine_sim[i, j] = 0  # Si un vecteur a une norme nulle, la similarité est 0
    return cosine_sim

def jaccard_similarity(matrix):
    n = matrix.shape[0]  # Nombre de nœuds
    degrees = np.sum(matrix, axis=1)  # Degré de chaque nœud
    sim_common = np.dot(matrix, matrix.T)  # Nombre de voisins communs entre chaque paire de nœuds
    sim_jac = np.zeros((n, n))  
    # Parcourir chaque paire de nœuds
    for i in range(n):
        for k in range(n):
            # Calculer la mesure de Jaccard (éviter la division par zéro)
            denominator = degrees[i] + degrees[k] - sim_common[i, k]
            if denominator > 0:
                sim_jac[i, k] = sim_common[i, k] / denominator
            else:
                sim_jac[i, k] = 0   
    return sim_jac


# Fonction pour calculer la similarité de Dice entre tous les paires de nœuds
def dice_similarity(matrix):
    n = matrix.shape[0]  # Nombre de nœuds
    degrees = np.sum(matrix, axis=1)  # Degré de chaque nœud
    sim_common = np.dot(matrix, matrix.T)  # Nombre de voisins communs entre chaque paire de nœuds
    sim_dice = np.zeros((n, n))
    
    # Parcourir chaque paire de nœuds
    for i in range(n):
        for k in range(n):
            # Calculer la mesure de Dice (éviter la division par zéro)
            denominator = degrees[i] + degrees[k]
            if denominator > 0:
                sim_dice[i, k] = 2 * sim_common[i, k] / denominator
            else:
                sim_dice[i, k] = 0  
    return sim_dice

def Katz(graph, alpha):
    #Calculer la matrice de Katz
    A = graph
    n = A.shape[0] # nombre de noeuds
    I = np.identity(n) # matrice identité
    Katz = np.linalg.inv(I - alpha*A) - I # matrice de Katz, on calcule d'abord l'inverse de I - alpha*A puis on soustrait I
    print(f"Ceci est la matrice de Katz :\n {Katz}\n")


def transition_proba(graph):
    A = graph
    n = A.shape[0] # nombre de noeuds
    degrees = [sum(A[i]) for i in range(n)] # degré de chaque noeud
    transition_graph = np.zeros((n,n))
    """
    au début je n'avais pas mis (transition_graph) mais c'est nécessaire sinon on modifie la matrice A pendant
    le calcul et on obtient un matrice null
    """

    #Calculer la matrice de transition
    for i in range(len(A)):
        for k in range(len(A)):
            transition_graph[i,k] = A[i,k]/degrees[i]
    print(f"Ceci est la matrice de transition :\n {transition_graph}\n")

def matrice_FPT_et_CT(graph):
    #Claculer la pseudo-inverse de la matrice Laplacienne
    A = graph
    n = A.shape[0] # nombre de noeuds
    degrees = [sum(A[i]) for i in range(n)] # degré de chaque noeud
    D = np.diag(degrees) # matrice diagonale des degrés
    L = D - A # matrice Laplacienne
    eeT = np.ones((n,n)) # matrice de 1
    Lplus = np.linalg.inv((L - eeT/n)) + (eeT/n)
    # print(f"ceci est la pseudo-inverse de la matrice Laplacienne :\n {Lplus}\n")

    #Calculer la matrice de FPT
    FPT = np.zeros((n,n))
    for i in range(len(A)):
        for k in range(len(A)):
            if i != k:
                sum_term = 0
                for j in range(len(A)):
                    sum_term += (Lplus[i,j] - Lplus[i,k] - Lplus[k,j] + Lplus[k,k])*D[j,j]
                FPT[k,i] = sum_term
    FPT_transpose = np.transpose(FPT)
    print(f"Ceci est la matrice de FPT :\n {FPT_transpose}\n")

    CT = FPT_transpose + FPT
    print(f"Ceci est la matrice de CT :\n {CT}\n")

def scores_Hub_Authority(graph):
    A = graph
    Hub_score = hits.fit_predict(A)[0]
    print(f"Ceci est le score de Hub :\n {Hub_score}\n")