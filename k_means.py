# IA - Labo 01 - K-Means
# Ruben Wihler
# 04.09.2026

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Notes:
#
# Cluster: "Groupe" qui contient les points (echantillons) qui sont proche (espace des features)
#
# Centroid: Point (espace des features) qui represente un cluster 
#   -> Ici la position moyenne des points du cluster
#


# Premiere version pas tres performente a cause des loop python
def k_means(data, k, max_iter=100):
    # On choisis k points aleatoire de data comme centroids intial
    rng = np.random.default_rng()
    centroids = rng.choice(data, size=k, replace=False)
    clusters = {} # contient pour chaque points l'idx du clusters au quel il appartient
    variances = [] #variences intra-class au fil des iterations
    it = 0 # uniquement pour limiter le max d'iterations

    while it < max_iter:
        #assignation
        new_clusters = {}

        #itere sur les points
        for idx, pt in enumerate(data):
            #tableau des distances entre le points et tous les centroids
            distances = np.linalg.norm(pt - centroids, axis=1)
            cluster_idx = np.argmin(distances) # prends le cluster le plus proche
            new_clusters[idx] = cluster_idx

        #condition d'arret : si les association points/clusters n'ont pas changer 
        if new_clusters == clusters:
            break
        clusters = new_clusters

        #mise a jours des centroids o(K*N) : Goulot d'etranglement - A OPTI car boucle lent en python
        new_centroids = np.zeros_like(centroids)
        for c in range(k): #pour chaque centroid c (index)
            # Remplie un tableau des points/data associer au cluster : PAS TRES PERFORMENTS
            pts_in = [data[idx] for idx, cluster in clusters.items() if cluster == c]
            if pts_in:
                new_centroids[c] = np.mean(pts_in, axis=0) # Le centroid c devient la moyenne des points du cluster
        centroids = new_centroids

        #fonction objectif
        sum = 0
        # itere sur les points (idx) et leur cluster associer (cluster_idx)
        for idx, cluster_idx in clusters.items():
            pt = data[idx] #recupere le points
            centroid = centroids[cluster_idx] #recupere le centroid associer au cluster
            sum += np.sum((pt - centroid) ** 2) #ajoute la somme des erreurs carrer sur toutes les dimmension/features
        L = sum / len(data) # diviser par N (nb data) pour obtenir varience intra-classe
        variances.append(L) # stocke pour l'affichage de la var. au file des iterations
        it += 1

    return clusters, centroids, variances


# Version ameliorer/corriger (a l'aide de l'IA)
# Utilise la vectorisation numpy pour un gain de perf enorme
def k_means_opt(data, k, max_iter=100):
    # data : numpy array de dimension (N, D)
    #      : N -> nombre de points
    #      : D -> nombre de features

    #initialisation
    rng = np.random.default_rng()
    centroids = rng.choice(data, size=k, replace=False)
    clusters = np.zeros(data.shape[0]) # CHANGEMENT PRINCIPALE : Simple tableau qui contient 
    variances = []                     #                  l'idx du cluster pour chaque point
    it = 0

    while it < max_iter:
        #assignation
        # Broadcasting pour calculer distances entre tt les pts et tt les centroids simultanement
        # data[:, np.newaxis] crée une vue de shape (N, 1, D)
        # centroids crée implicitement une vue de shape (1, k, D)
        # La soustraction donne une matrice de shape (N, k, D) contenant les différences de coordonnées
        diff = data[:, np.newaxis] - centroids # Honnetement c'est de la magie noir pour moi hahaha
        distances = np.linalg.norm(diff, axis=2) # grille 2D ou val(i,j) donne distance entre point i et centroid j
        new_clusters = np.argmin(distances, axis=1) # on ballaye les centroids (axis=1) pour chaque point et prend le plus petit/proche
                                                    # on obtient donc un tableau 1D de taille N qui contient les clusters associer pour chaque pts
        
        #condition d'arret
        if np.array_equal(clusters, new_clusters):
            break
        clusters = new_clusters

        # mis a jours des centroids
        # data[clusters == c].mean(axis=0) : on prends la moyenne des points appatrenant au cluster c
        # (le if np.sum(clusters == c) > 0 else centroids[c] est une securite au cas ou le cluster est vide)
        new_centroids = np.array([
            data[clusters == c].mean(axis=0) if np.sum(clusters == c) > 0 else centroids[c]
            for c in range(k)
        ])
        centroids = new_centroids

        # fonction objectif
        # centroids[clusters] cree un tableau ou chque pt est remplacer par les coord de son centroide
        L = np.sum((data - centroids[clusters]) ** 2) / len(data)
        variances.append(L)
        it += 1

    return clusters, centroids, variances

#fonction de plot (Faite par IA ;)
def plot(data, clusters, centroids, variances):
    # On crée une figure avec 1 ligne et 2 colonnes pour avoir les deux graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- GRAPHIQUE 1 : Les clusters ---
    # On convertit ton dictionnaire `clusters` en une liste de couleurs/labels ordonnée
    labels = [clusters[i] for i in range(len(data))]
    
    # Affichage des points (c=labels permet d'assigner une couleur par groupe)
    ax1.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
    
    # Affichage des centroïdes (en croix rouge, plus gros)
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroïdes')
    
    ax1.set_title("Résultat du K-Means")
    ax1.set_xlabel("Grade 1")
    ax1.set_ylabel("Grade 2")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- GRAPHIQUE 2 : La fonction objectif ---
    # On affiche l'évolution de la variance L au fil des itérations
    ax2.plot(range(1, len(variances) + 1), variances, marker='o', linestyle='-', color='b')
    
    ax2.set_title("Évolution de la fonction objectif")
    ax2.set_xlabel("Nombre d'itérations")
    ax2.set_ylabel("Variance totale (L)")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


#quasi meme algo que k-means
def k_means_iris_eval(X, y, k):
    rng = np.random.default_rng()
    centroids = rng.choice(X, size=k, replace=False)
    clusters = np.zeros(X.shape[0])
    it = 1

    while True:
        #assignation
        diff = X[:, np.newaxis] - centroids
        distances = np.linalg.norm(diff, axis=2)
        new_clusters = np.argmin(distances, axis=1)

        #condition d'arret
        if np.array_equal(clusters, new_clusters):
            print(f"=== Convergence atteinte à l'itération {it-1} ===")
            break
        clusters = new_clusters

        # mis a jours des centroids
        centroids = np.array([
            X[clusters == c].mean(axis=0) if np.sum(clusters == c) > 0 else centroids[c]
            for c in range(k)
        ])

        # TAUX DE CLASSIFICATION CORRECTE
        print(f"--- Itération {it} ---")
        total_correct = 0

        for c in range(k): # Itere sur les clusters
            # mask pour isoler les pts du cluster c
            mask = (clusters == c)
            labels_in_cluster = y[mask] #donne les classe qui appartienne a ce cluster

            if len(labels_in_cluster) > 0:
                # grace a Counter, on trouve l'etiquette la plus frequente instantanement
                counter = Counter(labels_in_cluster)
                majority_class, count = counter.most_common(1)[0] #ont prend uniquement la plus commune
                total_correct += count #ajoute le nb de correct

                print(f"  Cluster {c} : Majorité = '{majority_class}' -> ({count}/{len(labels_in_cluster)} corrects)")
            else:
                print(f"  Cluster {c} : Vide")

        # calcule du taux global : (Somme des corrects / len(dataset))
        accuracy = total_correct / len(y)
        print(f"  => Taux de classification correcte global : {accuracy:.2%}\n")

        it += 1

    return clusters, centroids


#Execution
if __name__ == "__main__":
    #Dataset Iris - Taux de classification correcte
    df_iris = pd.read_csv('iris.csv', header=None)
    X_iris = df_iris.iloc[:, :-1].values
    y_iris = df_iris.iloc[:, -1].values
    k_means_iris_eval(X_iris, y_iris, k=5) # K=3 car 3 espece de fleurs (minimum 3, mais ont peut augmenter)

    #Dataset Student - Visualisation
    df_student = pd.read_csv('student-data-test.csv')
    X_student = df_student[['grade_1', 'grade_2']].values
    clusters, centroids, variances = k_means_opt(X_student, k=4)
    plot(X_student, clusters, centroids, variances)

