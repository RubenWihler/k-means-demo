# 🚀 K-Means Clustering from Scratch

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Optimized-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Clustering-orange.svg)

## 📌 À propos de ce projet
Ce dépôt contient une implémentation complète de l'algorithme d'apprentissage non supervisé **K-Means**, codée de A à Z en Python sans utiliser de librairies de Machine Learning préconçues (comme scikit-learn). 

Ce projet a été réalisé dans le cadre d'un laboratoire d'Intelligence Artificielle et met l'accent sur la compréhension interne des mathématiques du clustering et l'optimisation des calculs matriciels.

## ✨ Fonctionnalités implémentées

1. **Algorithme K-Means Standard** :
   * Assignation des points aux centroïdes via la **Distance Euclidienne**.
   * Recalcul itératif de la position moyenne des centroïdes.
   * Calcul de la fonction objectif (variance intra-cluster $L$).
2. **Optimisation Vectorielle (NumPy)** :
   * Remplacement des boucles imbriquées par le **Broadcasting** de tenseurs.
   * Gain de performance drastique ($O(K \times N)$ calculé simultanément en mémoire contiguë C).
3. **Évaluation et Suivi** :
   * **Données 2D (Student)** : Visualisation de la formation des clusters et de la courbe de descente de la fonction objectif à l'aide de `matplotlib`.
   * **Données Multi-dimensionnelles (Iris)** : Évaluation automatique du taux de classification correcte à chaque itération via un système de vote majoritaire.

## 📂 Jeux de données utilisés
* `student-data-test.csv` : Données en 2 dimensions (notes d'étudiants) idéales pour la visualisation spatiale des centroïdes.
* `iris.csv` : Le célèbre dataset d'Anderson (4 dimensions, 3 classes d'iris) utilisé ici pour valider la précision de l'algorithme à l'aveugle.

## 🛠️ Technologies
* **Python** (Cœur de la logique)
* **NumPy** (Calcul tensoriel, Broadcasting, Algèbre linéaire)
* **Pandas** (Extraction et manipulation des CSV)
* **Matplotlib** (Visualisation des graphiques et courbes de perte)

## 🚀 Comment exécuter le projet

Assurez-vous d'avoir installé les dépendances requises :
```bash
pip install numpy pandas matplotlib
```

Lancez simplement le script principal pour voir le K-Means en action :
```bash
python k_means.py
```

Le script affichera la progression du taux de classification dans le terminal (pour le dataset Iris) et générera une fenêtre graphique avec le résultat du clustering (pour le dataset Student).

## 👨‍💻 Auteur
**Ruben Wihler** - *Laboratoire d'IA - 2026*
