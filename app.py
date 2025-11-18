import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Système de Recommandation SVD",
    page_icon="🎬",
    layout="wide"
)

# Titre principal
st.title("🎥 Système de Recommandation SVD - MovieLens")
st.markdown("---")

# Fonctionnalités de caching pour la performance
@st.cache_data
def load_data(file_path):
    """Étape 1 & 2 : Charger et extraire les 30 000 premiers quadruplets"""
    df = pd.read_csv(file_path, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
    df_extracted = df.head(30000).copy()
    return df_extracted

@st.cache_data
def preprocess_data(df):
    """Étape 3 & 4 : Trier et splitter les données"""
    # Trier par timestamp
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # Split 70/30
    train_size = 21000
    train_data = df_sorted.iloc[:train_size].copy()
    test_data = df_sorted.iloc[train_size:].copy()
    
    return train_data, test_data

@st.cache_data
def build_matrix(train_data):
    """Étape 4 : Construire la matrice user-movie"""
    n_users = train_data['userId'].nunique()
    n_movies = train_data['movieId'].nunique()
    
    # Mapping des IDs
    user_to_idx = {user: idx for idx, user in enumerate(sorted(train_data['userId'].unique()))}
    movie_to_idx = {movie: idx for idx, movie in enumerate(sorted(train_data['movieId'].unique()))}
    
    # Créer la matrice
    matrix = np.zeros((n_users, n_movies))
    
    for _, row in train_data.iterrows():
        user_idx = user_to_idx[row['userId']]
        movie_idx = movie_to_idx[row['movieId']]
        matrix[user_idx, movie_idx] = row['rating']
    
    return matrix, user_to_idx, movie_to_idx

@st.cache_data
def compute_svd(matrix, k_values):
    """Étape 5 : Appliquer SVD"""
    # Centrer la matrice
    user_means = np.mean(matrix, axis=1)
    matrix_centered = matrix - user_means.reshape(-1, 1)
    
    # SVD
    U, sigma, Vt = svds(matrix_centered, k=max(k_values))
    sigma = np.diag(sigma)
    
    return U, sigma, Vt, user_means, matrix_centered

def predict_svd(U, sigma, Vt, user_means, k):
    """Prédiction avec SVD et k facteurs latents"""
    pred_matrix = np.dot(np.dot(U[:, :k], sigma[:k, :k]), Vt[:k, :])
    return pred_matrix + user_means.reshape(-1, 1)

def predict_user_similarity(matrix, user_means, k=50):
    """Prédiction par similarité entre users (cosine similarity)"""
    n_users = matrix.shape[0]
    predictions = np.zeros_like(matrix)
    
    # Calculer similarité cosine
    for user in range(n_users):
        # Masquer les ratings non observés
        user_vector = matrix[user, :]
        observed_mask = user_vector > 0
        
        # Similarité avec tous les autres users
        sims = []
        for other in range(n_users):
            if other != user:
                other_vector = matrix[other, :]
                # Cosine similarity sur les items co-notés
                common_items = (user_vector > 0) & (other_vector > 0)
                if common_items.sum() > 0:
                    dot = np.dot(user_vector[common_items], other_vector[common_items])
                    norm_user = np.linalg.norm(user_vector[common_items])
                    norm_other = np.linalg.norm(other_vector[common_items])
                    sim = dot / (norm_user * norm_other) if norm_user * norm_other > 0 else 0
                    sims.append((sim, other_vector))
        
        # Prédiction comme moyenne pondérée
        if sims:
            sims.sort(reverse=True, key=lambda x: x[0])
            top_k = sims[:k]
            
            for item in range(matrix.shape[1]):
                if not observed_mask[item]:  # Prédire seulement les items non notés
                    weighted_sum = 0
                    sim_sum = 0
                    for sim, other_vector in top_k:
                        if other_vector[item] > 0:
                            weighted_sum += sim * other_vector[item]
                            sim_sum += abs(sim)
                    
                    if sim_sum > 0:
                        predictions[user, item] = weighted_sum / sim_sum
    
    return predictions

def predict_item_similarity(matrix, k=50):
    """Prédiction par similarité entre movies (item-based)"""
    n_movies = matrix.shape[1]
    predictions = np.zeros_like(matrix)
    
    # Transposer pour calculer similarité entre items
    item_matrix = matrix.T
    
    for item in range(n_movies):
        item_vector = item_matrix[item, :]
        observed_mask = item_vector > 0
        
        sims = []
        for other in range(n_movies):
            if other != item:
                other_vector = item_matrix[other, :]
                # Cosine similarity sur les users co-notants
                common_users = (item_vector > 0) & (other_vector > 0)
                if common_users.sum() > 0:
                    dot = np.dot(item_vector[common_users], other_vector[common_users])
                    norm_item = np.linalg.norm(item_vector[common_users])
                    norm_other = np.linalg.norm(other_vector[common_users])
                    sim = dot / (norm_item * norm_other) if norm_item * norm_other > 0 else 0
                    sims.append((sim, other_vector))
        
        # Prédiction
        if sims:
            sims.sort(reverse=True, key=lambda x: x[0])
            top_k = sims[:k]
            
            for user in range(matrix.shape[0]):
                if not matrix[user, item] > 0:  # Item non noté par cet user
                    weighted_sum = 0
                    sim_sum = 0
                    for sim, other_vector in top_k:
                        if other_vector[user] > 0:
                            weighted_sum += sim * other_vector[user]
                            sim_sum += abs(sim)
                    
                    if sim_sum > 0:
                        predictions[user, item] = weighted_sum / sim_sum
    
    return predictions

def evaluate_rmse(test_data, predictions, user_to_idx, movie_to_idx, train_data):
    """Étape 7 : Calculer RMSE"""
    y_true = []
    y_pred = []
    
    train_pairs = set(zip(train_data['userId'], train_data['movieId']))
    
    for _, row in test_data.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        
        # Vérifier si le couple est nouveau et existe dans la matrice
        if (user_id not in user_to_idx) or (movie_id not in movie_to_idx):
            continue
        
        if (user_id, movie_id) in train_pairs:
            continue
        
        user_idx = user_to_idx[user_id]
        movie_idx = movie_to_idx[movie_id]
        
        pred = predictions[user_idx, movie_idx]
        if pred != 0:  # Ignorer les prédictions non calculées
            y_true.append(row['rating'])
            y_pred.append(pred)
    
    if len(y_true) == 0:
        return float('inf'), [], [], []
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    errors = [(abs(true - pred), true, pred) for true, pred in zip(y_true, y_pred)]
    errors.sort()
    
    return rmse, y_true, y_pred, errors

def precision_at_k(test_data, predictions, user_to_idx, movie_to_idx, train_data, k=10):
    """Étape 8 : Calculer Precision@k"""
    # Regrouper les vrais ratings par user
    user_true_movies = defaultdict(set)
    train_pairs = set(zip(train_data['userId'], train_data['movieId']))
    
    for _, row in test_data.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        
        if (user_id in user_to_idx) and (movie_id in movie_to_idx):
            if (user_id, movie_id) not in train_pairs:
                if row['rating'] >= 4:  # Recommandations pertinentes (rating >= 4)
                    user_true_movies[user_id].add(movie_id)
    
    # Calculer Precision@k pour chaque user
    precisions = {}
    recommendations = {}
    
    for user_id in user_true_movies:
        user_idx = user_to_idx[user_id]
        
        # Obtenir les prédictions pour cet user
        user_preds = predictions[user_idx, :]
        
        # Items déjà notés dans le train
        user_train_items = set(train_data[train_data['userId'] == user_id]['movieId'])
        
        # Filtrer et classer
        candidate_items = []
        for movie_id in movie_to_idx:
            if movie_id not in user_train_items:
                movie_idx = movie_to_idx[movie_id]
                pred_rating = user_preds[movie_idx]
                if pred_rating > 0:
                    candidate_items.append((pred_rating, movie_id))
        
        # Top-k recommandations
        candidate_items.sort(reverse=True)
        top_k = [item for _, item in candidate_items[:k]]
        
        # Calculer precision
        relevant = user_true_movies[user_id]
        if len(top_k) > 0:
            precision = len(set(top_k) & relevant) / len(top_k)
        else:
            precision = 0
        
        precisions[user_id] = precision
        recommendations[user_id] = top_k
    
    return precisions, recommendations

# Interface Streamlit
def main():
    st.sidebar.header("📁 Configuration")
    
    # Upload du fichier
    uploaded_file = st.sidebar.file_uploader("Charger MovieLens (ratings.csv)", type=['data'])
    
    if uploaded_file is None:
        st.info("Veuillez charger le fichier `ratings.csv` du dataset MovieLens dans la barre latérale.")
        st.markdown("""
        **Instructions :**
        1. Téléchargez MovieLens depuis [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
        3. Chargez le fichier `ratings.csv` dans la barre latérale
        """)
        return
    
    # Chargement des données
    with st.spinner('Chargement des données...'):
        df = load_data(uploaded_file)
        st.success(f"✅ Dataset chargé : {len(df)} lignes")
    
    # Prétraitement
    train_data, test_data = preprocess_data(df)
    st.success(f"✅ Données split : {len(train_data)} train, {len(test_data)} test")
    
    # Construction de la matrice
    matrix, user_to_idx, movie_to_idx = build_matrix(train_data)
    st.success(f"✅ Matrice construite : {matrix.shape[0]} users × {matrix.shape[1]} movies")
    
    # Calcul SVD
    k_values = [3, 5]
    with st.spinner('Calcul SVD en cours...'):
        U, sigma, Vt, user_means, matrix_centered = compute_svd(matrix, k_values)
        st.success("✅ Décomposition SVD calculée")
    
    # Zone de visualisation
    st.header("📊 Résultats")
    
    # Tabs pour organiser les résultats
    tabs = st.tabs(["📈 RMSE & Prédictions", "🎯 Précision@10", "📋 Analyse Comparative"])
    
    # Initialisation des résultats
    results_rmse = {}
    results_precision = {}
    
    # Étape 6-7 : Calculs pour chaque k
    progress_bar = st.progress(0)
    for i, k in enumerate(k_values):
        with st.spinner(f'Calcul pour k={k}...'):
            # Prédiction SVD
            pred_svd = predict_svd(U, sigma, Vt, user_means, k)
            
            # Prédiction User-based
            pred_user = predict_user_similarity(matrix, user_means, k)
            
            # Prédiction Item-based
            pred_item = predict_item_similarity(matrix, k)
            
            # Évaluation RMSE
            rmse_svd, y_true_svd, y_pred_svd, errors_svd = evaluate_rmse(
                test_data, pred_svd, user_to_idx, movie_to_idx, train_data)
            
            rmse_user, y_true_user, y_pred_user, errors_user = evaluate_rmse(
                test_data, pred_user, user_to_idx, movie_to_idx, train_data)
            
            rmse_item, y_true_item, y_pred_item, errors_item = evaluate_rmse(
                test_data, pred_item, user_to_idx, movie_to_idx, train_data)
            
            results_rmse[k] = {
                'svd': (rmse_svd, errors_svd[:5]),
                'user': (rmse_user, errors_user[:5]),
                'item': (rmse_item, errors_item[:5]),
            }
            
            # Précision@10
            prec_svd, rec_svd = precision_at_k(test_data, pred_svd, user_to_idx, movie_to_idx, train_data)
            prec_user, rec_user = precision_at_k(test_data, pred_user, user_to_idx, movie_to_idx, train_data)
            prec_item, rec_item = precision_at_k(test_data, pred_item, user_to_idx, movie_to_idx, train_data)
            
            results_precision[k] = {
                'svd': (np.mean(list(prec_svd.values())) if prec_svd else 0, 
                       dict(list(prec_svd.items())[:10])),
                'user': (np.mean(list(prec_user.values())) if prec_user else 0,
                        dict(list(prec_user.items())[:10])),
                'item': (np.mean(list(prec_item.values())) if prec_item else 0,
                        dict(list(prec_item.items())[:10])),
            }
            
            progress_bar.progress((i + 1) / len(k_values))
    
    # Onglet 1 : RMSE et Prédictions
    with tabs[0]:
        st.subheader("📉 RMSE pour différentes valeurs de k")
        
        # Tableau comparatif RMSE
        rmse_df = pd.DataFrame({
            'k': k_values,
            'RMSE SVD': [results_rmse[k]['svd'][0] for k in k_values],
            'RMSE User-Based': [results_rmse[k]['user'][0] for k in k_values],
            'RMSE Item-Based': [results_rmse[k]['item'][0] for k in k_values],
        })
        
        st.dataframe(rmse_df.style.highlight_min(axis=0), use_container_width=True)
        
        # Graphique RMSE
        fig_rmse, ax = plt.subplots(figsize=(10, 6))
        rmse_df.set_index('k').plot(marker='o', ax=ax)
        ax.set_ylabel("RMSE")
        ax.set_title("Comparaison des RMSE selon k")
        st.pyplot(fig_rmse)
        
        # Meilleures prédictions
        st.subheader("✅ 5 Meilleures prédictions (erreur minimale)")
        
        k_selected = st.selectbox("Sélectionner k pour visualiser les prédictions", k_values)
        
        method = st.radio("Méthode", ["SVD", "User-Based", "Item-Based"], horizontal=True)
        method_key = method.lower().replace('-', '_')
        
        _, errors = results_rmse[k_selected][method_key]
        
        st.write(f"Prédictions avec {method} pour k={k_selected}")
        pred_df = pd.DataFrame(errors, columns=['Erreur', 'Note Réelle', 'Note Prédite'])
        st.dataframe(pred_df.round(3), use_container_width=True)
    
    # Onglet 2 : Précision@10
    with tabs[1]:
        st.subheader("🎯 Précision@10")
        
        # Tableau comparatif
        precision_df = pd.DataFrame({
            'k': k_values,
            'Precision@10 SVD': [results_precision[k]['svd'][0] for k in k_values],
            'Precision@10 User-Based': [results_precision[k]['user'][0] for k in k_values],
            'Precision@10 Item-Based': [results_precision[k]['item'][0] for k in k_values],
        })
        
        st.dataframe(precision_df.style.highlight_max(axis=0), use_container_width=True)
        
        # Top users avec meilleure précision
        st.subheader("🏆 Top 10 Users avec meilleure Precision@10")
        
        k_prec = st.selectbox("Sélectionner k pour Precision@10", k_values, key='prec_k')
        method_prec = st.radio("Méthode pour recommandations", 
                               ["SVD", "User-Based", "Item-Based"], 
                               horizontal=True, key='prec_method')
        method_key_prec = method_prec.lower().replace('-', '_')
        
        avg_prec, top_users = results_precision[k_prec][method_key_prec]
        
        st.metric("Précision moyenne@10", f"{avg_prec:.4f}")
        
        # Afficher les recommandations top users
        for user_id, precision in list(top_users.items())[:10]:
            st.write(f"**User {user_id}** - Precision@10: **{precision:.4f}**")
    
    # Onglet 3 : Analyse comparative
    with tabs[2]:
        st.subheader("📊 Analyse Comparative des Méthodes")
        
        # Heatmap des RMSE
        st.write("### Heatmap RMSE")
        fig_heatmap, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(rmse_df.set_index('k').T, annot=True, cmap='YlOrRd_r', ax=ax)
        st.pyplot(fig_heatmap)
        
        # Conclusions
        st.write("### 📋 Conclusions et Recommandations")
        
        best_svd_k = precision_df.loc[precision_df['Precision@10 SVD'].idxmax(), 'k']
        best_user_k = precision_df.loc[precision_df['Precision@10 User-Based'].idxmax(), 'k']
        best_item_k = precision_df.loc[precision_df['Precision@10 Item-Based'].idxmax(), 'k']
        
        st.success(f"""
        **Résumé des performances :**
        
        - **Meilleure configuration SVD** : k={best_svd_k}
        - **Meilleure configuration User-Based** : k={best_user_k}
        - **Meilleure configuration Item-Based** : k={best_item_k}
        
        - **RMSE le plus bas** : {rmse_df.min().min():.4f}
        - **Precision@10 la plus haute** : {precision_df.max().max():.4f}
        """)
        
        # Téléchargement des résultats
        st.download_button(
            "📥 Télécharger les résultats (CSV)",
            rmse_df.to_csv(index=False).encode('utf-8'),
            "resultats_svd.csv",
            "text/csv"
        )

if __name__ == "__main__":
    main()
