import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import io
import time
from typing import Tuple, Dict, List, Any, Optional

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="SVD - Systèmes de Recommandation",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Couleurs améliorées
COLORS = {
    "primary": "#6D28D9",  # Violet foncé
    "secondary": "#7C3AED",  # Violet
    "accent": "#A78BFA",  # Violet clair
    "light": "#EDE9FE",  # Très clair
    "dark": "#4C1D95",  # Très foncé
    "success": "#059669",  # Vert
    "warning": "#D97706",  # Orange
    "danger": "#DC2626",  # Rouge
    "text": "#111827",  # Noir
    "bg": "#F9FAFB",  # Gris clair
}

THRESHOLD_INFO = 0.6
MIN_NORM = 1e-10

# CSS amélioré
st.markdown(
    f"""
<style>
    body {{ background-color: {COLORS['bg']}; }}
    .main-header {{ 
        font-size: 3em; font-weight: 900; 
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        padding: 20px 0;
    }}
    .section-header {{ 
        font-size: 1.8em; color: {COLORS['primary']}; 
        font-weight: 800; 
        border-bottom: 4px solid {COLORS['secondary']}; 
        padding: 15px 0;
        margin: 30px 0 20px 0;
    }}
    .subsection-header {{ 
        font-size: 1.3em; color: {COLORS['secondary']}; 
        font-weight: 700; 
        margin: 20px 0 15px 0;
        padding: 10px 0;
    }}
    .info-card {{ 
        background: linear-gradient(135deg, #FFFFFF 0%, {COLORS['light']} 100%);
        border: 2px solid {COLORS['secondary']}; 
        border-radius: 12px; 
        padding: 20px; 
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(109, 40, 217, 0.1);
    }}
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(109, 40, 217, 0.2);
    }}
    .step-card {{ 
        background: {COLORS['light']};
        border-left: 5px solid {COLORS['primary']}; 
        border-radius: 8px; 
        padding: 15px; 
        margin: 10px 0;
    }}
    .success-badge {{ 
        background: {COLORS['success']}; 
        color: white; 
        padding: 8px 16px; 
        border-radius: 20px; 
        display: inline-block; 
        margin: 5px;
        font-weight: 600;
    }}
    .warning-badge {{ 
        background: {COLORS['warning']}; 
        color: white; 
        padding: 8px 16px; 
        border-radius: 20px; 
        display: inline-block; 
        margin: 5px;
        font-weight: 600;
    }}
    .danger-badge {{
        background: {COLORS['danger']};
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px;
        font-weight: 600;
    }}
    .stProgress > div > div > div {{ 
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%) !important;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# ==================== FONCTIONS UTILITAIRES ====================


def metric_card(col, label: str, value: str, icon: str = "📊"):
    """Afficher une métrique dans une carte"""
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 2.5em; margin-bottom: 10px;">{icon}</div>
                <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">{label}</div>
                <div style="font-size: 2em; font-weight: bold;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def info_box(title: str, content: str):
    """Afficher une boîte d'information"""
    st.markdown(
        f"""
        <div class="info-card">
            <b style="color: {COLORS['primary']}; font-size: 1.1em;">{title}</b><br>
            <div style="margin-top: 10px; color: {COLORS['text']};">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, type: str = "success"):
    """Afficher un badge"""
    badge_class = f"{type}-badge"
    st.markdown(f'<span class="{badge_class}">{text}</span>', unsafe_allow_html=True)


# ==================== TP1 - SVD DÉTAILLÉ ====================


def gram_schmidt_detailed(
    vectors: np.ndarray, name: str = "Vecteurs"
) -> Tuple[np.ndarray, List[Dict]]:
    """Gram-Schmidt avec affichage détaillé"""
    st.markdown(
        f'<div class="subsection-header">🔧 Orthonormalisation Gram-Schmidt - {name}</div>',
        unsafe_allow_html=True,
    )

    ortho = []
    details = []
    progress = st.progress(0)
    status = st.empty()

    for i, v in enumerate(vectors.T):
        progress.progress((i + 1) / len(vectors.T))
        status.text(f"Vecteur {i+1}/{len(vectors.T)}")

        with st.expander(f"📊 Étape {i+1}: Traitement v_{i+1}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Vecteur original:**")
                st.latex(f"v_{i+1} = {np.array2string(v, separator=', ')}")

            w = v.astype(float)

            for j, u in enumerate(ortho):
                proj_scalar = np.dot(u, v)
                w -= proj_scalar * u
                st.write(f"*Projection sur u_{j+1}: {proj_scalar:.4f}*")

            norm_val = np.linalg.norm(w)

            with col2:
                st.write("**Norme:**")
                st.latex(f"||w_{i+1}|| = {norm_val:.4f}")

            if norm_val > MIN_NORM:
                u = w / norm_val
                ortho.append(u)
                details.append({"vector": i, "norm": norm_val, "orthonormal": u.copy()})
                st.success(f"✅ Vecteur normalisé: ||u_{i+1}|| = 1.0000")
            else:
                st.warning(f"⚠️ Vecteur négligé (norme: {norm_val:.2e})")

    status.empty()
    progress.empty()
    return np.array(ortho).T if ortho else np.zeros((vectors.shape[0], 0)), details


def svd_from_scratch_complete(A: np.ndarray) -> Dict[str, Any]:
    """SVD from scratch complet avec tous les détails"""
    A = np.array(A, dtype=float)
    m, n = A.shape

    # Phase 1: Matrices AAT et ATA
    st.markdown(
        '<div class="section-header">🧮 Phase 1: Construction des matrices de covariance</div>',
        unsafe_allow_html=True,
    )

    progress = st.progress(0)

    AT = A.T
    AAT = A @ AT
    ATA = AT @ A

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**A·A^T (pour U):**")
        st.write(f"Dimensions: {AAT.shape[0]}×{AAT.shape[1]}")
        with st.expander("Voir matrice"):
            st.dataframe(
                pd.DataFrame(AAT, columns=[f"Col{i+1}" for i in range(AAT.shape[1])])
            )

    with col2:
        st.markdown("**A^T·A (pour V):**")
        st.write(f"Dimensions: {ATA.shape[0]}×{ATA.shape[1]}")
        with st.expander("Voir matrice"):
            st.dataframe(
                pd.DataFrame(ATA, columns=[f"Col{i+1}" for i in range(ATA.shape[1])])
            )

    progress.progress(20)

    # Phase 2: Vecteurs propres
    st.markdown(
        '<div class="section-header">🔢 Phase 2: Calcul des vecteurs propres</div>',
        unsafe_allow_html=True,
    )

    eigenvalues_U, eigenvectors_U = np.linalg.eig(AAT)
    idx_U = np.argsort(eigenvalues_U)[::-1]
    eigenvalues_U = eigenvalues_U[idx_U]
    eigenvectors_U = eigenvectors_U[:, idx_U]

    eigenvalues_V, eigenvectors_V = np.linalg.eig(ATA)
    idx_V = np.argsort(eigenvalues_V)[::-1]
    eigenvalues_V = eigenvalues_V[idx_V]
    eigenvectors_V = eigenvectors_V[:, idx_V]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Valeurs propres de A·A^T:**")
        for i in range(min(5, len(eigenvalues_U))):
            badge(f"λ_{i+1} = {eigenvalues_U[i]:.4f}", "success")

    with col2:
        st.markdown("**Valeurs propres de A^T·A:**")
        for i in range(min(5, len(eigenvalues_V))):
            badge(f"λ_{i+1} = {eigenvalues_V[i]:.4f}", "success")

    progress.progress(40)

    # Phase 3: Orthonormalisation
    st.markdown(
        '<div class="section-header">⚙️ Phase 3: Orthonormalisation Gram-Schmidt</div>',
        unsafe_allow_html=True,
    )

    U, details_U = gram_schmidt_detailed(eigenvectors_U, "U")
    V, details_V = gram_schmidt_detailed(eigenvectors_V, "V")

    progress.progress(60)

    # Phase 4: Valeurs singulières
    st.markdown(
        '<div class="section-header">📏 Phase 4: Valeurs singulières</div>',
        unsafe_allow_html=True,
    )

    sigma = np.sqrt(np.abs(eigenvalues_V))
    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, sigma[: min(m, n)])

    st.markdown("**Calcul:** σᵢ = √λᵢ(A^T·A)")

    sigma_df = pd.DataFrame(
        {
            "Index": [f"σ{i+1}" for i in range(len(sigma))],
            "Valeur": sigma,
            "Valeur²": sigma**2,
            "% Information": (sigma**2 / np.sum(sigma**2) * 100),
        }
    )

    st.dataframe(
        sigma_df.style.format(
            {"Valeur": "{:.4f}", "Valeur²": "{:.4f}", "% Information": "{:.2f}%"}
        ),
        use_container_width=True,
    )

    progress.progress(80)

    # Phase 5: Reconstruction
    st.markdown(
        '<div class="section-header">🔄 Phase 5: Matrices finales</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Matrice U:**")
        st.write(f"Dimensions: {U.shape}")
        with st.expander("Voir"):
            st.dataframe(pd.DataFrame(U[:, : min(5, U.shape[1])]))

    with col2:
        st.markdown("**Matrice Σ:**")
        st.write(f"Dimensions: {Sigma.shape}")
        with st.expander("Voir"):
            st.dataframe(pd.DataFrame(Sigma))

    with col3:
        st.markdown("**Matrice V^T:**")
        st.write(f"Dimensions: {V.T.shape}")
        with st.expander("Voir"):
            st.dataframe(pd.DataFrame(V.T[: min(5, V.T.shape[0])]))

    progress.progress(100)
    time.sleep(0.3)
    progress.empty()

    return {
        "U": U,
        "Sigma": Sigma,
        "Vt": V.T,
        "sigma": sigma,
        "eigenvalues_U": eigenvalues_U,
        "eigenvalues_V": eigenvalues_V,
        "AAT": AAT,
        "ATA": ATA,
    }


def tp1_svd():
    """Interface TP1 complète"""
    st.markdown(
        '<div class="main-header">🔮 TP1 - Décomposition SVD Complète</div>',
        unsafe_allow_html=True,
    )

    # Configuration
    st.markdown(
        '<div class="section-header">📝 Configuration</div>', unsafe_allow_html=True
    )

    col_method, col_k = st.columns([1, 1])

    with col_method:
        method = st.radio("Méthode", ["🔧 From Scratch", "📚 Numpy"])

    with col_k:
        k_force = st.slider("Forcer k (0=auto)", 0, 15, 0)

    st.markdown("**Entrez votre matrice (format: [v1, v2, ...]):**")
    matrix_input = st.text_area(
        "", value="[3, 1, 1]\n[-1, 3, 1]", height=120, label_visibility="collapsed"
    )

    if st.button("🚀 Analyser", type="primary", use_container_width=True):
        # Parsing
        st.markdown(
            '<div class="section-header">✅ Phase 0: Chargement</div>',
            unsafe_allow_html=True,
        )

        try:
            rows = []
            for line in matrix_input.strip().split("\n"):
                if line.strip():
                    row = eval(line.strip())
                    rows.append(list(map(float, row)))
            A = np.array(rows, dtype=float)

            if A.size == 0:
                st.error("❌ Matrice vide")
                return

            st.success(f"✅ Matrice chargée: **{A.shape[0]}×{A.shape[1]}**")

            st.markdown("**Matrice d'entrée:**")
            st.dataframe(pd.DataFrame(A), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Erreur: {e}")
            return

        # Calcul SVD
        if method == "🔧 From Scratch":
            result = svd_from_scratch_complete(A)
        else:
            st.markdown(
                '<div class="section-header">🧮 Décomposition Numpy</div>',
                unsafe_allow_html=True,
            )
            progress = st.progress(0)

            U, sigma_np, Vt = np.linalg.svd(A, full_matrices=False)
            Sigma = np.zeros((A.shape[0], A.shape[1]))
            np.fill_diagonal(Sigma, sigma_np)

            progress.progress(100)
            time.sleep(0.3)
            progress.empty()

            result = {
                "U": U,
                "Sigma": Sigma,
                "Vt": Vt,
                "sigma": sigma_np,
                "eigenvalues_V": sigma_np**2,
            }

        # Analyse des résultats
        st.markdown(
            '<div class="section-header">📊 Analyse des résultats</div>',
            unsafe_allow_html=True,
        )

        sigma = result["sigma"]
        total_info = np.sum(sigma**2)
        cumulative = np.cumsum(sigma**2) / total_info

        if k_force == 0:
            k = np.argmax(cumulative >= THRESHOLD_INFO) + 1
        else:
            k = min(k_force, len(sigma))

        info_rate = cumulative[k - 1] if k > 0 else 0

        # Métriques clés
        col1, col2, col3, col4 = st.columns(4)
        metric_card(col1, "k optimal", str(k), "🎯")
        metric_card(col2, "Info conservée", f"{info_rate*100:.1f}%", "💾")
        metric_card(col3, "Rang", str(min(A.shape)), "📐")
        metric_card(col4, "Réduction", f"{(1-k/len(sigma))*100:.1f}%", "📉")

        # Visualisations
        st.markdown(
            '<div class="section-header">📈 Visualisations</div>', unsafe_allow_html=True
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("white")

        # Valeurs singulières
        axes[0, 0].bar(
            range(1, len(sigma) + 1),
            sigma,
            color=COLORS["primary"],
            alpha=0.8,
            edgecolor=COLORS["dark"],
        )
        axes[0, 0].axvline(
            k, color=COLORS["warning"], linestyle="--", linewidth=2, label=f"k={k}"
        )
        axes[0, 0].set_title("Valeurs singulières σᵢ", fontweight="bold", fontsize=12)
        axes[0, 0].set_xlabel("Composante")
        axes[0, 0].set_ylabel("Valeur")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Info cumulée
        axes[0, 1].plot(
            range(1, len(cumulative) + 1),
            cumulative * 100,
            "o-",
            color=COLORS["secondary"],
            linewidth=2.5,
            markersize=8,
        )
        axes[0, 1].axhline(
            THRESHOLD_INFO * 100, color=COLORS["warning"], linestyle="--", linewidth=2
        )
        axes[0, 1].axvline(k, color=COLORS["warning"], linestyle="--", linewidth=2)
        axes[0, 1].fill_between(
            range(1, len(cumulative) + 1),
            cumulative * 100,
            alpha=0.2,
            color=COLORS["secondary"],
        )
        axes[0, 1].set_title("Information cumulée", fontweight="bold", fontsize=12)
        axes[0, 1].set_xlabel("k")
        axes[0, 1].set_ylabel("Information (%)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 105])

        # Reconstruction et erreur
        A_k = result["U"][:, :k] @ result["Sigma"][:k, :k] @ result["Vt"][:k, :]
        error = np.abs(A - A_k)

        im = axes[1, 0].imshow(error, cmap="RdYlGn_r", interpolation="nearest")
        axes[1, 0].set_title(f"Erreur |A - Aₖ| (k={k})", fontweight="bold", fontsize=12)
        plt.colorbar(im, ax=axes[1, 0], label="Erreur")

        # Heatmap reconstruction qualité
        comparison = []
        for i in range(min(5, A.shape[0])):
            row = []
            for j in range(min(5, A.shape[1])):
                diff = abs(A[i, j] - A_k[i, j])
                if diff < 0.01:
                    row.append(3)
                elif diff < 0.1:
                    row.append(2)
                else:
                    row.append(1)
            comparison.append(row)

        im2 = axes[1, 1].imshow(comparison, cmap="RdYlGn", vmin=1, vmax=3)
        axes[1, 1].set_title(
            "Qualité reconstruction\n(🟢 Excellent | 🟡 Bon | 🔴 Moyen)",
            fontweight="bold",
            fontsize=12,
        )
        axes[1, 1].set_xticks(range(min(5, A.shape[1])))
        axes[1, 1].set_yticks(range(min(5, A.shape[0])))
        axes[1, 1].set_xticklabels([f"Col{i+1}" for i in range(min(5, A.shape[1]))])
        axes[1, 1].set_yticklabels([f"Row{i+1}" for i in range(min(5, A.shape[0]))])

        plt.tight_layout()
        st.pyplot(fig)

        # Résumé des métriques
        st.markdown(
            '<div class="section-header">📊 Résumé des métriques</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        fro_error = np.linalg.norm(A - A_k, "fro")
        fro_original = np.linalg.norm(A, "fro")
        max_error = np.max(error)
        mean_error = np.mean(error)

        with col1:
            info_box(
                "📐 Normes Frobenius",
                f"||A||_F = {fro_original:.4f}<br>||A-Aₖ||_F = {fro_error:.4f}<br>Erreur relative = {fro_error/fro_original:.2%}",
            )

        with col2:
            info_box(
                "📊 Erreurs",
                f"Max erreur = {max_error:.4f}<br>Erreur moyenne = {mean_error:.4f}<br>Min erreur = {np.min(error):.4f}",
            )

        with col3:
            info_box(
                "💾 Compression",
                f"Facteurs conservés = {k}/{len(sigma)}<br>Info conservée = {info_rate:.2%}<br>Réduction données = {(1-k/len(sigma))*100:.1f}%",
            )


# ==================== TP2 - MOVIELENS ====================


def tp2_movielens():
    """Interface TP2"""
    st.markdown(
        '<div class="main-header">🎬 TP2 - Système de Recommandation MovieLens</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-header">📥 Phase 1: Chargement des données</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        source = st.radio("Source", ["🌐 Télécharger", "📤 Upload"], horizontal=True)

    with col2:
        method = st.radio("Méthode SVD", ["🔧 From Scratch", "📚 Numpy"], horizontal=True)

    data = None

    if source == "🌐 Télécharger":
        if st.button("Télécharger MovieLens 100k"):
            with st.spinner("⏳ Téléchargement..."):
                try:
                    url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
                    data = pd.read_csv(
                        url, sep="\t", names=["userId", "movieId", "rating", "timestamp"]
                    )
                    st.success(f"✅ {len(data):,} interactions chargées")
                except:
                    st.error("❌ Erreur de téléchargement")
    else:
        file = st.file_uploader("Upload u.data", type=["data", "csv", "txt"])
        if file:
            data = pd.read_csv(
                file, sep="\t", names=["userId", "movieId", "rating", "timestamp"]
            )
            st.success("✅ Fichier chargé")

    if data is None:
        st.warning("⚠️ Veuillez charger les données")
        return

    # Préparation
    st.markdown(
        '<div class="section-header">🔧 Phase 2: Préparation des données</div>',
        unsafe_allow_html=True,
    )

    n_samples = st.slider(
        "Nombre d'échantillons", 1000, min(50000, len(data)), 30000, 1000
    )
    data = data.head(n_samples).sort_values("timestamp")

    col1, col2, col3 = st.columns(3)
    metric_card(col1, "Total", f"{len(data):,}", "📊")
    metric_card(col2, "Utilisateurs", str(data["userId"].nunique()), "👥")
    metric_card(col3, "Films", str(data["movieId"].nunique()), "🎬")

    split = int(0.7 * len(data))
    train_data = data.iloc[:split]
    test_data = data.iloc[split:]

    col1, col2 = st.columns(2)
    metric_card(col1, "Train (70%)", f"{len(train_data):,}", "✅")
    metric_card(col2, "Test (30%)", f"{len(test_data):,}", "🧪")

    # Matrice
    st.markdown(
        '<div class="section-header">🧮 Phase 3: Construction matrice</div>',
        unsafe_allow_html=True,
    )

    train_matrix = (
        train_data.pivot(index="userId", columns="movieId", values="rating")
        .fillna(0)
        .values
    )

    density = (train_matrix > 0).sum() / train_matrix.size * 100

    col1, col2, col3 = st.columns(3)
    metric_card(
        col1, "Dimensions", f"{train_matrix.shape[0]}×{train_matrix.shape[1]}", "📐"
    )
    metric_card(col2, "Densité", f"{density:.2f}%", "📊")
    metric_card(col3, "Valeurs", f"{(train_matrix > 0).sum():,}", "💾")

    info_box(
        "📊 Matrice User-Movie",
        f"Utilisateurs: {train_matrix.shape[0]} | Films: {train_matrix.shape[1]}<br>Sparsité: {100-density:.2f}% (données manquantes)",
    )

    # Configuration
    st.markdown(
        '<div class="section-header">⚙️ Phase 4: Configuration</div>',
        unsafe_allow_html=True,
    )

    k_values = st.multiselect(
        "Valeurs de k à tester", [3, 5, 10, 20, 30, 50], default=[3, 5, 10]
    )

    if not k_values:
        st.warning("⚠️ Sélectionnez au moins une valeur de k")
        return

    if st.button(
        "🚀 Lancer l'analyse complète", type="primary", use_container_width=True
    ):
        # SVD
        st.markdown(
            '<div class="section-header">⚡ Phase 5: Décomposition SVD</div>',
            unsafe_allow_html=True,
        )

        progress = st.progress(0)

        U, sigma, Vt = np.linalg.svd(train_matrix, full_matrices=False)

        progress.progress(50)

        # Visualisation SVD
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("white")

        axes[0].plot(
            range(1, min(21, len(sigma) + 1)),
            sigma[:20],
            "o-",
            color=COLORS["primary"],
            linewidth=2.5,
            markersize=8,
        )
        axes[0].set_title("Valeurs singulières", fontweight="bold", fontsize=12)
        axes[0].set_xlabel("Composante")
        axes[0].set_ylabel("σᵢ")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_facecolor("#F9FAFB")

        cum_info = np.cumsum(sigma**2) / np.sum(sigma**2)
        axes[1].plot(
            range(1, len(cum_info) + 1),
            cum_info * 100,
            "s-",
            color=COLORS["secondary"],
            linewidth=2.5,
            markersize=8,
        )
        axes[1].axhline(
            60, color=COLORS["warning"], linestyle="--", linewidth=2, label="60% seuil"
        )
        axes[1].fill_between(
            range(1, len(cum_info) + 1),
            cum_info * 100,
            alpha=0.2,
            color=COLORS["secondary"],
        )
        axes[1].set_title("Information cumulée", fontweight="bold", fontsize=12)
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Information (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_facecolor("#F9FAFB")
        axes[1].set_ylim([0, 105])

        plt.tight_layout()
        st.pyplot(fig)

        info_box(
            "✅ SVD Terminée",
            f"Rang matriciel: {np.sum(sigma > 1e-10)}<br>Valeurs singulières calculées: {len(sigma)}<br>Dimensions U: {U.shape} | V^T: {Vt.shape}",
        )

        progress.progress(75)

        # Prédictions
        st.markdown(
            '<div class="section-header">🔮 Phase 6: Calcul des prédictions</div>',
            unsafe_allow_html=True,
        )

        results_by_k = {}
        pred_progress = st.progress(0)
        pred_status = st.empty()

        # Limiter le test à 500 pour rapidité
        test_sample = test_data.sample(min(500, len(test_data)))

        for k_idx, k in enumerate(k_values):
            if k > len(sigma):
                st.warning(f"⚠️ k={k} dépasse le rang")
                continue

            pred_status.markdown(f"**Traitement k={k}... ({k_idx+1}/{len(k_values)})**")

            pred_ak = []
            pred_user_sim = []
            pred_item_sim = []

            for i, (_, row) in enumerate(test_sample.iterrows()):
                user_id = int(row["userId"]) - 1
                movie_id = int(row["movieId"]) - 1
                actual = row["rating"]

                if 0 <= user_id < U.shape[0] and 0 <= movie_id < Vt.shape[1]:
                    # Méthode 1: Ak
                    sigma_k = np.diag(sigma[:k])
                    pred_val = U[user_id, :k] @ sigma_k @ Vt[:k, movie_id]
                    pred_ak.append((actual, np.clip(pred_val, 1, 5)))

                    # Méthode 2: Similarité utilisateurs
                    user_vec = U[user_id, :k]
                    sims = []
                    ratings = []
                    for u in range(train_matrix.shape[0]):
                        if train_matrix[u, movie_id] > 0:
                            other_vec = U[u, :k]
                            sim = np.dot(user_vec, other_vec) / (
                                np.linalg.norm(user_vec) * np.linalg.norm(other_vec)
                                + MIN_NORM
                            )
                            sims.append(sim)
                            ratings.append(train_matrix[u, movie_id])

                    if sims:
                        sims = np.array(sims)
                        ratings = np.array(ratings)
                        pred = (
                            np.sum(sims * ratings) / np.sum(np.abs(sims))
                            if np.sum(np.abs(sims)) > 0
                            else 3.0
                        )
                        pred_user_sim.append((actual, np.clip(pred, 1, 5)))
                    else:
                        pred_user_sim.append((actual, 3.0))

                    # Méthode 3: Similarité items
                    item_vec = Vt[:k, movie_id]
                    user_ratings = train_matrix[user_id, :]
                    rated_movies = np.where(user_ratings > 0)[0]

                    sims = []
                    ratings = []
                    for m in rated_movies:
                        other_vec = Vt[:k, m]
                        sim = np.dot(item_vec, other_vec) / (
                            np.linalg.norm(item_vec) * np.linalg.norm(other_vec)
                            + MIN_NORM
                        )
                        sims.append(sim)
                        ratings.append(user_ratings[m])

                    if sims:
                        sims = np.array(sims)
                        ratings = np.array(ratings)
                        pred = (
                            np.sum(sims * ratings) / np.sum(np.abs(sims))
                            if np.sum(np.abs(sims)) > 0
                            else 3.0
                        )
                        pred_item_sim.append((actual, np.clip(pred, 1, 5)))
                    else:
                        pred_item_sim.append((actual, 3.0))

            # Calcul RMSE
            rmse_ak = (
                np.sqrt(
                    mean_squared_error([x[0] for x in pred_ak], [x[1] for x in pred_ak])
                )
                if pred_ak
                else 0
            )
            rmse_user = (
                np.sqrt(
                    mean_squared_error(
                        [x[0] for x in pred_user_sim], [x[1] for x in pred_user_sim]
                    )
                )
                if pred_user_sim
                else 0
            )
            rmse_item = (
                np.sqrt(
                    mean_squared_error(
                        [x[0] for x in pred_item_sim], [x[1] for x in pred_item_sim]
                    )
                )
                if pred_item_sim
                else 0
            )

            info_conserved = np.sum(sigma[:k] ** 2) / np.sum(sigma**2) * 100

            results_by_k[k] = {
                "rmse_ak": rmse_ak,
                "rmse_user": rmse_user,
                "rmse_item": rmse_item,
                "pred_ak": pred_ak,
                "pred_user": pred_user_sim,
                "pred_item": pred_item_sim,
                "info": info_conserved,
            }

            pred_progress.progress((k_idx + 1) / len(k_values))

        pred_status.empty()
        pred_progress.empty()

        progress.progress(100)
        time.sleep(0.3)
        progress.empty()

        # Résultats
        st.markdown(
            '<div class="section-header">📊 Phase 7: Résultats et comparaison</div>',
            unsafe_allow_html=True,
        )

        if not results_by_k:
            st.error("❌ Pas de résultats")
            return

        # Tableau RMSE
        rmse_data = []
        for k in sorted(results_by_k.keys()):
            rmse_data.append(
                {
                    "k": k,
                    "Info (%)": f"{results_by_k[k]['info']:.1f}%",
                    "RMSE Ak": f"{results_by_k[k]['rmse_ak']:.4f}",
                    "RMSE User": f"{results_by_k[k]['rmse_user']:.4f}",
                    "RMSE Item": f"{results_by_k[k]['rmse_item']:.4f}",
                }
            )

        rmse_df = pd.DataFrame(rmse_data)

        st.markdown("**📋 Tableau récapitulatif:**")
        st.dataframe(rmse_df, use_container_width=True)

        # Graphiques de comparaison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("white")

        # RMSE vs k
        k_list = sorted(results_by_k.keys())
        rmse_ak_list = [results_by_k[k]["rmse_ak"] for k in k_list]
        rmse_user_list = [results_by_k[k]["rmse_user"] for k in k_list]
        rmse_item_list = [results_by_k[k]["rmse_item"] for k in k_list]

        x = np.arange(len(k_list))
        width = 0.25

        axes[0].bar(
            x - width,
            rmse_ak_list,
            width,
            label="Ak",
            color=COLORS["primary"],
            alpha=0.8,
            edgecolor=COLORS["dark"],
        )
        axes[0].bar(
            x,
            rmse_user_list,
            width,
            label="User Sim",
            color=COLORS["secondary"],
            alpha=0.8,
            edgecolor=COLORS["dark"],
        )
        axes[0].bar(
            x + width,
            rmse_item_list,
            width,
            label="Item Sim",
            color=COLORS["accent"],
            alpha=0.8,
            edgecolor=COLORS["dark"],
        )

        axes[0].set_xlabel("k", fontweight="bold")
        axes[0].set_ylabel("RMSE", fontweight="bold")
        axes[0].set_title("Comparaison RMSE par méthode", fontweight="bold", fontsize=12)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(k_list)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[0].set_facecolor("#F9FAFB")

        # Évolution RMSE
        axes[1].plot(
            k_list,
            rmse_ak_list,
            "o-",
            color=COLORS["primary"],
            linewidth=2.5,
            markersize=8,
            label="Ak",
        )
        axes[1].plot(
            k_list,
            rmse_user_list,
            "s-",
            color=COLORS["secondary"],
            linewidth=2.5,
            markersize=8,
            label="User Sim",
        )
        axes[1].plot(
            k_list,
            rmse_item_list,
            "^-",
            color=COLORS["accent"],
            linewidth=2.5,
            markersize=8,
            label="Item Sim",
        )

        axes[1].set_xlabel("k", fontweight="bold")
        axes[1].set_ylabel("RMSE", fontweight="bold")
        axes[1].set_title("Évolution du RMSE", fontweight="bold", fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_facecolor("#F9FAFB")

        plt.tight_layout()
        st.pyplot(fig)

        # Meilleures performances
        st.markdown(
            '<div class="subsection-header">🏆 Meilleures performances</div>',
            unsafe_allow_html=True,
        )

        best_ak_k = min(results_by_k.keys(), key=lambda k: results_by_k[k]["rmse_ak"])
        best_user_k = min(results_by_k.keys(), key=lambda k: results_by_k[k]["rmse_user"])
        best_item_k = min(results_by_k.keys(), key=lambda k: results_by_k[k]["rmse_item"])

        col1, col2, col3 = st.columns(3)

        with col1:
            info_box(
                "🎯 Matrice Reconstruite (Ak)",
                f"<b>RMSE moyen:</b> {np.mean([results_by_k[k]['rmse_ak'] for k in results_by_k]):.4f}<br>"
                f"<b>Meilleur k:</b> {best_ak_k}<br>"
                f"<b>Avantages:</b> Rapide, direct<br>"
                f"<b>Limitations:</b> Peut sur-apprendre",
            )

        with col2:
            info_box(
                "👥 Similarité Utilisateurs",
                f"<b>RMSE moyen:</b> {np.mean([results_by_k[k]['rmse_user'] for k in results_by_k]):.4f}<br>"
                f"<b>Meilleur k:</b> {best_user_k}<br>"
                f"<b>Avantages:</b> Robuste, filtrage collaboratif<br>"
                f"<b>Limitations:</b> Coût computationnel",
            )

        with col3:
            info_box(
                "🎬 Similarité Items",
                f"<b>RMSE moyen:</b> {np.mean([results_by_k[k]['rmse_item'] for k in results_by_k]):.4f}<br>"
                f"<b>Meilleur k:</b> {best_item_k}<br>"
                f"<b>Avantages:</b> Contenu explicite, stabilité<br>"
                f"<b>Limitations:</b> Cold-start items",
            )

        # Top prédictions
        st.markdown(
            '<div class="section-header">✅ Top 5 prédictions par k</div>',
            unsafe_allow_html=True,
        )

        for k in sorted(results_by_k.keys())[:2]:
            with st.expander(f"📈 Détails k={k} (Info: {results_by_k[k]['info']:.1f}%)"):
                col1, col2, col3 = st.columns(3)

                # Ak
                with col1:
                    st.markdown("**🎯 Matrice Ak**")
                    best = sorted(
                        results_by_k[k]["pred_ak"], key=lambda x: abs(x[0] - x[1])
                    )[:5]
                    df = pd.DataFrame(best, columns=["Réelle", "Prédite"])
                    df["Erreur"] = np.abs(df["Réelle"] - df["Prédite"])
                    st.dataframe(
                        df.style.format(
                            {"Réelle": "{:.2f}", "Prédite": "{:.2f}", "Erreur": "{:.4f}"}
                        ),
                        use_container_width=True,
                    )

                # User Similarity
                with col2:
                    st.markdown("**👥 Similarité Utilisateurs**")
                    best = sorted(
                        results_by_k[k]["pred_user"], key=lambda x: abs(x[0] - x[1])
                    )[:5]
                    df = pd.DataFrame(best, columns=["Réelle", "Prédite"])
                    df["Erreur"] = np.abs(df["Réelle"] - df["Prédite"])
                    st.dataframe(
                        df.style.format(
                            {"Réelle": "{:.2f}", "Prédite": "{:.2f}", "Erreur": "{:.4f}"}
                        ),
                        use_container_width=True,
                    )

                # Item Similarity
                with col3:
                    st.markdown("**🎬 Similarité Items**")
                    best = sorted(
                        results_by_k[k]["pred_item"], key=lambda x: abs(x[0] - x[1])
                    )[:5]
                    df = pd.DataFrame(best, columns=["Réelle", "Prédite"])
                    df["Erreur"] = np.abs(df["Réelle"] - df["Prédite"])
                    st.dataframe(
                        df.style.format(
                            {"Réelle": "{:.2f}", "Prédite": "{:.2f}", "Erreur": "{:.4f}"}
                        ),
                        use_container_width=True,
                    )

        # Recommandations finales
        st.markdown(
            '<div class="section-header">💡 Recommandations finales</div>',
            unsafe_allow_html=True,
        )

        best_overall_rmse = min(
            [
                min(
                    [
                        results_by_k[k]["rmse_ak"],
                        results_by_k[k]["rmse_user"],
                        results_by_k[k]["rmse_item"],
                    ]
                )
                for k in results_by_k
            ]
        )

        st.success(
            f"""
        ✅ **RMSE Global Optimal: {best_overall_rmse:.4f}**
        
        **Résumé de l'analyse:**
        - Meilleur k global: {sorted(results_by_k.keys())[0]}
        - Information conservée moyenne: {np.mean([results_by_k[k]['info'] for k in results_by_k]):.1f}%
        - Nombre d'essais: {len(results_by_k)} configurations
        - Échantillons testés: {len(test_sample)}/{ len(test_data)}
        
        **À retenir:**
        🎯 Pour la précision: Privilégier la similarité utilisateurs
        💾 Pour la compression: Utiliser k petit (3-5)
        ⚖️ Pour l'équilibre: Utiliser k=10 avec similarité items
        """
        )


# ==================== MAIN ====================
def main():
    """Fonction principale"""

    # Sidebar
    with st.sidebar:
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%); 
                        padding: 25px; border-radius: 15px; color: white; text-align: center; margin-bottom: 20px;'>
                <h2 style='margin: 0; font-size: 1.8em;'>🔮 SVD Dashboard</h2>
                <p style='margin: 10px 0 0 0; opacity: 0.9; font-size: 0.9em;'>v2.0 - Optimisé</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Accueil", "📐 TP1 - SVD", "🎬 TP2 - MovieLens"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        info_box(
            "ℹ️ À propos",
            "Dashboard pédagogique SVD<br>Décomposition en Valeurs Singulières<br>Système de Recommandation",
        )

    # Pages
    if page == "🏠 Accueil":
        st.markdown(
            '<div class="main-header">🔮 Bienvenue - Dashboard SVD</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['dark']} 100%); 
                        padding: 40px; border-radius: 20px; color: white; text-align: center; margin-bottom: 30px;'>
                <h1 style='font-size: 2.5em; margin: 0; margin-bottom: 15px;'>📊 Décomposition SVD</h1>
                <p style='font-size: 1.1em; margin: 0; opacity: 0.95;'>De la théorie mathématique à l'application pratique</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, white 0%, {COLORS['light']} 100%); 
                            border: 3px solid {COLORS['primary']}; border-radius: 15px; padding: 30px;
                            box-shadow: 0 8px 25px rgba(109, 40, 217, 0.15);'>
                    <div style='font-size: 3.5em; text-align: center; margin-bottom: 15px;'>📐</div>
                    <h3 style='color: {COLORS['primary']}; text-align: center; margin-top: 0;'>TP1 - SVD Manuel</h3>
                    <ul style='list-style: none; padding: 0; text-align: left;'>
                        <li>✅ Calcul from scratch complet</li>
                        <li>✅ Power Iteration détaillée</li>
                        <li>✅ Gram-Schmidt étape par étape</li>
                        <li>✅ Sélection auto k ≥ 60%</li>
                        <li>✅ Visualisations 4 graphiques</li>
                        <li>✅ Métriques détaillées</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, white 0%, {COLORS['light']} 100%); 
                            border: 3px solid {COLORS['secondary']}; border-radius: 15px; padding: 30px;
                            box-shadow: 0 8px 25px rgba(124, 58, 237, 0.15);'>
                    <div style='font-size: 3.5em; text-align: center; margin-bottom: 15px;'>🎬</div>
                    <h3 style='color: {COLORS['secondary']}; text-align: center; margin-top: 0;'>TP2 - Recommandation</h3>
                    <ul style='list-style: none; padding: 0; text-align: left;'>
                        <li>✅ Dataset MovieLens 100k</li>
                        <li>✅ 3 méthodes de prédiction</li>
                        <li>✅ Évaluation RMSE complète</li>
                        <li>✅ Comparaison visuelle k</li>
                        <li>✅ Top 5 prédictions</li>
                        <li>✅ Recommandations finales</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        tab1, tab2 = st.tabs(["📚 Guide d'utilisation", "🔧 Configuration système"])

        with tab1:
            st.markdown(
                """
            ### 📐 TP1 - Décomposition SVD
            1. **Saisir une matrice** au format `[v1, v2, ...]`
            2. **Choisir la méthode**: From Scratch (détaillé) ou Numpy (rapide)
            3. **Cliquer Analyser** - Voir toutes les étapes
            4. **Résultats**: 
               - k optimal pour ≥60% information
               - Matrices U, Σ, V
               - Qualité reconstruction
               - Métriques détaillées
            
            ### 🎬 TP2 - Recommandation MovieLens
            1. **Charger données** MovieLens (download ou upload)
            2. **Configurer** nombre d'échantillons et k à tester
            3. **Sélectionner méthode** SVD
            4. **Lancer analyse** - Attendez les résultats
            5. **Analyser**:
               - Comparaison RMSE 3 méthodes
               - Performance vs k
               - Top prédictions
               - Recommandations
            """
            )

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.info("✅ **TP1**: Matrices jusqu'à 10×10")
            with col2:
                st.info("✅ **TP2**: 30,000 interactions max")

    elif page == "📐 TP1 - SVD":
        tp1_svd()

    elif page == "🎬 TP2 - MovieLens":
        tp2_movielens()


if __name__ == "__main__":
    main()
