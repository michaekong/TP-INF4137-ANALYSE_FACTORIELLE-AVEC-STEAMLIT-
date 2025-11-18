import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="Mon Analyse Interactive",
    page_icon="📊",
    layout="wide"
)

# Titre principal
st.title("📈 Dashboard Interactif")
st.markdown("Ceci est une application Streamlit de démonstration.")

# Sidebar avec des contrôles
st.sidebar.header("Paramètres")
nb_points = st.sidebar.slider("Nombre de points", 10, 1000, 100)
bruit = st.sidebar.slider("Niveau de bruit", 0.0, 2.0, 1.0)

# Génération de données
data = pd.DataFrame({
    'x': np.arange(nb_points),
    'y': np.sin(np.arange(nb_points) * 0.1) * bruit + np.random.randn(nb_points) * 0.2
})

# Métriques
col1, col2, col3 = st.columns(3)
col1.metric("Moyenne", f"{data['y'].mean():.2f}")
col2.metric("Écart-type", f"{data['y'].std():.2f}")
col3.metric("Max", f"{data['y'].max():.2f}")

# Graphique
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['x'], data['y'], label='Données')
ax.set_xlabel("Index")
ax.set_ylabel("Valeur")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Tableau interactif
st.subheader("Aperçu des données")
st.dataframe(data.head(10), use_container_width=True)

# Bouton de téléchargement
csv = data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Télécharger les données (CSV)",
    data=csv,
    file_name="donnees.csv",
    mime="text/csv",
)