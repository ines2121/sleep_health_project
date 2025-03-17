import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import os

# Configuration de la page - DOIT ÊTRE EN PREMIER
st.set_page_config(
    page_title="Analyse de la Santé du Sommeil",
    page_icon="😴",
    layout="wide"
)

# Styles CSS simplifiés
st.markdown("""
    <style>
        .stDownloadButton button::after { content: 'Télécharger'; }
        .stDeployButton button::after { content: 'Déployer'; }
        button[title="View fullscreen"]::after { content: 'Plein écran'; }
    </style>
""", unsafe_allow_html=True)

# Configuration
API_URL = os.environ.get('API_URL', 'http://localhost:8002')
blood_pressure_mapping = {"Normale": "Normal", "Élevée": "High"}

@st.cache_data
def load_data():
    df = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")
    # Renommer les colonnes pour correspondre au format attendu
    column_mapping = {
        'BMI Category': 'BMI',
        'Physical Activity Level': 'Physical Activity Level',
        'Quality of Sleep': 'Quality of Sleep',
        'Sleep Duration': 'Sleep Duration',
        'Daily Steps': 'Daily Steps',
        'Heart Rate': 'Heart Rate',
        'Blood Pressure': 'Blood Pressure',
        'Stress Level': 'Stress Level'
    }
    df = df.rename(columns=column_mapping)
    
    # Obtenir la liste unique des professions pour le selectbox
    occupations = sorted(df['Occupation'].unique())
    
    return df, occupations

df, occupations = load_data()

st.title("😴 Analyse de la Santé du Sommeil")

# Sidebar
st.sidebar.title("ℹ️ Informations")
st.sidebar.write("""
Cette application analyse la qualité du sommeil et le mode de vie.

**Catégories de Sommeil :**
- 🟢 **Excellent** : Score > 8
- 🟡 **Moyen** : Score 5-8
- 🔴 **Mauvais** : Score < 5

**Développé par :** Ines HAMMOUCH
""")

# Onglets
tab1, tab2, tab3 = st.tabs(["🔍 Analyse", "📈 Suivi", "📊 Visualisations"])

with tab1:
    st.write("### 📊 Analyse du Sommeil et Mode de Vie")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Genre", ["Homme", "Femme"])
        occupation = st.selectbox("Profession", occupations)
        physical_activity = st.slider("Niveau d'Activité Physique (minutes/jour)", 0, 120, 30)
    
    with col2:
        sleep_duration = st.slider("Durée du Sommeil (heures)", 4.0, 10.0, 7.0, 0.1)
        stress_level = st.slider("Niveau de Stress (1-10)", 1, 10, 5)
        daily_steps = st.number_input("Pas Quotidiens", min_value=2000, max_value=25000, value=8000)
    
    with col3:
        bmi = st.number_input("IMC", min_value=15.0, max_value=40.0, value=23.0)
        blood_pressure = st.selectbox("Pression Artérielle", ["Normale", "Élevée"])
        heart_rate = st.number_input("Fréquence Cardiaque", min_value=60, max_value=120, value=75)

    if st.button("🔍 Analyser mon sommeil", type="primary"):
        data = {
            'Age': age,
            'Gender': 'Male' if gender == "Homme" else "Female",
            'Occupation': occupation,
            'Sleep Duration': sleep_duration,
            'Physical Activity Level': physical_activity,
            'Stress Level': stress_level,
            'BMI': bmi,
            'Blood Pressure': blood_pressure_mapping[blood_pressure],
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps
        }
        
        try:
            with st.spinner("Analyse en cours..."):
                response = requests.post(f"{API_URL}/predict", json={"data": data})
                response.raise_for_status()
                result = response.json()
                
                sleep_quality = result["sleep_quality"]
                recommendations = result["recommendations"]
                
                # Affichage des résultats
                st.write(f"Score de qualité du sommeil : {sleep_quality:.2f}/10")
                
                if sleep_quality > 8:
                    st.success("🟢 Qualité du Sommeil : Excellente")
                elif sleep_quality > 5:
                    st.warning("🟡 Qualité du Sommeil : Moyenne")
                else:
                    st.error("🔴 Qualité du Sommeil : Mauvaise")
                
                st.write("### 💡 Recommandations Personnalisées")
                for rec in recommendations:
                    st.info(rec)
                
                # Graphique radar des facteurs
                factors = ['Durée du Sommeil', 'Activité Physique', 'Pas Quotidiens', 'Fréquence Cardiaque', 'Stress']
                
                normalized_values = {
                    'Durée du Sommeil': min(sleep_duration / 8, 1),
                    'Activité Physique': min(physical_activity / 60, 1),
                    'Pas Quotidiens': min(daily_steps / 10000, 1),
                    'Fréquence Cardiaque': min(max(60, 120 - heart_rate) / 60, 1),
                    'Stress': 1 - (stress_level / 10)
                }
                
                fig = go.Figure(data=[go.Scatterpolar(
                    r=[normalized_values[factor] for factor in factors],
                    theta=factors,
                    fill='toself',
                    name='Vos Valeurs'
                )])
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False
                )
                
                st.plotly_chart(fig)
                
        except Exception as e:
            st.error(f"Une erreur est survenue : {str(e)}")

with tab2:
    st.write("### 📈 Suivi Temporel du Sommeil")
    
    if 'sleep_history' not in st.session_state:
        st.session_state.sleep_history = pd.DataFrame(columns=['Date', 'Qualité', 'Durée', 'Stress', 'Activité'])
    
    # Formulaire d'ajout manuel
    with st.form("ajout_manuel"):
        st.write("#### Ajouter une entrée manuelle")
        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("Date", value=pd.Timestamp.now())
            qualite = st.slider("Qualité du sommeil", 1, 10, 7)
            duree = st.number_input("Durée du sommeil (heures)", 4.0, 10.0, 7.0, 0.5)
        with col2:
            stress = st.slider("Niveau de stress", 1, 10, 5)
            activite = st.number_input("Activité physique (minutes)", 0, 120, 30)
        
        if st.form_submit_button("Ajouter l'entrée"):
            new_entry = pd.DataFrame({
                'Date': [date],
                'Qualité': [qualite],
                'Durée': [duree],
                'Stress': [stress],
                'Activité': [activite]
            })
            st.session_state.sleep_history = pd.concat([st.session_state.sleep_history, new_entry], ignore_index=True)
            st.success("Entrée ajoutée avec succès !")
    
    if not st.session_state.sleep_history.empty:
        # Sélection des métriques à afficher
        metrics = st.multiselect(
            "Choisir les métriques à afficher",
            ['Qualité', 'Durée', 'Stress', 'Activité'],
            default=['Qualité']
        )
        
        # Graphique d'évolution
        fig = px.line(st.session_state.sleep_history, x='Date', y=metrics,
                     title='Évolution des métriques de sommeil')
        st.plotly_chart(fig)
        
        # Statistiques
        st.write("#### 📊 Statistiques de suivi")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Qualité moyenne", f"{st.session_state.sleep_history['Qualité'].mean():.1f}/10")
        with col2:
            st.metric("Durée moyenne", f"{st.session_state.sleep_history['Durée'].mean():.1f}h")
        with col3:
            st.metric("Niveau de stress moyen", f"{st.session_state.sleep_history['Stress'].mean():.1f}/10")
        
        # Tableau des données
        st.write("#### 📋 Historique détaillé")
        st.dataframe(st.session_state.sleep_history.sort_values('Date', ascending=False))
    else:
        st.info("Aucune donnée de suivi disponible. Ajoutez des entrées pour voir les statistiques.")

with tab3:
    st.write("### 📊 Analyse Approfondie des Données")
    
    # Sélection du type d'analyse
    analysis_type = st.selectbox(
        "Choisir le type d'analyse",
        ["Distribution des données", "Corrélations", "Analyse par catégorie", "Matrice de corrélation", "Analyse par profession", "Stress et Sommeil"]
    )
    
    if analysis_type == "Distribution des données":
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(df, x='Quality of Sleep',
                              title='Distribution de la Qualité du Sommeil',
                              nbins=20)
            st.plotly_chart(fig1)
        with col2:
            fig2 = px.histogram(df, x='Sleep Duration',
                              title='Distribution de la Durée du Sommeil',
                              nbins=20)
            st.plotly_chart(fig2)
            
    elif analysis_type == "Corrélations":
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.scatter(df, x='Physical Activity Level', y='Quality of Sleep',
                            title="Activité Physique vs Qualité du Sommeil",
                            trendline="ols")
            st.plotly_chart(fig1)
        with col2:
            fig2 = px.scatter(df, x='Stress Level', y='Quality of Sleep',
                            title="Stress vs Qualité du Sommeil",
                            trendline="ols")
            st.plotly_chart(fig2)
            
    elif analysis_type == "Analyse par catégorie":
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.box(df, x='BMI', y='Quality of Sleep',
                         title='Qualité du Sommeil par IMC')
            st.plotly_chart(fig1)
        with col2:
            fig2 = px.box(df, x='Blood Pressure', y='Quality of Sleep',
                         title='Qualité du Sommeil par Pression Artérielle')
            st.plotly_chart(fig2)
            
    elif analysis_type == "Matrice de corrélation":
        # Sélectionner les colonnes numériques pertinentes
        numeric_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 
                       'Physical Activity Level', 'Stress Level', 
                       'Heart Rate', 'Daily Steps']
        
        # Calculer la matrice de corrélation
        corr_matrix = df[numeric_cols].corr()
        
        # Créer la heatmap avec plotly
        fig = px.imshow(corr_matrix,
                       labels=dict(color="Coefficient de corrélation"),
                       x=numeric_cols,
                       y=numeric_cols,
                       color_continuous_scale="RdBu",
                       aspect="auto")
        
        fig.update_layout(
            title="Matrice de Corrélation des Variables",
            xaxis_title="",
            yaxis_title="",
            width=800,
            height=800
        )
        
        st.plotly_chart(fig)
        
        st.write("""
        #### 📌 Interprétation des corrélations
        - Une corrélation proche de 1 (rouge foncé) indique une forte relation positive
        - Une corrélation proche de -1 (bleu foncé) indique une forte relation négative
        - Une corrélation proche de 0 (blanc) indique une faible relation
        """)
        
    elif analysis_type == "Stress et Sommeil":
        # Créer le scatter plot
        fig = px.scatter(df, 
                    x='Stress Level', 
                    y='Sleep Duration',
                    title="Relation entre Niveau de Stress et Durée du Sommeil",
                    labels={'Stress Level': 'Niveau de Stress', 'Sleep Duration': 'Durée du Sommeil (heures)'},
                    height=600,
                    trendline="ols")  # Ajoute une ligne de régression
        
        # Personnaliser le graphique
        fig.update_layout(
            xaxis=dict(
                range=[0, 10],
                dtick=1
            ),
            yaxis=dict(
                range=[0, df['Sleep Duration'].max() + 1],
                dtick=1
            )
        )
        
        # Afficher le graphique
        st.plotly_chart(fig)
        
        # Calculer la corrélation
        correlation = df['Stress Level'].corr(df['Sleep Duration']).round(3)
        
        # Afficher l'interprétation
        st.write(f"""
        #### 📊 Analyse de la relation
        - **Corrélation** : {correlation}
        - Une corrélation négative indique que plus le stress augmente, plus la durée de sommeil diminue
        - La ligne de tendance (en rouge) montre la tendance générale de la relation
        - Chaque point représente une personne dans le dataset
        """)
        
    else:  # Analyse par profession
        # Créer le box plot
        fig = px.box(df, 
                    x='Occupation', 
                    y='Quality of Sleep',
                    title="Distribution de la Qualité du Sommeil par Profession",
                    labels={'Occupation': 'Profession', 'Quality of Sleep': 'Qualité du Sommeil'},
                    height=500)
        
        # Personnaliser le graphique
        fig.update_layout(
            xaxis_tickangle=45,
            yaxis=dict(
                range=[0, 10],
                dtick=1
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig)
        
        # Afficher les statistiques détaillées
        st.write("#### 📊 Statistiques détaillées par profession")
        prof_stats = df.groupby('Occupation').agg({
            'Quality of Sleep': ['count', 'mean', 'median', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        }).round(2)
        prof_stats.columns = ['Nombre', 'Moyenne', 'Médiane', 'Écart-type', 'Q1', 'Q3']
        st.dataframe(prof_stats.sort_values(('Médiane'), ascending=False))

        # Ajouter une note explicative
        st.write("""
        #### 📌 Note sur l'interprétation
        - La boîte montre les quartiles (Q1, médiane, Q3)
        - Les "moustaches" montrent la distribution des données
        - Les points individuels sont les valeurs extrêmes
        - Un grand écart entre Q1 et Q3 indique une grande variabilité
        """)
