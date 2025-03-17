import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests
import json
from datetime import datetime
import os

# Configuration de la page
st.set_page_config(
    page_title="Analyse de la Santé du Sommeil",
    page_icon="😴",
    layout="wide"
)

# Configuration des options en français
st.markdown("""
    <style>
        .stDownloadButton button {
            visibility: hidden;
        }
        .stDownloadButton button::after {
            visibility: visible;
            content: 'Télécharger';
        }
        .stDeployButton button {
            visibility: hidden;
        }
        .stDeployButton button::after {
            visibility: visible;
            content: 'Déployer';
        }
        button[title="View fullscreen"] {
            visibility: hidden;
        }
        button[title="View fullscreen"]::after {
            visibility: visible;
            content: 'Plein écran';
        }
    </style>
""", unsafe_allow_html=True)

# Dictionnaire de traduction des caractéristiques
feature_names = {
    'Person ID': 'ID Personne',
    'Gender': 'Genre',
    'Age': 'Âge',
    'Occupation': 'Profession',
    'Sleep Duration': 'Durée du Sommeil',
    'Quality of Sleep': 'Qualité du Sommeil',
    'Physical Activity Level': "Niveau d'Activité Physique",
    'Stress Level': 'Niveau de Stress',
    'BMI Category': 'Catégorie IMC',
    'Blood Pressure': 'Pression Artérielle',
    'Heart Rate': 'Fréquence Cardiaque',
    'Daily Steps': 'Pas Quotidiens',
    'Sleep Disorder': 'Trouble du Sommeil'
}

# URL de l'API
API_URL = os.environ.get('API_URL', 'http://api:8002')

# Chargement des données d'exemple
@st.cache_data
def load_data():
    return pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")

df = load_data()

st.title("😴 Analyse de la Santé du Sommeil")

# Sidebar avec informations
st.sidebar.title("ℹ️ Informations")
st.sidebar.write("""
Cette application analyse la qualité du sommeil et le mode de vie pour fournir 
des insights personnalisés sur la qualité du sommeil.

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
    
    # Création de colonnes pour l'entrée des données
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Genre", ["Homme", "Femme"])
        occupation = st.selectbox("Profession", df['Occupation'].unique())
        physical_activity = st.slider("Niveau d'Activité Physique (minutes/jour)", 0, 120, 30)
    
    with col2:
        sleep_duration = st.slider("Durée du Sommeil (heures)", 4.0, 10.0, 7.0, 0.1)
        stress_level = st.slider("Niveau de Stress (1-10)", 1, 10, 5)
        daily_steps = st.number_input("Pas Quotidiens", min_value=2000, max_value=25000, value=8000)
    
    with col3:
        bmi = st.number_input("IMC", min_value=15.0, max_value=40.0, value=23.0)
        blood_pressure = st.selectbox("Pression Artérielle", ["Normale", "Élevée"])
        heart_rate = st.number_input("Fréquence Cardiaque", min_value=60, max_value=120, value=75)

    # Bouton d'analyse
    if st.button("🔍 Analyser mon sommeil", type="primary"):
        # Préparation des données
        data = {
            'Age': age,
            'Gender': 'Male' if gender == "Homme" else "Female",
            'Occupation': occupation,
            'Sleep Duration': sleep_duration,
            'Physical Activity Level': physical_activity,
            'Stress Level': stress_level,
            'BMI': bmi,
            'Blood Pressure': blood_pressure,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps
        }
        
        try:
            with st.spinner("Analyse en cours..."):
                response = requests.post(f"{API_URL}/predict", json={"data": data})
                result = response.json()
                sleep_quality = result["sleep_quality"]
                recommendations = result["recommendations"]
                
                # Affichage des résultats
                if sleep_quality > 8:
                    st.success("🟢 Qualité du Sommeil : Excellente")
                elif sleep_quality > 5:
                    st.warning("🟡 Qualité du Sommeil : Moyenne")
                else:
                    st.error("🔴 Qualité du Sommeil : Mauvaise")
                
                # Affichage des recommandations
                st.write("### 💡 Recommandations Personnalisées")
                for rec in recommendations:
                    st.info(rec)
                
                # Graphique radar des facteurs
                st.write("### 📈 Analyse des Facteurs")
                
                # Normalisation des valeurs pour le graphique radar
                factors = [
                    'Durée du Sommeil',
                    'Activité Physique',
                    'Pas Quotidiens',
                    'Fréquence Cardiaque',
                    'Stress'
                ]
                
                max_values = {
                    'Durée du Sommeil': 10,
                    'Activité Physique': 120,
                    'Pas Quotidiens': 25000,
                    'Fréquence Cardiaque': 120,
                    'Stress': 10
                }
                
                normalized_values = {
                    'Durée du Sommeil': min(sleep_duration / 8, 1),  # Optimal autour de 8h
                    'Activité Physique': min(physical_activity / 60, 1),  # Optimal autour de 60min
                    'Pas Quotidiens': min(daily_steps / 10000, 1),  # Optimal autour de 10000 pas
                    'Fréquence Cardiaque': min(max(60, 120 - heart_rate) / 60, 1),  # Optimal autour de 60-80
                    'Stress': 1 - (stress_level / 10)  # Inverse pour que moins de stress soit mieux
                }
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=[normalized_values[factor] for factor in factors],
                    theta=factors,
                    fill='toself',
                    name='Vos valeurs'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            tickmode='array',
                            ticktext=['0%', '25%', '50%', '75%', '100%'],
                            tickvals=[0, 0.25, 0.5, 0.75, 1],
                            tickfont=dict(color='red')
                        )
                    ),
                    showlegend=False,
                    title={
                        'text': "Analyse des Facteurs de Sommeil",
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    }
                )
                
                st.plotly_chart(fig)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Erreur : Impossible de se connecter au service d'analyse")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

with tab2:
    st.write("### 📈 Suivi Temporel du Sommeil")
    
    # Initialiser l'historique s'il n'existe pas
    if 'sleep_history' not in st.session_state:
        st.session_state.sleep_history = []
    
    # Afficher l'historique
    if st.session_state.sleep_history:
        history_df = pd.DataFrame(st.session_state.sleep_history)
        
        # Graphique d'évolution
        fig_evolution = px.line(
            history_df,
            x='Date',
            y='Qualité du Sommeil',
            title="Évolution de votre Qualité de Sommeil",
            markers=True
        )
        st.plotly_chart(fig_evolution)
        
        # Tableau récapitulatif
        st.write("### 📋 Historique Détaillé")
        st.dataframe(history_df.style.highlight_max(subset=['Qualité du Sommeil'], color='#90EE90'))
    else:
        st.info("Aucun historique disponible. Faites votre première analyse !")

with tab3:
    st.write("### 📊 Visualisations des Données")
    
    viz_type = st.selectbox(
        "Choisir le type de visualisation",
        ["Distribution", "Corrélation", "Tendances par Profession", "Analyse par Durée", "Analyse Multifactorielle"]
    )
    
    if viz_type == "Distribution":
        fig = px.box(
            df,
            x="Quality of Sleep",
            y="Sleep Duration",
            color="Quality of Sleep",
            title="Distribution de la Durée du Sommeil par Qualité",
            labels=feature_names
        )
        st.plotly_chart(fig)
        
    elif viz_type == "Corrélation":
        # Créer une copie du dataframe pour la corrélation
        df_corr = df.copy()
        
        # Convertir les variables catégorielles en numériques
        df_corr['Gender_num'] = (df_corr['Gender'] == 'Male').astype(int)
        df_corr['Blood_Pressure_num'] = (df_corr['Blood Pressure'] == 'High').astype(int)
        
        # Sélectionner uniquement les colonnes numériques pour la corrélation
        numeric_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                       'Stress Level', 'Heart Rate', 'Daily Steps', 'Gender_num', 'Blood_Pressure_num']
        
        correlation = df_corr[numeric_cols].corr()
        
        # Traduire les noms des colonnes
        correlation.index = [feature_names.get(col, col.replace('_num', '')) for col in correlation.index]
        correlation.columns = [feature_names.get(col, col.replace('_num', '')) for col in correlation.columns]
        
        fig = px.imshow(
            correlation,
            labels=dict(color="Corrélation"),
            color_continuous_scale="RdBu",
            title="Matrice de Corrélation"
        )
        
        # Ajuster la taille et la rotation des labels
        fig.update_layout(
            height=700,
            width=700,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig)
        
    elif viz_type == "Tendances par Profession":
        fig = px.box(
            df,
            x="Occupation",
            y="Quality of Sleep",
            title="Qualité du Sommeil par Profession",
            labels=feature_names
        )
        st.plotly_chart(fig)
        
    elif viz_type == "Analyse par Durée":
        # Créer des catégories de durée de sommeil
        df['Sleep Duration Category'] = pd.cut(
            df['Sleep Duration'],
            bins=[0, 6, 7, 8, 12],
            labels=['< 6h', '6-7h', '7-8h', '> 8h']
        )
        
        fig = px.box(
            df,
            x="Sleep Duration Category",
            y="Quality of Sleep",
            color="Sleep Duration Category",
            title="Qualité du Sommeil selon la Durée",
            labels=feature_names
        )
        st.plotly_chart(fig)
        
        st.write("""
        **Observations :**
        - Une durée de sommeil entre 7 et 8 heures est associée à une meilleure qualité de sommeil
        - Les personnes dormant moins de 6 heures ont généralement une qualité de sommeil plus faible
        - Un sommeil trop long (> 8h) n'améliore pas nécessairement la qualité
        """)
        
    elif viz_type == "Analyse Multifactorielle":
        st.write("""
        Ce graphique montre les relations entre le stress, l'activité physique et la qualité du sommeil.
        La taille des points représente la durée du sommeil.
        """)
        
        fig = px.scatter(
            df,
            x="Stress Level",
            y="Quality of Sleep",
            color="Physical Activity Level",
            size="Sleep Duration",
            title="Impact du Stress et de l'Activité Physique sur le Sommeil",
            labels={
                "Stress Level": "Niveau de Stress",
                "Quality of Sleep": "Qualité du Sommeil",
                "Sleep Duration": "Durée du Sommeil",
                "Physical Activity Level": "Niveau d'Activité Physique (minutes)"
            },
            color_continuous_scale='Blues',  # Utiliser un dégradé de bleus
            size_max=25
        )
        
        # Personnalisation du graphique avec thème gris foncé
        fig.update_layout(
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            title_font_color='white',
            legend_title_text="Minutes d'Activité Physique",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color='white')
            ),
            font=dict(color='white')
        )
        
        # Personnalisation de la barre de couleur
        fig.update_coloraxes(
            colorbar=dict(
                title="Minutes d'activité",
                titleside="right",
                tickcolor='white',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            )
        )
        
        # Ajout des axes et de la grille avec couleurs adaptées
        fig.update_xaxes(
            gridcolor='rgba(255, 255, 255, 0.1)',
            showgrid=True,
            color='white',
            linecolor='white'
        )
        fig.update_yaxes(
            gridcolor='rgba(255, 255, 255, 0.1)',
            showgrid=True,
            color='white',
            linecolor='white'
        )
        
        st.plotly_chart(fig)

# Mise à jour de l'historique après l'analyse
if "sleep_quality" in locals():
    st.session_state.sleep_history.append({
        'Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'Qualité du Sommeil': sleep_quality,
        'Durée du Sommeil': sleep_duration,
        'Niveau de Stress': stress_level,
        'Activité Physique': physical_activity,
        'Pas Quotidiens': daily_steps
    })
