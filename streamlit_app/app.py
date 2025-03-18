import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import os
import json

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

# Charger les métadonnées du modèle
@st.cache_data
def load_model_metadata():
    try:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'comparison_results')
        with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Erreur lors du chargement des métadonnées : {str(e)}")
        return None

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
metadata = load_model_metadata()
feature_ranges = metadata['feature_ranges'] if metadata else {}

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
        age_range = feature_ranges.get('Age', {'min': 18, 'max': 100})
        age = st.number_input("Âge", 
                            min_value=int(age_range['min']), 
                            max_value=int(age_range['max']), 
                            value=int(age_range['mean']))
        gender = st.selectbox("Genre", ["Homme", "Femme"])
        occupation = st.selectbox("Profession", occupations)
        activity_range = feature_ranges.get('Physical Activity Level', {'min': 0, 'max': 120})
        physical_activity = st.slider("Niveau d'Activité Physique (minutes/jour)", 
                                   int(activity_range['min']), 
                                   int(activity_range['max']), 
                                   int(activity_range['mean']))
    
    with col2:
        sleep_range = feature_ranges.get('Sleep Duration', {'min': 4, 'max': 10})
        sleep_duration = st.slider("Durée du Sommeil (heures)", 
                                float(sleep_range['min']), 
                                float(sleep_range['max']), 
                                float(sleep_range['mean']), 
                                0.1)
        stress_range = feature_ranges.get('Stress Level', {'min': 1, 'max': 10})
        stress_level = st.slider("Niveau de Stress (1-10)", 
                              1, 
                              int(stress_range['max']), 
                              int(stress_range['mean']))
        steps_range = feature_ranges.get('Daily Steps', {'min': 2000, 'max': 25000})
        daily_steps = st.number_input("Pas Quotidiens", 
                                    min_value=int(steps_range['min']), 
                                    max_value=int(steps_range['max']), 
                                    value=int(steps_range['mean']))
    
    with col3:
        bmi = st.number_input("IMC", min_value=15.0, max_value=40.0, value=23.0)
        blood_pressure = st.selectbox("Pression Artérielle", ["Normale", "Élevée"])
        heart_range = feature_ranges.get('Heart Rate', {'min': 60, 'max': 120})
        heart_rate = st.number_input("Fréquence Cardiaque", 
                                   min_value=int(heart_range['min']), 
                                   max_value=int(heart_range['max']), 
                                   value=int(heart_range['mean']))

    if st.button("🔍 Analyser mon sommeil", type="primary"):
        api_data = {
            'Age': age,
            'Gender': 'Male' if gender == "Homme" else "Female",
            'Sleep_Duration': sleep_duration,
            'Physical_Activity_Level': physical_activity,
            'Stress_Level': stress_level,
            'Heart_Rate': heart_rate,
            'Daily_Steps': daily_steps,
            'Blood_Pressure': blood_pressure_mapping[blood_pressure]
        }
        
        try:
            with st.spinner("Analyse en cours..."):
                response = requests.post(f"{API_URL}/predict", json=api_data)
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
        st.write("### 📊 Statistiques")
        stats = st.session_state.sleep_history[metrics].describe()
        st.dataframe(stats)
        
        # Option de téléchargement
        csv = st.session_state.sleep_history.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger l'historique",
            data=csv,
            file_name='sleep_history.csv',
            mime='text/csv'
        )

with tab3:
    st.write("### 📊 Analyse Approfondie des Données")
    
    # Sélection du type d'analyse
    analysis_type = st.selectbox(
        "Type d'analyse",
        ["Corrélations", "Distribution des Variables", "Histogramme du Sommeil", "Comparaison des Modèles"]
    )
    
    if analysis_type == "Corrélations":
        st.write("## 📊 Matrice de Corrélation")
        
        # Sélectionner les colonnes numériques
        numeric_cols = ['Quality of Sleep', 'Sleep Duration', 'Physical Activity Level',
                       'Stress Level', 'Heart Rate', 'Daily Steps', 'Age']
        
        # Calculer la matrice de corrélation
        corr_matrix = df[numeric_cols].corr()
        
        # Créer la heatmap avec plotly
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Matrice de Corrélation des Variables',
            xaxis_title='Variables',
            yaxis_title='Variables'
        )
        
        st.plotly_chart(fig)
        
        # Afficher les corrélations les plus fortes avec la qualité du sommeil
        st.write("### 🎯 Corrélations avec la Qualité du Sommeil")
        sleep_corr = corr_matrix['Quality of Sleep'].sort_values(ascending=False)
        
        for var, corr in sleep_corr.items():
            if var != 'Quality of Sleep':
                if abs(corr) > 0.5:
                    emoji = "🟢" if corr > 0 else "🔴"
                elif abs(corr) > 0.3:
                    emoji = "🟡" if corr > 0 else "🟠"
                else:
                    emoji = "⚪️"
                st.write(f"{emoji} {var}: {corr:.3f}")
    
    elif analysis_type == "Distribution des Variables":
        st.write("## 📊 Distribution des Variables")
        
        # Sélection de la variable
        variable = st.selectbox(
            "Choisir une variable",
            ['Quality of Sleep', 'Sleep Duration', 'Physical Activity Level',
             'Stress Level', 'Heart Rate', 'Daily Steps', 'Age']
        )
        
        # Créer un subplot avec histogramme et boxplot
        fig = make_subplots(rows=2, cols=1, 
                          subplot_titles=('Distribution', 'Boîte à Moustaches'),
                          row_heights=[0.7, 0.3])
        
        # Ajouter l'histogramme
        fig.add_trace(
            go.Histogram(x=df[variable], name='Distribution'),
            row=1, col=1
        )
        
        # Ajouter le boxplot
        fig.add_trace(
            go.Box(x=df[variable], name='Boîte à Moustaches'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text=f"Analyse de {variable}")
        st.plotly_chart(fig)
        
        # Statistiques descriptives
        st.write("### 📈 Statistiques Descriptives")
        stats = df[variable].describe()
        st.dataframe(stats)
    
    elif analysis_type == "Histogramme du Sommeil":
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogramme de la durée du sommeil
            fig = px.histogram(df, x='Sleep Duration',
                             title='Distribution de la Durée du Sommeil',
                             labels={'Sleep Duration': 'Durée du Sommeil (heures)',
                                    'count': 'Nombre de Personnes'})
            st.plotly_chart(fig)
        
        with col2:
            # Histogramme de la qualité du sommeil
            fig = px.histogram(df, x='Quality of Sleep',
                             title='Distribution de la Qualité du Sommeil',
                             labels={'Quality of Sleep': 'Qualité du Sommeil (1-10)',
                                    'count': 'Nombre de Personnes'})
            st.plotly_chart(fig)
    
    elif analysis_type == "Comparaison des Modèles":
        st.write("## 📊 Comparaison des Modèles de Prédiction")
        
        # Charger les résultats de comparaison
        comparison_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'models', 'comparison_results', 'model_comparison.csv')
        
        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path)
            
            # Afficher les résultats détaillés
            st.write("### 📋 Résultats Détaillés")
            st.dataframe(comparison_df)
