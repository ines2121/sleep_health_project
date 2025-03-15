import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Configuration de la page
st.set_page_config(
    page_title="Analyse de la Santé du Sommeil",
    page_icon="😴",
    layout="wide"
)

# Dictionnaire de traduction des caractéristiques
feature_names = {
    'Person ID': 'ID Personne',
    'Gender': 'Genre',
    'Age': 'Âge',
    'Occupation': 'Profession',
    'Sleep Duration': 'Durée du Sommeil',
    'Quality of Sleep': 'Qualité du Sommeil',
    'Physical Activity Level': 'Niveau d\'Activité Physique',
    'Stress Level': 'Niveau de Stress',
    'BMI Category': 'Catégorie IMC',
    'Blood Pressure': 'Pression Artérielle',
    'Heart Rate': 'Fréquence Cardiaque',
    'Daily Steps': 'Pas Quotidiens',
    'Sleep Disorder': 'Trouble du Sommeil'
}

# URL de l'API depuis la variable d'environnement
API_URL = os.getenv('API_URL', 'http://localhost:8002')

# Chargement des données d'exemple
@st.cache_data
def load_data():
    return pd.read_csv("/Users/ines/Desktop/sleep_health_project/data/Sleep_health_and_lifestyle_dataset.csv")

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
                max_values = {
                    'Durée du Sommeil': 10,
                    'Activité Physique': 120,
                    'Stress': 10,
                    'Pas Quotidiens': 25000,
                    'Fréquence Cardiaque': 120
                }
                
                normalized_values = {
                    'Durée du Sommeil': sleep_duration / max_values['Durée du Sommeil'],
                    'Activité Physique': physical_activity / max_values['Activité Physique'],
                    'Stress': (10 - stress_level) / 10,  # Inverse pour que moins de stress soit mieux
                    'Pas Quotidiens': daily_steps / max_values['Pas Quotidiens'],
                    'Fréquence Cardiaque': (120 - heart_rate) / 50  # Normalise pour que plus bas soit mieux
                }
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=list(normalized_values.values()),
                    theta=list(normalized_values.keys()),
                    fill='toself',
                    name='Vos valeurs'
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Analyse des Facteurs de Sommeil"
                )
                
                st.plotly_chart(fig)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Erreur : Impossible de se connecter au service d'analyse")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

with tab2:
    st.write("### 📈 Suivi Temporel du Sommeil")
    
    # Charger les données de suivi depuis la session state
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
        st.write("### 📋 Historique Détailé")
        st.dataframe(history_df)
        
        # Export des données
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Exporter l'historique (Format CSV)",
            data=csv,
            file_name="mon_historique_sommeil.csv",
            mime="text/csv"
        )
    else:
        st.info("👋 Utilisez l'onglet Analyse pour commencer votre suivi du sommeil !")

with tab3:
    st.write("### 📊 Visualisation des Données")
    
    # Sélection du type de visualisation
    viz_type = st.selectbox(
        "Choisissez votre visualisation",
        ["Distribution du Sommeil", "Corrélation", "Tendances par Profession", "Analyse par Durée", "Impact du Stress"]
    )
    
    if viz_type == "Distribution du Sommeil":
        fig = px.histogram(
            df,
            x="Sleep Duration",
            color="Quality of Sleep",
            nbins=20,
            title="Distribution de la Durée du Sommeil par Qualité",
            labels=feature_names
        )
        st.plotly_chart(fig)
        
    elif viz_type == "Corrélation":
        # Sélection des colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr,
            labels=dict(color="Corrélation"),
            color_continuous_scale="RdBu",
            title="Matrice de Corrélation"
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
        # Créer des catégories d'heures de sommeil
        df['Sleep Category'] = pd.cut(
            df['Sleep Duration'],
            bins=[0, 6, 7, 8, 12],
            labels=['< 6h', '6-7h', '7-8h', '> 8h']
        )
        
        fig = px.violin(
            df,
            x='Sleep Category',
            y='Quality of Sleep',
            box=True,
            title="Qualité du Sommeil selon la Durée",
            labels=feature_names
        )
        st.plotly_chart(fig)
        
        st.write("""
        ### 📝 Comprendre les Durées de Sommeil
        - **< 6h** : Sommeil insuffisant, attention aux risques pour la santé
        - **6-7h** : Durée minimale acceptable pour un adulte
        - **7-8h** : Durée idéale pour un bon repos
        - **> 8h** : Sommeil prolongé, possible signe de fatigue excessive
        """)
        
    elif viz_type == "Impact du Stress":
        # Créer la figure
        fig = px.scatter(
            df,
            x="Stress Level",
            y="Quality of Sleep",
            color="Physical Activity Level",  # Utiliser la valeur numérique pour le dégradé
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
        
        # Analyse de corrélation et explications
        stress_corr = df['Stress Level'].corr(df['Quality of Sleep'])
        st.write(f"""
        ### 📊 Analyse de l'Impact du Stress
        
        **Interprétation des couleurs :**
        La couleur des points représente le niveau d'activité physique :
        - Plus le bleu est clair, plus l'activité physique est faible
        - Plus le bleu est foncé, plus l'activité physique est élevée
        
        **La taille des points** représente la durée du sommeil : plus le point est grand, plus la durée de sommeil est longue.
        
        **Observations :**
        - Corrélation stress/sommeil : {stress_corr:.2f}
        - Un niveau de stress élevé est généralement associé à une moins bonne qualité de sommeil
        - Les personnes ayant une activité physique plus élevée (points bleu foncé) semblent mieux gérer l'impact du stress sur leur sommeil
        """)

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

# Footer
st.markdown("---")
st.markdown("*Cette application est un outil d'aide à la décision et ne remplace pas l'avis d'un professionnel de santé.*")
