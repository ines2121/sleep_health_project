# Projet d'Analyse de la Qualité du Sommeil

Ce projet utilise des techniques de machine learning pour prédire la qualité du sommeil en fonction de divers facteurs de santé et de style de vie.

## Structure du Projet

```
sleep_health_project/
├── api/                    # API FastAPI pour servir le modèle
│   └── main.py
├── data/                   # Données brutes et prétraitées
│   ├── processed/         # Données prétraitées et métadonnées
│   └── Sleep_health_and_lifestyle_dataset.csv
├── models/                 # Modèles entraînés et résultats
│   └── comparison_results/
├── notebooks/             # Notebooks Jupyter pour l'analyse et l'entraînement
│   ├── 1_data_preprocessing.ipynb  # Prétraitement et analyse des données
│   └── 2_model_training.ipynb      # Entraînement et évaluation des modèles
├── scripts/               # Scripts utilitaires
│   └── generate_synthetic_data.py
├── streamlit_app/        # Application web Streamlit
│   └── app.py
├── tests/                # Tests unitaires
│   └── test_model.py
├── requirements.txt      # Dépendances Python
└── README.md            # Documentation du projet
```

## Installation

1. Cloner le repository
2. Créer un environnement virtuel : `python -m venv venv`
3. Activer l'environnement : `source venv/bin/activate`
4. Installer les dépendances : `pip install -r requirements.txt`

## Utilisation

1. **Analyse des Données et Entraînement**
   - Ouvrir et exécuter `notebooks/1_data_preprocessing.ipynb`
   - Puis exécuter `notebooks/2_model_training.ipynb`

2. **Application Web**
   - Lancer l'application : `streamlit run streamlit_app/app.py`

3. **API**
   - Lancer l'API : `uvicorn api.main:app --reload`

## Fonctionnalités

- Analyse exploratoire des données de sommeil
- Détection et analyse des outliers
- Comparaison de différents modèles de ML
- Validation croisée imbriquée pour une évaluation robuste
- Interface web interactive avec Streamlit
- API REST pour les prédictions

## Résultats

Le modèle SVR (Support Vector Regression) a été sélectionné comme le meilleur modèle avec :
- R² score : ~0.85
- RMSE : ~0.4

Les détails complets sont disponibles dans `models/comparison_results/`.

## 👩‍💻 Développé par

Ines Hammouch
