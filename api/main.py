from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict, List, Union
import os
import json
import pandas as pd

app = FastAPI()

# Obtenir le chemin absolu du répertoire des modèles
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'comparison_results')

try:
    # Charger les métadonnées du modèle
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    features = metadata['selected_features']
    numeric_features = metadata['numeric_features']
    model = joblib.load(os.path.join(MODEL_DIR, 'best_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement des métadonnées : {str(e)}")

class SleepData(BaseModel):
    """Modèle de données pour les prédictions de qualité du sommeil"""
    Age: int
    Gender: str
    Sleep_Duration: float
    Physical_Activity_Level: int
    Stress_Level: int
    Heart_Rate: int
    Daily_Steps: int
    Blood_Pressure: str

def get_sleep_recommendations(data: Dict[str, Union[float, str]], sleep_quality: float) -> Dict[str, str]:
    """Génère des recommandations basées sur les données et la qualité du sommeil prédite"""
    recommendations = []
    
    if sleep_quality < 7:
        if data['Sleep_Duration'] < 7:
            recommendations.append("Augmentez votre durée de sommeil à au moins 7-8 heures par nuit")
        if data['Physical_Activity_Level'] < 60:
            recommendations.append("Augmentez votre niveau d'activité physique")
        if data['Stress_Level'] > 6:
            recommendations.append("Essayez des techniques de réduction du stress comme la méditation")
        if data['Daily_Steps'] < 8000:
            recommendations.append("Visez au moins 8000 pas par jour")
            
    return {"recommendations": recommendations}

@app.post("/predict")
async def predict_sleep_quality(sleep_data: SleepData) -> Dict[str, Union[float, List[str]]]:
    """Prédit la qualité du sommeil et fournit des recommandations"""
    try:
        # Préparer les données
        data = {
            'Age': sleep_data.Age,
            'Gender_num': 1 if sleep_data.Gender.lower() == 'male' else 0,
            'Sleep_Duration': sleep_data.Sleep_Duration,
            'Physical_Activity_Level': sleep_data.Physical_Activity_Level,
            'Stress_Level': sleep_data.Stress_Level,
            'Heart_Rate': sleep_data.Heart_Rate,
            'Daily_Steps': sleep_data.Daily_Steps,
            'Blood_Pressure_num': 1 if sleep_data.Blood_Pressure.lower() == 'high' else 0
        }
        
        # Créer un dictionnaire avec les noms de colonnes corrects
        model_data = {
            'Age': data['Age'],
            'Gender_num': data['Gender_num'],
            'Sleep Duration': data['Sleep_Duration'],
            'Physical Activity Level': data['Physical_Activity_Level'],
            'Stress Level': data['Stress_Level'],
            'Heart Rate': data['Heart_Rate'],
            'Daily Steps': data['Daily_Steps'],
            'Blood_Pressure_num': data['Blood_Pressure_num']
        }
        
        # Créer un DataFrame pour la standardisation
        X_df = pd.DataFrame([model_data])
        
        # Standardiser les features numériques
        numeric_columns = [col for col in numeric_features if col in X_df.columns]
        X_df[numeric_columns] = scaler.transform(X_df[numeric_columns])
        
        # Extraire les features dans le bon ordre
        X = np.array([[X_df[f].iloc[0] for f in features]])
        
        # Faire la prédiction
        prediction = float(model.predict(X)[0])
        
        # Générer des recommandations
        recommendations = get_sleep_recommendations(data, prediction)
        
        return {
            "sleep_quality": prediction,
            "recommendations": recommendations["recommendations"]
        }
        
    except Exception as e:
        print(f"Erreur dans /predict : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Vérifie que l'API fonctionne"""
    return {"status": "healthy"}

@app.get("/info")
async def model_info() -> Dict[str, Union[str, List[str]]]:
    """Retourne les informations sur le modèle"""
    try:
        return {
            "model_type": metadata['model_type'],
            "features": metadata['selected_features'],
            "numeric_features": metadata['numeric_features'],
            "feature_ranges": metadata['feature_ranges']
        }
    except Exception as e:
        print(f"Erreur dans /info : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
