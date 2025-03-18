from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict, List, Union
import os
import json

app = FastAPI()

# Obtenir le chemin absolu du répertoire des modèles
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'comparison_results')

# Charger les métadonnées pour avoir la liste des features
try:
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    features = metadata['selected_features']
    numeric_features = metadata['numeric_features']
    model = joblib.load(os.path.join(MODEL_DIR, 'best_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {str(e)}")

class SleepData(BaseModel):
    data: Dict[str, Union[float, str]]

def get_sleep_recommendations(data: Dict[str, Union[float, str]], sleep_quality: float) -> List[str]:
    recommendations = []
    
    if data['Sleep Duration'] < 7:
        recommendations.append("🛏️ Essayez d'augmenter votre temps de sommeil à au moins 7 heures par nuit.")
    elif data['Sleep Duration'] > 9:
        recommendations.append("⚠️ Dormir plus de 9 heures peut affecter la qualité du sommeil. Visez 7-8 heures.")
    
    if data['Physical Activity Level'] < 30:
        recommendations.append("🏃‍♂️ Augmentez votre activité physique à au moins 30 minutes par jour.")
    
    if data['Stress Level'] > 6:
        recommendations.append("🧘‍♂️ Votre niveau de stress est élevé. Essayez la méditation ou le yoga.")
    
    if data['Daily Steps'] < 8000:
        recommendations.append("👣 Essayez d'atteindre au moins 8000 pas par jour pour améliorer votre sommeil.")
    
    if data['Heart Rate'] > 80:
        recommendations.append("❤️ Une fréquence cardiaque élevée peut affecter le sommeil. Pratiquez des exercices de respiration.")
    
    return recommendations

@app.post("/predict")
def predict_sleep_quality(sleep_data: SleepData):
    try:
        data = sleep_data.data
        X = np.zeros((1, len(features)))
        
        # Préparer les données dans le bon ordre des features
        for i, feature in enumerate(features):
            if feature == 'Gender_num':
                X[0, i] = 1 if data['Gender'] == 'Male' else 0
            elif feature == 'Blood_Pressure_num':
                X[0, i] = 1 if data['Blood Pressure'] == 'High' else 0
            else:
                X[0, i] = data[feature]
        
        # Créer un DataFrame pour faciliter la standardisation
        import pandas as pd
        X_df = pd.DataFrame(X, columns=features)
        
        # Standardiser uniquement les features numériques
        X_df[numeric_features] = scaler.transform(X_df[numeric_features])
        
        # Faire la prédiction
        prediction = float(model.predict(X_df)[0])
        
        # Obtenir les recommandations
        recommendations = get_sleep_recommendations(data, prediction)
        
        return {
            "sleep_quality": prediction,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/model-info")
def model_info():
    return {
        "model_type": "SVR",
        "features": features,
        "numeric_features": numeric_features,
        "metadata": metadata,
        "performance": {
            "r2_mean": metadata['cv_results']['r2_mean'],
            "r2_std": metadata['cv_results']['r2_std'],
            "rmse_mean": metadata['cv_results']['rmse_mean'],
            "rmse_std": metadata['cv_results']['rmse_std']
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
