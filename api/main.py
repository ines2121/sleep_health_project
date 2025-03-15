from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict, List, Union
import os
import json

app = FastAPI()

# Charger le modèle et le scaler
model = joblib.load('../models/saved_models/best_sleep_model.joblib')
scaler = joblib.load('../models/saved_models/scaler.joblib')

# Charger la liste des features
with open('../models/saved_models/feature_list.txt', 'r') as f:
    features = f.read().splitlines()

# Modèle de données pour les prédictions
class SleepData(BaseModel):
    data: Dict[str, Union[float, str]]

def get_sleep_recommendations(data: Dict[str, Union[float, str]], sleep_quality: float) -> List[str]:
    recommendations = []
    
    # Recommandations basées sur la durée du sommeil
    if data['Sleep Duration'] < 7:
        recommendations.append("🛏️ Essayez d'augmenter votre temps de sommeil à au moins 7 heures par nuit.")
    elif data['Sleep Duration'] > 9:
        recommendations.append("⚠️ Dormir plus de 9 heures peut affecter la qualité du sommeil. Visez 7-8 heures.")
    
    # Recommandations basées sur l'activité physique
    if data['Physical Activity Level'] < 30:
        recommendations.append("🏃‍♂️ Augmentez votre activité physique à au moins 30 minutes par jour.")
    
    # Recommandations basées sur le stress
    if data['Stress Level'] > 6:
        recommendations.append("🧘‍♂️ Votre niveau de stress est élevé. Essayez la méditation ou le yoga.")
    
    # Recommandations basées sur les pas quotidiens
    if data['Daily Steps'] < 8000:
        recommendations.append("👣 Essayez d'atteindre au moins 8000 pas par jour pour améliorer votre sommeil.")
    
    # Recommandations basées sur la fréquence cardiaque
    if data['Heart Rate'] > 80:
        recommendations.append("❤️ Une fréquence cardiaque élevée peut affecter le sommeil. Pratiquez des exercices de respiration.")
    
    return recommendations

@app.post("/predict")
def predict_sleep_quality(sleep_data: SleepData):
    try:
        data = sleep_data.data
        
        # Préparer les features pour le modèle
        input_data = []
        for feature in features:
            if feature == 'Gender_num':
                input_data.append(1 if data['Gender'] == 'Male' else 0)
            elif feature == 'Blood_Pressure_num':
                input_data.append(1 if data['Blood Pressure'] == 'High' else 0)
            else:
                input_data.append(data[feature])
        
        # Standardiser les données
        input_scaled = scaler.transform([input_data])
        
        # Prédire la qualité du sommeil
        sleep_quality = float(model.predict(input_scaled)[0])
        
        # Obtenir les recommandations personnalisées
        recommendations = get_sleep_recommendations(data, sleep_quality)
        
        return {
            "sleep_quality": sleep_quality,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    try:
        with open('../models/saved_models/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        return {
            "model_name": metrics['model_name'],
            "performance": {
                "cv_score": metrics['cv_score'],
                "test_rmse": metrics['test_rmse'],
                "test_r2": metrics['test_r2']
            },
            "features_used": metrics['features']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Impossible de charger les informations du modèle")
