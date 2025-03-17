from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict, List, Union
import os
import json

app = FastAPI()

# Obtenir le chemin absolu du répertoire des modèles
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved_models')

# Charger le modèle
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'best_sleep_model.joblib'))
    
    # Charger la liste des features
    with open(os.path.join(MODEL_DIR, 'feature_list.txt'), 'r') as f:
        features = f.read().splitlines()
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {str(e)}")
    raise

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
        X = np.zeros((1, len(features)))
        for i, feature in enumerate(features):
            if feature == 'Gender_num':
                X[0, i] = 1 if data['Gender'] == 'Male' else 0
            elif feature == 'Blood_Pressure_num':
                X[0, i] = 1 if data['Blood Pressure'] == 'High' else 0
            else:
                X[0, i] = data[feature]
        
        # Prédire la qualité du sommeil
        sleep_quality = float(model.predict(X)[0])
        
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
        metrics_path = os.path.join(MODEL_DIR, 'random_forest_model_metrics.json')
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return {
            "model_type": "Random Forest",
            "performance": metrics,
            "features": features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Impossible de charger les informations du modèle : {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
