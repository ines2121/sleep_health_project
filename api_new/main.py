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

try:
    # Charger les métadonnées du modèle
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    features = metadata['selected_features']
    model = joblib.load(os.path.join(MODEL_DIR, 'best_sleep_model.joblib'))
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
            'Sleep Duration': sleep_data.Sleep_Duration,
            'Physical Activity Level': sleep_data.Physical_Activity_Level,
            'Stress Level': sleep_data.Stress_Level,
            'Heart Rate': sleep_data.Heart_Rate,
            'Daily Steps': sleep_data.Daily_Steps,
            'Blood_Pressure_num': 1 if sleep_data.Blood_Pressure.lower() == 'high' else 0
        }
        
        # Extraire les features dans le bon ordre
        X = np.array([[data[f] for f in features]])
        
        # Faire la prédiction
        prediction = float(model.predict(X)[0])
        
        # Générer des recommandations
        recommendations = get_sleep_recommendations(data, prediction)
        
        return {
            "sleep_quality": prediction,
            "recommendations": recommendations["recommendations"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Vérifie que l'API fonctionne"""
    return {"status": "healthy"}

@app.get("/info")
async def model_info() -> Dict[str, Union[str, List[str], Dict[str, float]]]:
    """Retourne les informations sur le modèle"""
    return {
        "model_type": metadata['best_model'],
        "features": features,
        "performance": {
            "r2_score": metadata['final_test_results']['r2_score'],
            "rmse": metadata['final_test_results']['rmse']
        },
        "dataset_size": metadata['dataset_size'],
        "train_size": metadata['train_size'],
        "test_size": metadata['test_size']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
