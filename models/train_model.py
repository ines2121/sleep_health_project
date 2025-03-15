import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Charger les données
data = pd.read_csv("../data/Sleep_health_and_lifestyle_dataset.csv")

# Préparer les features
features = [
    'Age', 'Sleep Duration', 'Physical Activity Level',
    'Stress Level', 'Heart Rate', 'Daily Steps'
]

# Convertir Gender et Blood Pressure en variables numériques
data['Gender_num'] = (data['Gender'] == 'Male').astype(int)
data['Blood_Pressure_num'] = (data['Blood Pressure'] == 'High').astype(int)

features.extend(['Gender_num', 'Blood_Pressure_num'])

X = data[features]
y = data['Quality of Sleep']

# Split les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardiser les features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définir les modèles à tester
models = {
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Évaluer chaque modèle
results = {}
for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # Entraînement sur l'ensemble complet
    model.fit(X_train_scaled, y_train)
    
    # Prédictions sur le test set
    y_pred = model.predict(X_test_scaled)
    
    # Métriques
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'test_r2': r2_score(y_test, y_pred),
        'model': model
    }

# Sélectionner le meilleur modèle
best_model_name = max(results, key=lambda k: results[k]['test_r2'])
best_model = results[best_model_name]['model']

# Créer le dossier saved_models s'il n'existe pas
os.makedirs('../models/saved_models', exist_ok=True)

# Sauvegarder le modèle et le scaler
joblib.dump(best_model, '../models/saved_models/best_sleep_model.joblib')
joblib.dump(scaler, '../models/saved_models/scaler.joblib')

# Sauvegarder les features utilisées
with open('../models/saved_models/feature_list.txt', 'w') as f:
    f.write('\n'.join(features))

# Afficher les résultats
print("\nRésultats de l'évaluation des modèles :")
for name, metrics in results.items():
    print(f"\n{name.upper()}:")
    print(f"CV Score: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']*2:.3f})")
    print(f"Test RMSE: {metrics['test_rmse']:.3f}")
    print(f"Test R²: {metrics['test_r2']:.3f}")

print(f"\nMeilleur modèle: {best_model_name}")

# Sauvegarder les métriques du meilleur modèle
best_metrics = {
    'model_name': best_model_name,
    'cv_score': float(results[best_model_name]['cv_mean']),
    'test_rmse': float(results[best_model_name]['test_rmse']),
    'test_r2': float(results[best_model_name]['test_r2']),
    'features': features
}

import json
with open('../models/saved_models/model_metrics.json', 'w') as f:
    json.dump(best_metrics, f, indent=4)
