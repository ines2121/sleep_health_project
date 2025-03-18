import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

# Fixer la graine aléatoire pour la reproductibilité
np.random.seed(42)

# Charger les données
data = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')

# Préparer les données
data['Gender_num'] = (data['Gender'] == 'Male').astype(int)
data['Blood_Pressure_num'] = (data['Blood Pressure'] == 'High').astype(int)

# Préparer les features
features = ['Age', 'Gender_num', 'Sleep Duration', 'Physical Activity Level', 
           'Stress Level', 'Heart Rate', 'Daily Steps', 'Blood_Pressure_num']
X = data[features].copy()
y = data['Quality of Sleep'].copy()

# Calculer les plages de valeurs pour chaque feature
feature_ranges = {}
for feature in features:
    if feature in ['Gender_num', 'Blood_Pressure_num']:
        feature_ranges[feature] = {'type': 'binary'}
    else:
        feature_ranges[feature] = {
            'type': 'numeric',
            'min': float(X[feature].min()),
            'max': float(X[feature].max()),
            'mean': float(X[feature].mean()),
            'std': float(X[feature].std())
        }

# Standardiser les features numériques (pas les variables binaires)
numeric_features = ['Age', 'Sleep Duration', 'Physical Activity Level', 
                   'Stress Level', 'Heart Rate', 'Daily Steps']
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Créer le modèle SVR avec les meilleurs paramètres
svr = SVR(kernel='rbf', C=1.0, epsilon=0.2, gamma='auto')

# Configurer la validation croisée avec graine fixe
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Calculer les scores de validation croisée
cv_scores = cross_val_score(svr, X, y, cv=cv)
cv_rmse = np.sqrt(-cross_val_score(svr, X, y, cv=cv, scoring='neg_mean_squared_error'))

# Entraîner le modèle final sur toutes les données
svr.fit(X, y)

# Sauvegarder le modèle et le scaler
model_dir = 'models/comparison_results'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(svr, os.path.join(model_dir, 'best_model.joblib'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))

# Sauvegarder les métadonnées
metadata = {
    'model_type': 'SVR',
    'selected_features': features,
    'numeric_features': numeric_features,
    'feature_ranges': feature_ranges,
    'hyperparameters': {
        'kernel': 'rbf',
        'C': 1.0,
        'epsilon': 0.2,
        'gamma': 'auto'
    },
    'cv_results': {
        'r2_scores': cv_scores.tolist(),
        'r2_mean': float(cv_scores.mean()),
        'r2_std': float(cv_scores.std()),
        'rmse_scores': cv_rmse.tolist(),
        'rmse_mean': float(cv_rmse.mean()),
        'rmse_std': float(cv_rmse.std())
    },
    'timestamp': '2025-03-17 23:20:10'
}

with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)
