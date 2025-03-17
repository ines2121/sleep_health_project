import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Sleep_health_and_lifestyle_dataset.csv')
    df = pd.read_csv(data_path)
    
    # Convertir les variables catégorielles
    df['Gender_num'] = (df['Gender'] == 'Male').astype(int)
    df['Blood_Pressure_num'] = (df['Blood Pressure'] == 'High').astype(int)
    
    # Séparer les features et la target
    X = df[['Age', 'Gender_num', 'Sleep Duration', 'Physical Activity Level', 
            'Stress Level', 'Heart Rate', 'Daily Steps', 'Blood_Pressure_num']]
    y = df['Quality of Sleep']
    
    # Sauvegarder la liste des features
    features = X.columns.tolist()
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'feature_list.txt'), 'w') as f:
        f.write('\n'.join(features))
    
    return X, y, features

def train_model(X, y):
    # Diviser les données en train et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Définir les paramètres pour la recherche par grille
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Créer et entraîner le modèle avec GridSearchCV
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=2)
    print("Début de l'entraînement avec GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nMeilleurs paramètres : {grid_search.best_params_}")
    
    # Obtenir le meilleur modèle
    best_model = grid_search.best_estimator_
    
    # Évaluer le modèle
    train_rmse = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
    train_r2 = r2_score(y_train, best_model.predict(X_train))
    test_r2 = r2_score(y_test, best_model.predict(X_test))
    
    print("\nMétriques d'évaluation:")
    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Train R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    
    return best_model, {
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'best_params': grid_search.best_params_
    }

def save_model(model, metrics):
    # Créer le répertoire des modèles s'il n'existe pas
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Sauvegarder le modèle
    model_path = os.path.join(models_dir, 'best_sleep_model.joblib')
    joblib.dump(model, model_path)
    
    # Sauvegarder les métriques
    metrics_path = os.path.join(models_dir, 'random_forest_model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nModèle et métriques sauvegardés dans {models_dir}")

if __name__ == "__main__":
    # Charger les données
    X, y, features = load_data()
    
    # Entraîner le modèle
    model, metrics = train_model(X, y)
    
    # Sauvegarder le modèle et les métriques
    save_model(model, metrics)
