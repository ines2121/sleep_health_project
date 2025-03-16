import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

def prepare_data(data):
    """Prépare les données pour l'entraînement du modèle"""
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
    
    return X, y, features

def train_model(X, y, model_type='random_forest', param_grid=None):
    """Entraîne un modèle avec GridSearchCV"""
    if param_grid is None:
        param_grid = get_default_param_grid(model_type)
    
    # Initialiser le modèle
    model = get_model_instance(model_type)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    # Entraînement
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_

def get_model_instance(model_type):
    """Retourne une instance du modèle spécifié"""
    models = {
        'random_forest': RandomForestRegressor(random_state=42),
        'gradient_boosting': GradientBoostingRegressor(random_state=42)
    }
    return models.get(model_type)

def get_default_param_grid(model_type):
    """Retourne la grille de paramètres par défaut pour chaque type de modèle"""
    param_grids = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
    }
    return param_grids.get(model_type)

def evaluate_model(model, X, y):
    """Évalue le modèle et retourne les métriques"""
    y_pred = model.predict(X)
    return {
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'r2': r2_score(y, y_pred)
    }

def save_model(model, metrics, model_type, features):
    """Sauvegarde le modèle et ses métriques"""
    # Créer le dossier saved_models s'il n'existe pas
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    # Sauvegarder le modèle
    model_path = os.path.join(save_dir, f'best_{model_type}_model.joblib')
    joblib.dump(model, model_path)
    
    # Sauvegarder la liste des features
    features_path = os.path.join(save_dir, 'feature_list.txt')
    with open(features_path, 'w') as f:
        f.write('\n'.join(features))
    
    # Sauvegarder les métriques
    metrics_path = os.path.join(save_dir, f'{model_type}_model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    """Fonction principale d'entraînement"""
    # Charger les données
    data = pd.read_csv("../data/Sleep_health_and_lifestyle_dataset.csv")
    
    # Préparer les données
    X, y, features = prepare_data(data)
    
    # Split les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardiser les features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir en DataFrame pour conserver les noms des colonnes
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Entraîner les modèles
    models_to_train = ['random_forest', 'gradient_boosting']
    results = {}
    
    for model_type in models_to_train:
        print(f"\nEntraînement du modèle {model_type}...")
        model, best_params = train_model(X_train_scaled, y_train, model_type)
        
        # Évaluation
        train_metrics = evaluate_model(model, X_train_scaled, y_train)
        test_metrics = evaluate_model(model, X_test_scaled, y_test)
        
        results[model_type] = {
            'model': model,
            'best_params': best_params,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
    
    # Sélectionner le meilleur modèle
    best_model_name = max(results, key=lambda k: results[k]['test_metrics']['r2'])
    best_model_results = results[best_model_name]
    
    # Sauvegarder le meilleur modèle et ses métriques
    metrics = {
        'model_name': best_model_name,
        'best_parameters': best_model_results['best_params'],
        'train_metrics': best_model_results['train_metrics'],
        'test_metrics': best_model_results['test_metrics'],
        'features': features
    }
    
    save_model(
        best_model_results['model'],
        metrics,
        best_model_name,
        features
    )
    
    # Afficher les résultats
    print("\nRésultats de l'évaluation des modèles :")
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"Meilleurs paramètres : {result['best_params']}")
        print(f"Métriques d'entraînement :")
        print(f"  - RMSE : {result['train_metrics']['rmse']:.3f}")
        print(f"  - R² : {result['train_metrics']['r2']:.3f}")
        print(f"Métriques de test :")
        print(f"  - RMSE : {result['test_metrics']['rmse']:.3f}")
        print(f"  - R² : {result['test_metrics']['r2']:.3f}")
    
    print(f"\nMeilleur modèle : {best_model_name}")

if __name__ == "__main__":
    main()
