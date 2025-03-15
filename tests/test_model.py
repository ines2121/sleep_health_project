import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.train_model import prepare_data, train_model, evaluate_model

class TestSleepModel(unittest.TestCase):
    def setUp(self):
        """Préparation des données de test"""
        # Créer un jeu de données plus grand pour la validation croisée
        np.random.seed(42)
        n_samples = 20  # Au moins 4 fois le nombre de plis pour la validation croisée
        
        self.test_data = pd.DataFrame({
            'Age': np.random.randint(25, 60, n_samples),
            'Sleep Duration': np.random.uniform(5, 9, n_samples),
            'Physical Activity Level': np.random.randint(30, 90, n_samples),
            'Stress Level': np.random.randint(3, 9, n_samples),
            'Heart Rate': np.random.randint(60, 100, n_samples),
            'Daily Steps': np.random.randint(5000, 15000, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Blood Pressure': np.random.choice(['Normal', 'High'], n_samples),
            'Quality of Sleep': np.random.uniform(4, 9, n_samples)
        })
        
    def test_prepare_data(self):
        """Test de la préparation des données"""
        X, y, features = prepare_data(self.test_data)
        
        # Vérifier la forme des données
        self.assertEqual(X.shape[0], len(self.test_data))  # Nombre d'échantillons
        self.assertEqual(X.shape[1], 8)  # 8 features après transformation
        self.assertEqual(y.shape[0], len(self.test_data))  # Nombre de labels
        
        # Vérifier la conversion des variables catégorielles
        self.assertIn('Gender_num', X.columns)
        self.assertIn('Blood_Pressure_num', X.columns)
        
        # Vérifier que les valeurs sont dans les bonnes plages
        self.assertTrue(X['Gender_num'].isin([0, 1]).all())
        self.assertTrue(X['Blood_Pressure_num'].isin([0, 1]).all())
        
    def test_train_model(self):
        """Test de l'entraînement du modèle"""
        X, y, features = prepare_data(self.test_data)
        
        # Test avec RandomForest
        rf_params = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        }
        model, best_params = train_model(X, y, 'random_forest', rf_params)
        
        # Vérifier que le modèle est bien entraîné
        self.assertIsInstance(model, RandomForestRegressor)
        self.assertIsInstance(best_params, dict)
        self.assertIn('n_estimators', best_params)
        self.assertIn('max_depth', best_params)
        
        # Test avec GradientBoosting
        gb_params = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1]
        }
        model, best_params = train_model(X, y, 'gradient_boosting', gb_params)
        self.assertIsInstance(model, GradientBoostingRegressor)
        self.assertIn('n_estimators', best_params)
        self.assertIn('learning_rate', best_params)
        
    def test_evaluate_model(self):
        """Test de l'évaluation du modèle"""
        X, y, features = prepare_data(self.test_data)
        
        # Entraîner un modèle simple pour le test
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Évaluer le modèle
        metrics = evaluate_model(model, X, y)
        
        # Vérifier les métriques
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        self.assertGreaterEqual(metrics['r2'], 0)  # R² doit être positif
        self.assertGreaterEqual(metrics['rmse'], 0)  # RMSE doit être positif
        self.assertLessEqual(metrics['r2'], 1)  # R² doit être inférieur ou égal à 1

if __name__ == '__main__':
    unittest.main()
