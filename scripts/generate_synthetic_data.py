import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Définir un seed pour la reproductibilité
np.random.seed(42)

# Nombre d'échantillons à générer
n_samples = 10000

# Générer des données avec des relations réalistes
data = {
    'Age': np.random.normal(35, 10, n_samples).clip(18, 80),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Sleep Duration': np.concatenate([
        np.random.normal(4.5, 0.5, n_samples//5),  # 20% mauvais dormeurs
        np.random.normal(7.5, 0.5, n_samples//5),  # 20% bons dormeurs
        np.random.normal(6, 1, 3*n_samples//5)     # 60% dormeurs moyens
    ]),
    'Physical Activity Level': np.concatenate([
        np.random.normal(15, 5, n_samples//4),     # 25% sédentaires
        np.random.normal(45, 10, n_samples//2),    # 50% modérément actifs
        np.random.normal(75, 15, n_samples//4)     # 25% très actifs
    ]),
    'Stress Level': np.concatenate([
        np.random.normal(2, 1, n_samples//4),      # 25% peu stressés
        np.random.normal(5, 1, n_samples//2),      # 50% stress moyen
        np.random.normal(8, 1, n_samples//4)       # 25% très stressés
    ]),
    'Heart Rate': np.random.normal(75, 10, n_samples).clip(60, 120),
    'Daily Steps': np.concatenate([
        np.random.normal(4000, 1000, n_samples//4),    # 25% sédentaires
        np.random.normal(8000, 2000, n_samples//2),    # 50% modérément actifs
        np.random.normal(12000, 2000, n_samples//4)    # 25% très actifs
    ]),
    'Blood Pressure': np.random.choice(['Normal', 'High'], n_samples, p=[0.7, 0.3])
}

# Nettoyer et clipper les données
data['Sleep Duration'] = np.clip(data['Sleep Duration'], 4, 10)
data['Physical Activity Level'] = np.clip(data['Physical Activity Level'], 0, 120)
data['Stress Level'] = np.clip(data['Stress Level'], 1, 10)
data['Daily Steps'] = np.clip(data['Daily Steps'], 2000, 25000)

df = pd.DataFrame(data)

def calculate_sleep_quality(row):
    # Commencer avec une qualité de base de 4 (en dessous de la moyenne)
    quality = 4.0
    
    # Impact de la durée du sommeil (optimal entre 7-8 heures)
    sleep_duration = row['Sleep Duration']
    if sleep_duration < 6:
        quality += (sleep_duration - 4) * 1.2  # Pénalité plus forte pour manque de sommeil
    elif sleep_duration > 9:
        quality += (10 - sleep_duration) * 1.0  # Pénalité pour trop de sommeil
    else:
        quality += 2.4 + (1.0 - abs(sleep_duration - 7.5) * 0.5)  # Max +3.4 points à 7.5h
    
    # Impact du stress (linéaire)
    quality += (1.0 - row['Stress Level'] / 10.0) * 1.5  # +1.5 points à stress=1
    
    # Impact de l'activité physique (linéaire jusqu'à 60 min)
    activity = min(row['Physical Activity Level'], 60)
    quality += activity / 60.0 * 1.2  # +1.2 points max
    
    # Impact des pas quotidiens (linéaire)
    steps = row['Daily Steps']
    if steps < 5000:
        quality += (steps - 2000) / 3000.0 * 0.6  # +0.6 point max
    else:
        quality += 0.6 + (min(steps, 10000) - 5000) / 5000.0 * 0.6  # +1.2 points max
    
    # Impact de la fréquence cardiaque (optimal à 70)
    hr_diff = abs(row['Heart Rate'] - 70)
    quality += (1.0 - hr_diff / 50.0) * 0.6  # +0.6 point max
    
    # Impact de la pression artérielle
    if row['Blood Pressure'] == 'High':
        quality -= 1.0  # Pénalité plus forte
    
    # Ajouter un peu de bruit aléatoire (±0.1 point maximum)
    quality += np.random.normal(0, 0.025)
    
    # S'assurer que la qualité reste dans les limites raisonnables
    return np.clip(quality, 1, 10)

# Calculer la qualité du sommeil
df['Quality of Sleep'] = df.apply(calculate_sleep_quality, axis=1)

# Convertir les variables catégorielles
df['Gender_num'] = (df['Gender'] == 'Male').astype(int)
df['Blood_Pressure_num'] = (df['Blood Pressure'] == 'High').astype(int)

# Sauvegarder les données
output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sleep_data_synthetic.csv')
df.to_csv(output_file, index=False)
print(f"Données générées et sauvegardées avec {n_samples} échantillons dans {output_file}")

# Afficher quelques statistiques sur les données générées
print("\nStatistiques sur la qualité du sommeil :")
print(df['Quality of Sleep'].describe())

# Afficher quelques exemples de cas extrêmes
print("\nExemples de cas avec mauvaise qualité de sommeil :")
print(df.nsmallest(3, 'Quality of Sleep')[['Sleep Duration', 'Stress Level', 'Physical Activity Level', 'Daily Steps', 'Quality of Sleep']])

print("\nExemples de cas avec bonne qualité de sommeil :")
print(df.nlargest(3, 'Quality of Sleep')[['Sleep Duration', 'Stress Level', 'Physical Activity Level', 'Daily Steps', 'Quality of Sleep']])
