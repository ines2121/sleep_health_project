# 😴 Analyse de la Santé du Sommeil

Cette application analyse la qualité du sommeil et le mode de vie pour fournir des recommandations personnalisées pour améliorer votre sommeil.

## 🌟 Fonctionnalités

- 📊 Analyse complète de la qualité du sommeil
- 💡 Recommandations personnalisées
- 📈 Visualisations interactives
- 👥 Analyse par profession
- 🔄 Corrélations entre les facteurs de sommeil

## 🛠️ Installation

1. Clonez le repository :
```bash
git clone https://github.com/ines2121/sleep_health_project.git
cd sleep_health_project
```

2. Lancez l'application avec Docker :
```bash
docker-compose up --build
```

## 🚀 Utilisation

1. Ouvrez votre navigateur et accédez à :
   - Sur l'ordinateur hébergeant l'application :
     - Interface utilisateur : http://localhost:8501
     - API : http://localhost:8002
   
   - Depuis un autre ordinateur sur le même réseau :
     - Interface utilisateur : http://<ADRESSE_IP>:8501
     - API : http://<ADRESSE_IP>:8002
     
   Note : Remplacez <ADRESSE_IP> par l'adresse IP de l'ordinateur qui héberge l'application.
   Pour trouver l'adresse IP sur macOS/Linux : ouvrez un terminal et tapez `ifconfig` ou `ip addr`.
   Pour Windows : ouvrez une invite de commande et tapez `ipconfig`.

2. Entrez vos données :
   - Âge
   - Genre
   - Profession
   - Durée du sommeil
   - Niveau d'activité physique
   - Niveau de stress
   - Pas quotidiens
   - etc.

3. Recevez une analyse détaillée et des recommandations personnalisées

## 📊 Données

Le jeu de données comprend les informations suivantes :
- Genre
- Âge
- Profession
- Durée du sommeil
- Qualité du sommeil
- Niveau d'activité physique
- Niveau de stress
- Catégorie IMC
- Pression artérielle
- Fréquence cardiaque
- Pas quotidiens
- Troubles du sommeil

## 👩‍💻 Développé par

Ines Hammouch
