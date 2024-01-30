from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

class DataPreparation:
    def __init__(self, csv_path):
        """
        Cette classe prend en entrée un chemin de fichier csv.
        Elle split le jeu de donnée en 2 bases:
        + une train 75 %
        + une test 25 %
        Ces 2 bases sont ensuite divisées en vecteurs x et y.
        Quatre arrays sont extraits: x_train, y_train, x_test, y_test.
        """
        self.dataset_df = pd.read_csv(csv_path)
        self.dataset_df["Years"] = pd.to_datetime(self.dataset_df["Years"])
        self.prepare_data()


    def prepare_data(self):
        # Extraction des caractéristiques basées sur le temps
        self.dataset_df['Year'] = self.dataset_df['Years'].dt.year
        self.dataset_df['Month'] = self.dataset_df['Years'].dt.month
        self.dataset_df['Quarter'] = self.dataset_df['Years'].dt.quarter

        # Séparation en features et target
        X = self.dataset_df[['Year', 'Month', 'Quarter']]  # Utilisation des caractéristiques temporelles
        y = self.dataset_df['Sales']

        # Split en données d'entraînement et de test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    def show_graph(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.dataset_df["Years"], self.dataset_df["Sales"], "o:")
        plt.title("Ventes de Maillots de Bain au Fil du Temps")
        plt.xlabel("Année")
        plt.ylabel("Ventes")
        plt.show()


file_path = 'vente_maillots_de_bain.csv'

# Création d'une instance de la classe pour préparation des données
data_preparation = DataPreparation(file_path)
import statsmodels.api as sm

# Supposons que 'data_preparation_object.y_train' et 'data_preparation_object.x_train' sont déjà définis et corrects
# Modélisation avec une tendance additive et une saisonnalité additive
decomposition = sm.tsa.seasonal_decompose(data_preparation.y_train, model='additive', period=12)

# Ajustement d'un modèle SARIMA
model = sm.tsa.statespace.SARIMAX(data_preparation.y_train,
                                  order=(1, 1, 1),
                                  seasonal_order=(1, 1, 1, 12),
                                  trend='c').fit()

# Prédictions
predictions = model.get_forecast(steps=24)  # Prévoir 24 périodes à l'avance
forecast = predictions.predicted_mean
conf_int = predictions.conf_int()

# Tracé des résultats, similaire à la première image
plt.figure(figsize=(15, 6))
plt.plot(data_preparation.y_train, label='Données d\'entraînement')
plt.plot(forecast, label='Prédictions du modèle additif')
plt.fill_between(forecast.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1], color='grey', alpha=.3)
plt.legend()
plt.show()
