import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy 

class DataPreparation:
    def __init__(self, csv_path):
        self.dataset_df = pd.read_csv(csv_path)
        self.dataset_df["Years"] = pd.to_datetime(self.dataset_df["Years"])
        self.dataset_df['Month'] = self.dataset_df['Years'].dt.month
        self.dataset_df = pd.get_dummies(self.dataset_df, columns=['Month'], drop_first=True)
        self.prepare_data()

    def prepare_data(self):
        self.dataset_df['index_mesure'] = np.arange(len(self.dataset_df))   

        dataset_train_df = self.dataset_df.iloc[:int(len(self.dataset_df)*0.75)]
        dataset_test_df = self.dataset_df.iloc[int(len(self.dataset_df)*0.75):]
        
        self.x_train = dataset_train_df.drop(['Sales', 'Years'], axis=1)
        self.y_train = dataset_train_df['Sales']
        
        self.x_test = dataset_test_df.drop(['Sales', 'Years'], axis=1)
        self.y_test = dataset_test_df['Sales']

    def show_graph(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.dataset_df["Years"], self.dataset_df["Sales"], "o:")
        plt.title("Sales Over Time")
        plt.xlabel("Year")
        plt.ylabel("Sales")
        plt.show()


from sklearn.linear_model import LinearRegression

class Additif:
    def __init__(self, data_preparation_object):
        self.data_preparation_object = data_preparation_object
        self.model = LinearRegression()

        self.model.fit(self.data_preparation_object.x_train, self.data_preparation_object.y_train)

        self.y_train_predicted = self.model.predict(self.data_preparation_object.x_train)

        self.y_test_predicted = self.model.predict(self.data_preparation_object.x_test)

        self.show_model_predictions()

    def show_model_predictions(self):

        train_error = numpy.mean(numpy.abs(self.y_train_predicted - self.data_preparation_object.y_train))
        test_error = numpy.mean(numpy.abs(self.y_test_predicted - self.data_preparation_object.y_test))
        print(f"Erreur moyenne absolue sur l'ensemble d'entraînement : {train_error:.2f}")
        print(f"Erreur moyenne absolue sur l'ensemble de test : {test_error:.2f}")

        plt.figure(figsize=(15, 6))
        
        train_data = self.data_preparation_object.dataset_df.iloc[:int(len(self.data_preparation_object.dataset_df)*0.75)]
        test_data = self.data_preparation_object.dataset_df.iloc[int(len(self.data_preparation_object.dataset_df)*0.75):]
        
        plt.plot(train_data['Years'], train_data['Sales'], color='blue', label='TimeSeries Data', marker='o', linestyle='--')
        
        plt.plot(test_data['Years'], test_data['Sales'], color='orange', label='True Future Data', marker='o', linestyle='--')
        
        plt.plot(train_data['Years'], self.y_train_predicted, color='turquoise', label='Fitted Additive Model', linestyle='-')
        
        plt.plot(test_data['Years'], self.y_test_predicted, color='red', label='Forecasted Additive Model Data', linestyle='-')
        
        plt.title("Modèle Additif: Ventes Réelles vs Prédictions")
        plt.xlabel("Année")
        plt.ylabel("Ventes")
        plt.legend()
        plt.show()





csv_path = "vente_maillots_de_bain.csv"
data_preparation_object = DataPreparation(csv_path)
additif_object = Additif(data_preparation_object)