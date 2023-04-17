## Card Fraud and Clients Clustering Project

This project aims to predict possible credit card fraud and cluster clients based on their behavior. The project is divided into two parts: Card Fraud and Clients Clustering.

### Project Structure

The project structure is as follows:

- Card Fraud

  - `notebooks`
  - `data`
  
- Clients Clustering

  - `notebooks`
  - `data`

The `notebooks` folder contains Jupyter notebooks with the code used for the project, while the data folder contains CSV files used as input data for the machine learning models.

### Card Fraud

The Card Fraud part of the project is focused on developing a machine learning model that can predict whether a credit card transaction is likely to be fraudulent or not. The notebook card_fraud.ipynb contains the code for building and testing the machine learning model.

#### - Data

The data used for the Card Fraud model is stored in the data folder under card_fraud_data.csv. The CSV file contains transaction data, including features such as transaction amount, transaction type, and the country where the transaction took place.

#### - Model

The machine learning model used for predicting credit card fraud is based on a random forest classifier. The model achieves an accuracy of X% on the test set.

### Clients Clustering

The Clients Clustering part of the project is focused on clustering clients based on their behavior. The notebook clients_clustering.ipynb contains the code for clustering the clients.

#### - Data

The data used for the Clients Clustering model is stored in the data folder under clients_data.csv. The CSV file contains client data, including features such as age, income, and transaction history.

#### - Model

The machine learning model used for clustering clients is based on K-Means clustering. The model identifies X different client clusters based on their behavior.

### Conclusion

The project successfully develops two machine learning models for predicting possible credit card fraud and clustering clients based on their behavior. These models can be used to target specific types of clients and prevent credit card fraud.