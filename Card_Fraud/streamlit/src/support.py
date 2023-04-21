import pickle
import pandas as pd
import streamlit as st

with open('../data/smote_random_forest.pkl', 'rb') as best_model:
    best_model = pickle.load(best_model)


def user_input_features():
    """
    Function to save the info that the user give in the app
    Args: 
        non receive parameters
    Returns:
        A dataframe with the characteristcs given by the user
    """
    distance_from_home = st.sidebar.slider('Distance from home', 0, 10633, 27) # the sidebar.slider magic function receive the max, min and default value in out sidebar
    distance_from_last_transaction = st.sidebar.slider('Distance from last transaction', 0.0001, 1185.5, 5.0)
    ratio_to_median_purchase_price = st.sidebar.slider('Average of purchase price', 0.004, 268.0, 1.82)
    repeat_retailer	= st.sidebar.slider('Repeat retailer', 0, 1, 0)
    used_chip = st.sidebar.slider('Used chip', 0, 1, 0)
    used_pin_number = st.sidebar.slider('Used pin number', 0, 1, 0)
    online_order = st.sidebar.slider('Online order', 0, 1, 0)
    data = {'distance_from_home': distance_from_home,
            'distance_from_last_transaction': distance_from_last_transaction,
            'ratio_to_median_purchase_price': ratio_to_median_purchase_price,
            'repeat_retailer': repeat_retailer,
            'used_chip': used_chip,
            'used_pin_number': used_pin_number,
            'online_order': online_order}

    return pd.DataFrame(data, index=[0])

def prediction(dataframe_user, model):

    pred = model.predict(dataframe_user)[0]
    
    prob = model.predict_proba(dataframe_user)
    
    return pred, prob
