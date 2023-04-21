import streamlit as st
import pandas as pd

from src import support as sp

st.title("Are you the next victim of a credit card fraud?")

st.image("images/credit_card.jpeg")

st.subheader("Prediction model of credit card fraud")
st.markdown("One of the main causes of the external fraud that has to be manage in the day-to-day of a bank is the credit card fraud. This project is focused in the prediction of a possible credit card fraud between all the non-fraud movements.")
st.markdown("To create the model we have used a dataset with the following variables: distance from home, last transaction distance, average purchase price, if is an usual retailer, if chip was used, if pin number was used and if the purchase was online.")
st.markdown("The most valuables variables for the prediction model are the average purchase price, the distance from home and online order. The model achieves an accuracy of 0,99 on the test set. ")

user_options = sp.user_input_features()

pred, prob = sp.prediction(user_options, sp.best_model)

prob_df = pd.DataFrame(prob)

if pred == 1:

    st.write(f"_The prediction is:_ Fraud")
    st.write(f"_The probability:_ {round(prob[0][1], 2)}")
else:
    st.write(f"_The prediction is:_ Not Fraud")

    st.write(f"_The probability:_ {round(prob[0][0], 2)}")
    #st.dataframe(prob_df)