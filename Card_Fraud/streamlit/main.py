import streamlit as st
import pandas as pd

from src import support as sp

user_options = sp.user_input_features()

pred, prob = sp.prediction(user_options, sp.best_model)

prob_df = pd.DataFrame(prob)

if pred == 1:

    st.write(f"La prediccion es: Fraude")
    st.write(f"La probabilidad es: {round(prob[0][1], 2)}")
else:
    st.write(f"La prediccion es: No Fraude")

    st.write(f"La probabilidad es: {round(prob[0][0], 2)}")
    #st.dataframe(prob_df)