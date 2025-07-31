import streamlit as st    
import numpy as np
import pandas as pd

st.title("hello tanmay")


# display as simple text

st.write("this is a simple text")


# create a dataframe

df = pd.DataFrame({
    "first column":[1,2,3,4],
    "second column":[10,20,30,40]
})


#display the dataframe

st.write("this is the dataframe")
st.write(df)




#create a line chart

chart_data = pd.DataFrame(
    np.random.randn(20,3), columns = ["a","b","c"]
)

st.line_chart(chart_data)