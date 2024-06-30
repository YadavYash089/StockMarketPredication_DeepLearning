import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st 


import yfinance as yf

start = '2022-1-1'
end = '2024-05-03'

st.title('Stock trend prediction')

user_input = st.text_input('Enter stock picker','ICICIBANK.ns')
# Fetch Apple's stock data from Yahoo Finance
df = yf.download(user_input, start=start, end=end)

# Print the first few rows of the dataframe


#describing data

st.subheader('Data of previous few years')

st.write(df.describe())

#visualization
st.subheader('Closing Price vs Time Chart')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Close'])

# Display the plot in Streamlit
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100= df['Close'].rolling(100).mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Close'])
ax.plot(ma100)
# Display the plot in Streamlit
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 200MA')
ma100= df['Close'].rolling(100).mean()
ma200= df['Close'].rolling(200).mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Close'])
ax.plot(ma100)
ax.plot(ma200)

# Display the plot in Streamlit
st.pyplot(fig)

#Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)


from sklearn.preprocessing import MinMaxScaler

# Define the scaler
scaler = MinMaxScaler(feature_range=(0,1))

# Fit the scaler to the training data and transform it
scaled_training_data = scaler.fit_transform(data_training)


#load model
model = load_model('keras_model.h5')

#testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)



x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler=scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Original vs Predicted')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test, 'b', label = 'original price')
ax.plot(y_predicted, 'r',label = 'Predicted Price' )
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()
# Display the plot in Streamlit
st.pyplot(fig)
