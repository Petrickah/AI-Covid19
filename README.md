# Covid-19 Predictions using Machine Learning and AI

### This project proposes a comparison between three artificial neural networks, the Data Science model ARIMA and Facebook's Prophet prediction model
</br>

Using Python, Tensorflow and Keras I've come with an artificial neural network which could predict the outcome of the current Covid-19 pandemic.

My strategy propose using the moving window to extract an interval of data from time series to train an ANN model. Data was normalized to 1000 people values using the difference between current day and past day.

Autoregressive Moving Avarage model uses these intervals to calculate an weekly avarage of the number of cases. By training the model using these values you could predict the next outcome for a maximum of 7 days. (If trained using weekly data)

Using a neural network on unaltered values it can descover the trend of the pandemic but not with exactity. Training more time, using two layers, it can predict with exactity. Using more layers it gets overwhelmed and overfits.
