import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error
from datetime import date
import pickle
import matplotlib.pyplot as plt

def get_stock_data(stock_ticker):
    today = str(date.today())
    twentythree_years_ago = str(int(today.split('-')[0]) - 23) + today[4:]

    stock_data = yf.download(stock_ticker, start=twentythree_years_ago, end=today)

    data = stock_data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler, stock_data

if __name__ == "__main__":
    stock_ticker = "TSLA"

    scaled_data, scaler, stock_data = get_stock_data(stock_ticker)

    train_size = int(0.8 * len(scaled_data))
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    X_train, y_train = train_data[:-1], train_data[1:, 0]
    X_test, y_test = test_data[:-1], test_data[1:, 0]

    with open('best_hyperparameters.pkl', 'rb') as f:
        best_params = pickle.load(f)

    lstm_units = best_params['lstm_units']
    dropout_rate = best_params['dropout_rate']

    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(1, 1), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    epochs = 100
    batch_size = 32
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Make predictions on test data
    predictions = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predictions)

    mse = mean_squared_error(y_test, predictions)  # Use unscaled predictions
    print("Mean Squared Error (MSE):", mse)

    yesterday_close = stock_data['Close'][-2]
    today_open = stock_data['Open'][-1]

    scaled_today_open = scaler.transform(np.array([[today_open]]))
    today_close_prediction = model.predict(scaled_today_open)
    today_close = scaler.inverse_transform(today_close_prediction)[0][0]

    print()
    print("Stock:", stock_ticker)
    print("Today's predicted closing price: $" + str(round(today_close, 2)))

    if today_close > yesterday_close:
        print("Action: Buy")
    elif today_close < yesterday_close:
        print("Action: Sell")
    else:
        print("Action: Hold")

    # Extract dates and actual prices from stock_data
    dates = stock_data.index
    actual_prices = stock_data['Close'].values

    # Plot actual stock prices
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual_prices, label='Actual Prices', color='blue')

    # Add the predicted prices to the plot
    plt.plot(dates[train_size+1:], predicted_prices, label='Predicted Prices', color='orange')

    plt.title('Actual vs. Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()