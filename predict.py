import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv("reg_indicators.csv")
y = dataset.iloc[:, -1].values # rate of return

# for our indicators, except for orderbook imbalance, we need to preprocess our data.
X_no_scale = dataset[["order imbalance"]].values
X_to_scale = dataset[["macd signal", "ADX", "obv", "mid price", "delta volume", "delta price", "spread"]].values

# split data into training ang test data
split_index = int(len(y) * 0.8)

X_to_scale_train = X_to_scale[:split_index]
X_to_scale_test = X_to_scale[split_index:]
X_no_scale_train = X_no_scale[:split_index]
X_no_scale_test = X_no_scale[split_index:]
y_train = y[:split_index]
y_test  = y[split_index:]

# To preprocess our data, we normalize it
sc = StandardScaler()
X_to_scale_train = sc.fit_transform(X_to_scale_train)
X_to_scale_test = sc.transform(X_to_scale_test)

X_train = np.hstack((X_no_scale_train, X_to_scale_train))
X_test = np.hstack((X_no_scale_test, X_to_scale_test))

# train our regression machine learning model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict with our model
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

# Backtest
data = pd.read_csv("data.csv")
df = data.copy()
price_plt = df.iloc[::5, :].reset_index(drop=True)

position = ""
initial_price = 0
end_price = 0
pos_return = 0
total_ret = []

ATR = dataset["ATR%"]

long_entries = []
short_entries = []
exits = []

for i in range(len(y_pred)):
    threshold = 1.5 * y_test[: i].std()

    if position == "":
        if y_pred[i] > threshold:
            position = "long"
            initial_price = price_plt["close"][split_index+i]
            long_entries.append((split_index+i, initial_price))

        elif y_pred[i] < -threshold:
            position = "short"
            initial_price = price_plt["close"][split_index+i]
            short_entries.append((split_index+i, initial_price))

    elif position == "long":
        end_price = price_plt["close"][split_index+i]
        pos_return = (end_price / initial_price) - 1
        if (pos_return >= 2*ATR[split_index+i]) or (pos_return <= -1*ATR[split_index+i]) or (y_pred[i] < -threshold):
            position = ""
            initial_price = 0
            total_ret.append(pos_return)
            exits.append((split_index+i, end_price))

    elif position == "short":
        end_price = price_plt["close"][split_index+i]
        pos_return = 1 - (end_price / initial_price)
        if (pos_return >= 2*ATR[split_index+i]) or (pos_return <= -1*ATR[split_index+i]) or (y_pred[i] > threshold):
            position = ""
            initial_price = 0
            total_ret.append(pos_return)
            exits.append((split_index+i, end_price))

ret = 1
for r in total_ret:
    ret *= (1+r)
ret -= 1
print("總報酬:", ret)

# visualization
plt.figure(figsize=(14,7))
plt.plot(price_plt["close"][split_index:], label="Price", color="blue")

plt.scatter([x[0] for x in long_entries], [x[1] for x in long_entries],
            color="green", marker="^", s=80, label="Long Entry")

plt.scatter([x[0] for x in short_entries], [x[1] for x in short_entries],
            color="red", marker="v", s=80, label="Short Entry")

plt.scatter([x[0] for x in exits], [x[1] for x in exits],
            color="black", marker="x", s=80, label="Exit")

plt.legend()
plt.title("Trading Strategy Simulation")
plt.xlabel("Index")
plt.ylabel("Price")
plt.show()