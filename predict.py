import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

training_set = pd.read_csv("training_indicators.csv")
testing_set = pd.read_csv("testing_indicators.csv")
y_train = training_set[["return"]].values
y_test = testing_set[["return"]].values

X_no_scale_train = training_set[["order imbalance", "RSI"]].values
X_to_scale_train = training_set[["macd signal", "ADX", "obv", "mid price", "delta volume", "delta price", "open interest", "spread"]].values
X_no_scale_test = testing_set[["order imbalance", "RSI"]].values
X_to_scale_test = testing_set[["macd signal", "ADX", "obv", "mid price", "delta volume", "delta price", "open interest", "spread"]].values

sc = StandardScaler()
X_to_scale_train = sc.fit_transform(X_to_scale_train)
X_to_scale_test = sc.transform(X_to_scale_test)

X_train = np.hstack((X_no_scale_train, X_to_scale_train))
X_test = np.hstack((X_no_scale_test, X_to_scale_test))

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

data = pd.read_csv("testing_data.csv")
df = data.copy()
price_plt = df.iloc[::5, :].reset_index(drop=True)

position = ""
initial_price = 0
end_price = 0
pos_return = 0
total_ret = []

ATR = testing_set["ATR%"]

long_entries = []
short_entries = []
exits = []
fee = 0.0005

for i in range(len(y_pred)):
    threshold = 0.5 * y_test[: i].std()
    if position == "":
        if y_pred[i] > threshold:
            position = "long"
            initial_price = price_plt["close"][i]
            long_entries.append((i, initial_price))

        elif y_pred[i] < -threshold:
            position = "short"
            initial_price = price_plt["close"][i]
            short_entries.append((i, initial_price))

    elif position == "long":
        end_price = price_plt["close"][i]
        pos_return = (end_price * (1-fee) / initial_price * (1+fee)) - 1
        if (pos_return >= 0.5*ATR[i]) or (pos_return <= -0.25*ATR[i]):
            position = ""
            initial_price = 0
            total_ret.append(pos_return)
            exits.append((i, end_price))

    elif position == "short":
        end_price = price_plt["close"][i]
        pos_return = 1 - (end_price  * (1+fee)/ initial_price * (1-fee))
        if (pos_return >= 0.5*ATR[i]) or (pos_return <= -0.25*ATR[i]):
            position = ""
            initial_price = 0
            total_ret.append(pos_return)
            exits.append((i, end_price))


ret = 1
for r in total_ret:
    ret *= (1+r)
ret -= 1
print("return:", ret)

vol = np.std(total_ret)
print("standard deviation:", vol)

sharp_ratio = ((1+ret)**121)-1/(vol*np.sqrt(121))
print("sharp ratio:", sharp_ratio)

((1+ret)**121)-1

plt.figure(figsize=(14,7))
plt.plot(price_plt["close"], label="Price", color="blue")

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