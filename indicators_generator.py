import pandas as pd
import numpy as np

# download raw data
data = pd.read_csv("data.csv")
df = data.copy()
df= df.iloc[::5, :]

# calculate spread and mid price
df["spread"] = [float(ask[1: -1].split(", ")[0]) - float(bid[1: -1].split(", ")[0]) for bid, ask in zip(df["bid"], df["ask"])]
df["mid price"] = [(float(ask[1: -1].split(", ")[0]) + float(bid[1: -1].split(", ")[0]))/2 for bid, ask in zip(df["bid"], df["ask"])]

# calculate total orderbook imbalance
total_bid_qty = []
for qty in df["bid_qty"]:
  qty_list = qty[1: -1].split(", ")
  temp_qty = 0
  for i in range(len(qty_list)):
    temp_qty += float(qty_list[i])
  total_bid_qty.append(temp_qty)

total_ask_qty = []
for qty in df["ask_qty"]:
  qty_list = qty[1: -1].split(", ")
  temp_qty = 0
  for i in range(len(qty_list)):
    temp_qty += float(qty_list[i])
  total_ask_qty.append(temp_qty)

df["bid total qty"] = total_bid_qty
df["ask total qty"] = total_ask_qty

df["order imbalance"] = (df["bid total qty"] - df["ask total qty"])/(df["bid total qty"] + df["ask total qty"])

# calculate changes in price and quantity
df["delta price"] = df["close"] - df["close"].shift(1)
df["delta volume"] = df["volume"] - df["volume"].shift(1)

# calculate RSI
window=14
df["gain"] = np.where(df["delta price"]>=0, df["delta price"], 0)
df["loss"] = np.where(df["delta price"]<0, -df["delta price"], 0)
df["avg gain"] = df["gain"].ewm(alpha=1/window, min_periods=window).mean()
df["avg loss"] = df["loss"].ewm(alpha=1/window, min_periods=window).mean()
df["RS"] = df["avg gain"]/df["avg loss"]
df["RSI"] = 100 - (100/(1+df["RS"]))

# calcalate OBV
df["daily_ret"] = df["close"].pct_change()
df['direction'] = np.where(df['daily_ret']>=0,1,-1)
df['direction'][0] = 0
df['vol_adj'] = df['volume'] * df['direction']
df['obv'] = df['vol_adj'].cumsum()

# calculate MACD
slow_window = 26
fast_window = 12
signal_window = 9
df["slow_ema"] = df["close"].ewm(span=slow_window, min_periods=slow_window).mean()
df["fast_ema"] = df["close"].ewm(span=fast_window, min_periods=fast_window).mean()
df["macd"] = df["fast_ema"] - df["slow_ema"]
df["signal"] = df["macd"].ewm(span=signal_window, min_periods=signal_window).mean()
df["macd signal"] = df["macd"] - df["signal"]

# calculate ATR
n = 20
df['H-L']=abs(df['high']-df['low'])
df['H-PC']=abs(df['high']-df['close'].shift(1))
df['L-PC']=abs(df['low']-df['close'].shift(1))
df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
df['ATR'] = df['TR'].rolling(n).mean()
df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
df['ATR%'] = df['ATR']/df['close']

# calculate ADX
window = 14
df["+change"] = df["high"] - df["high"].shift(1)
df["-change"] = df["low"].shift(1) - df["low"]
df["+DM"] = np.where((df["+change"] > df["-change"]) & (df["+change"] > 0), df["+change"], 0)
df["-DM"] = np.where((df["-change"] > df["+change"]) & (df["-change"] > 0), df["-change"], 0)
df["+DI14"] = 100 * (df["+DM"].ewm(com=window, min_periods=window).mean() / df["ATR"])
df["-DI14"] = 100 * (df["-DM"].ewm(com=window, min_periods=window).mean() / df["ATR"])
df["DX"] = 100 * (abs(df["+DI14"] - df["-DI14"]) / (df["+DI14"] + df["-DI14"]))
df["ADX"] = df["DX"].ewm(com=window, min_periods=window).mean()

# calculate rate of return
df["return"] = df["close"].pct_change()
df.dropna(inplace=True)

# save indicators data
indicators = df[["order imbalance", "macd signal", "spread", "mid price", "delta price", "delta volume", "ADX", "obv","RSI", "open interest", "return", "ATR%"]]
indicators.to_csv("indicators.csv", index=False)