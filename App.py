import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.write("App is running properly")

st.title("📈 Cryptocurrency Analytics Dashboard")

# -----------------------------------
# Load CSV
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_crypto_with_predictions (3).csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# -----------------------------------
# Sidebar Date Filter
# -----------------------------------
st.sidebar.header("Filter Data")

start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())

filtered_df = df[
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
]

# -----------------------------------
# Key Metrics
# -----------------------------------
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Latest Close Price", round(filtered_df["Close"].iloc[-1], 2))
col2.metric("Latest MA_7", round(filtered_df["MA_7"].iloc[-1], 2))
col3.metric("Latest Volatility", round(filtered_df["Volatility"].iloc[-1], 4))

# -----------------------------------
# Historical Price + Moving Average
# -----------------------------------
st.subheader("📊 Historical Price & Moving Averages")

price_fig = go.Figure()

price_fig.add_trace(go.Scatter(
    x=filtered_df["Date"],
    y=filtered_df["Close"],
    mode="lines",
    name="Actual Price"
))

price_fig.add_trace(go.Scatter(
    x=filtered_df["Date"],
    y=filtered_df["MA_7"],
    mode="lines",
    name="MA 7"
))

price_fig.add_trace(go.Scatter(
    x=filtered_df["Date"],
    y=filtered_df["MA_30"],
    mode="lines",
    name="MA 30"
))

price_fig.update_layout(template="plotly_dark")

st.plotly_chart(price_fig, use_container_width=True)

# -----------------------------------
# Prediction Comparison (Handles NaN Automatically)
# -----------------------------------
st.subheader("🔮 Model Prediction Comparison")

pred_fig = go.Figure()

# Actual Price
pred_fig.add_trace(go.Scatter(
    x=filtered_df["Date"],
    y=filtered_df["Close"],
    mode="lines",
    name="Actual Price"
))

# ARIMA Prediction (will show only where values exist)
if "ARIMA_Pred" in df.columns:
    pred_fig.add_trace(go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["ARIMA_Pred"],
        mode="lines",
        name="ARIMA Prediction"
    ))

# LSTM Prediction
if "LSTM_Pred" in df.columns:
    pred_fig.add_trace(go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["LSTM_Pred"],
        mode="lines",
        name="LSTM Prediction"
    ))

# Prophet Prediction (if exists)
if "Prophet_Pred" in df.columns:
    pred_fig.add_trace(go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Prophet_Pred"],
        mode="lines",
        name="Prophet Prediction"
    ))

pred_fig.update_layout(template="plotly_dark")

st.plotly_chart(pred_fig, use_container_width=True)

# -----------------------------------
# Volatility Chart
# -----------------------------------
st.subheader("📉 Volatility Trend")

vol_fig = go.Figure()

vol_fig.add_trace(go.Scatter(
    x=filtered_df["Date"],
    y=filtered_df["Volatility"],
    mode="lines",
    name="Volatility"
))

vol_fig.update_layout(template="plotly_dark")

st.plotly_chart(vol_fig, use_container_width=True)


st.subheader("📰 Sentiment Trend")

sent_fig = go.Figure()

sent_fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Sentiment_Score"],
    mode="lines",
    name="Sentiment Score"
))

sent_fig.update_layout(template="plotly_dark")

st.plotly_chart(sent_fig, use_container_width=True)



from sklearn.metrics import mean_squared_error
import numpy as np



valid_arima = df[["Close", "ARIMA_Pred"]].dropna()
arima_rmse = np.sqrt(mean_squared_error(
    valid_arima["Close"],
    valid_arima["ARIMA_Pred"]
))

st.metric("ARIMA RMSE", round(arima_rmse, 2))
valid_lstm = df[["Close", "LSTM_Pred"]].dropna()

lstm_rmse = np.sqrt(mean_squared_error(
    valid_lstm["Close"],
    valid_lstm["LSTM_Pred"]
))

st.metric("LSTM RMSE", round(lstm_rmse, 2))


# -----------------------------------
# Show Last 10 Rows
# -----------------------------------
st.subheader("📄 Data Preview")
st.dataframe(filtered_df.tail(10))