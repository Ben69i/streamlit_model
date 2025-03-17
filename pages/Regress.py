import streamlit as st
import yfinance as yf
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import seaborn as sb
import numpy as np
import plotly.graph_objects as go
from warnings import filterwarnings

filterwarnings('ignore')
sb.set(style="darkgrid")

st.set_page_config(
    page_title="Regression",
)

if any(key not in st.session_state.keys() for key in ["preds","ticker","data","batch_size","tab3_seed"]):
    st.session_state.preds = None
    st.session_state.ticker = "BTC-USD"
    st.session_state.data = 1
    st.session_state.batch_size = 30
    st.session_state.tab3_seed = None


st.cache_data(show_spinner="Loading data ...")
def ticker():
    data = yf.Ticker(st.session_state.ticker).history(period="max")
    data = pd.DataFrame(data, columns=["Close", "Open", "Volume"])
    data["Mid"] = (data["Close"] + data["Open"]) / 2
    data = data.dropna()
    data = data[["Volume", "Mid"]]
    return data


# Map for coins with their emojis
options = {
    "ADA-USD": "🌱 ADA",
    "BTC-USD": "₿ BTC",
    "ETH-USD": "💎 ETH",
    "XRP-USD": "⚡ XRP",
    "LTC-USD": "🌕 LTC",
    "DOGE-USD": "🐕 DOGE",
    "DOT-USD": "🌐 DOT",
    "BCH-USD": "🍀 BCH",
    "SOL-USD": "🌞 SOL",
    "BNB-USD": "🪙 BNB",
    "UNI-USD": "🦄 UNI",
    "LINK-USD": "🔗 LINK",
    "AAVE-USD": "🏦 AAVE",
    "XLM-USD": "⭐ XLM",
    "MATIC-USD": "🔹 MATIC",
    "MANA-USD": "🕹️ MANA",
    "SHIB-USD": "🐕‍🦺 SHIB",
    "CAKE-USD": "🍰 CAKE",
    "AXS-USD": "🛡️ AXS",
    "AVAX-USD": "🔥 AVAX",
    "BUSD-USD": "💵 BUSD",
    "DAI-USD": "🏅 DAI",
    "USDT-USD": "💲 USDT",
    "USDC-USD": "💵 USDC",
}


st.cache_resource(show_spinner="Loading model ...")
def modeling():
    model_input = st.session_state.model
    data = st.session_state.data
    data = pd.DataFrame(data)
    data["x"] = None
    if model_input:
        if model_input == "Linear Regression":
            final_model = LinearRegression()
        elif model_input == "Support Vector Regression":
            final_model = SVR()
        elif model_input == "Decision Tree Regression":
            final_model = DecisionTreeRegressor()
        elif model_input == "KNeighbors Regression":
            final_model = KNeighborsRegressor()
        elif model_input == "Extra Trees Regression":
            final_model = ExtraTreesRegressor()
        elif model_input == "HistGradientBoosting Regression":
            final_model = HistGradientBoostingRegressor()
        elif model_input == "MLPRegressor":
            final_model = MLPRegressor()
        else:
            st.error("Unknow model or bad parameter!")
        whole = len(data)
        batch_size = st.session_state.batch_size
        x_values = []

        for i in range(batch_size, whole):
            y = data["Mid"].iloc[i - batch_size:i].to_numpy()
            x_values.append(y)  # Store as list

        data = data.iloc[batch_size:].copy()  # Remove first few NaN rows
        data["x"] = x_values  # Assign he collected sequences
        data = data.dropna()
        x_train, x_test, y_train, y_test = train_test_split(data["x"].tolist(), data["Mid"], shuffle=False,
                                                            test_size=0.1)
        if st.session_state.tab3_seed is not None:
            seed = st.session_state.tab3_seed
            final_model.random_state=seed
        final_model.fit(x_train, y_train)
        preds = final_model.predict(x_test)
        days_ahead = pd.date_range(start=data.index[-1], periods=st.session_state.days_ahead, freq="1D")
        last = x_test[-1].reshape(1, -1)
        days_ahead_prices = list()
        for i in range(st.session_state.days_ahead):
            p = final_model.predict(last)
            days_ahead_prices.append(p[0])
            p = np.array(p).reshape(1, 1)  # Ensure p is 2D (1 sample, 1 feature)
            last = np.append(last[:, 1:], p, axis=1)
        days_ahead_prices = pd.DataFrame(days_ahead_prices, index=days_ahead)
        preds = pd.DataFrame({"y": y_test, "preds": preds}, index=y_test.index)
        st.session_state.preds = preds
        st.session_state.days_ahead_prices = days_ahead_prices

        return preds, days_ahead_prices


tab1, tab2, tab3 = st.tabs(["Data", "Model", "Environment"])
with tab1:
    with st.form("my_form"):
        ticker_choice = st.selectbox("Select Ticker", list(options.values()))
        ticker_symbol = [key for key, value in options.items() if value == ticker_choice][0]
        st.session_state.ticker = ticker_symbol  # Set the ticker to session state
        if st.form_submit_button("Submit"):
            st.session_state.data = ticker()
            if st.session_state.data.empty:
                st.error("Ticker not found!")
            else:
                st.success("Data Loaded Successfully!")

with tab2:
    with st.form("model_form"):
        col1, col2 = st.columns(2)
        col1.selectbox("Select Model", ["Linear Regression", "Support Vector Regression", "Decision Tree Regression",
                                        "KNeighbors Regression","Extra Trees Regression",
                                        "HistGradientBoosting Regression", "MLPRegressor"],
                       key="model")
        col2.number_input("Batch size", min_value=5, max_value=60, value=30, step=1, key="batch_size",help="more stable predictions with higher batch size")
        col2.number_input("Days ahead", min_value=5, max_value=150, value=30, step=1, key="days_ahead",help="The longer periods less accurate predictions")
        if st.form_submit_button("Submit", on_click=modeling):
            st.session_state.preds, st.session_state.days_ahead_prices = modeling()
    if st.session_state.preds is not None:
        with st.spinner("Plotting..."):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=st.session_state.data.index,y=st.session_state.data["Mid"], mode="lines",
                                     name=f"{st.session_state.ticker}"))
            fig.add_trace(
                go.Scatter(x=st.session_state.preds.index,y=st.session_state.preds.preds, mode="lines", name="Validation"))

            fig.add_trace(
                go.Scatter(x=st.session_state.days_ahead_prices.index,y=st.session_state.days_ahead_prices[0], mode="lines",
                           name="Future Predictions", line=dict(color='cyan', width=3)))
            fig.update_layout(
                title=f"Data for {st.session_state.get('ticker', 'Unknown')}",
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_white",
                showlegend=True,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
            st.table(st.session_state.days_ahead_prices)

st.sidebar.button("clear cache",on_click=lambda:st.cache_data.clear())

with tab3:
    with st.form("env_form"):
        st.header("Environment",divider="grey")
        col1,col2=st.columns(2)
        st.multiselect("Select the models",
                       ["Linear Regression", "Support Vector Regression", "Decision Tree Regression",
                        "KNeighbors Regression","Extra Trees Regression","HistGradientBoosting Regression", "MLPRegressor"],
                       default=["Linear Regression"], key="env")
        if col2.checkbox("Use Seed?"):
            col1.number_input("Seed",min_value=1,value=42,key="tab3_seed")
        st.form_submit_button("Submit")

    if st.session_state.env:
        df=pd.DataFrame()
        for model in st.session_state.env:
            del st.session_state.model
            st.session_state.model=model
            st.session_state.preds, st.session_state.days_ahead_prices = modeling()
            df[model]=st.session_state.days_ahead_prices
        if df is not None:
            ex=st.expander("Prediction")
            ex.table(df)
            fig=go.Figure()
            for i in df.columns:
                fig.add_trace(go.Scatter(x=df.index,y=df[i],mode="lines",name=i))
            st.plotly_chart(fig)