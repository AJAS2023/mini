import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
 
# Define selectable values
STOCKS = {
    'Google': 'GOOG',
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'GameStop': 'GME'
}
YEARS = [1, 2, 3, 4]

# Define functions
@st.cache
def load_data(ticker, start, today):
    data = yf.download(ticker, start, today)
    data = data.reset_index()
    data['Date'] = data['Date'].astype('datetime64[ns]')  # Convert 'Date' column to datetime data type
    data.set_index('Date', inplace=True)  # Set 'Date' column as the index
    return data

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Main app
def main():
    st.set_page_config(
        page_title = "StockWise: Intelligent Stock Forecasting System",
         page_icon = "📈",
         )
   
    st.title('StockWise: Intelligent Stock Forecasting System')

    # Select dataset for prediction
    selected_stock = st.selectbox('Select dataset for prediction', list(STOCKS.keys()), index=0, format_func=lambda x: STOCKS[x])

    # Years of prediction slider
    n_years = st.selectbox('Years of prediction:', YEARS, index=0)

    # Calculate period based on selected years
    period = n_years * 365

    # Load data
    data_load_state = st.text('Loading data...')
    data = load_data(STOCKS[selected_stock], START, TODAY)
    data_load_state.text('Loading data... done!')

    # Display raw data
    st.subheader('Raw data')
    st.write(data.tail())
    
    # Plot raw data
    plot_raw_data(data)

    # Predict forecast with Prophet
    df_train = data[['Close']]
    df_train = df_train.rename(columns={"Close": "y"})
    df_train = df_train.reset_index().rename(columns={"Date": "ds"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Display forecast data
    st.subheader('Forecast data')
    st.write(forecast.tail())

    # Plot forecast
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # Plot forecast components
    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

if __name__ == "__main__":
    # Define constants
    START = "2016-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    main()