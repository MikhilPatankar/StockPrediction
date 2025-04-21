from flask import Flask, request, jsonify, render_template
from prophet import Prophet
from yahooquery import Ticker
import pandas as pd
from datetime import datetime
import pandas_ta as ta
import logging
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



from verboselogs import VerboseLogger, VERBOSE
from coloredlogs import install as Cloginstall

#Initialize Logger
logger = VerboseLogger('APP')
Cloginstall(level=VERBOSE, fmt='[%(asctime)s] | [%(name)s] | %(levelname)-8s | %(message)s')

logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

app = Flask(__name__)

def get_stock_data(symbol):
    try:
        logger.info(f"Getting Historic Data for Stock: {symbol}")
        stock = Ticker(symbol)
        history = stock.history(period="48mo")
        if history.empty or isinstance(history, dict):
             raise ValueError(f"Could not retrieve data for symbol: {symbol}")

        history.reset_index(level=["symbol"], inplace=True)
        history['date'] = history.index
        history.set_index(pd.DatetimeIndex(history.index), inplace=True)
        data = history[['date','adjclose']].copy()
        data.ta.ema(close='adjclose', length=21, append=True)
        data.dropna(inplace=True)
        if data.empty:
             raise ValueError(f"Not enough historical data after processing for: {symbol}")
        
        logger.success(f"Retrived Stock Data Succesfully!")
        return data
    except Exception as e:
        logger.warning(f"Error getting stock data for {symbol}: {e}")
        raise

def prepare_for_prophet(data):
    df_train = data[['date','adjclose']].copy()
    df_train.columns = ['ds', 'y']
    return df_train

def train_and_forecast(df_train, periods=365):
    logger.info("Training Model")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df_train)
    future = model.make_future_dataframe(periods=periods)
    logger.success("Model Trained!")
    forecast = model.predict(future)
    logger.success("Forcast Predicted!")
    return model, forecast

def create_plot(model, forecast, symbol):
    try:
        logger.info("Creating Plot!")
        fig = model.plot(forecast)
        from prophet.plot import add_changepoints_to_plot
        ax = fig.gca()
        add_changepoints_to_plot(ax, model, forecast)
        ax.set_title(f'{symbol} Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Adjusted Close Price')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        logger.success("Plot Created!")
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        logger.warning(f"Error creating plot: {e}")
        plt.close(fig)
        return None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict')
def predict():
    symbol = request.args.get('symbol')
    if not symbol:
        return render_template("form.html", symbol="N/A", error="No stock symbol provided.")

    try:
        stock_data = get_stock_data(symbol)
        prophet_data = prepare_for_prophet(stock_data)

        model, forecast = train_and_forecast(prophet_data, periods=30)
        plot_url = create_plot(model, forecast, symbol)
        forecast_tail = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)

        return render_template("form.html", symbol=symbol, forecast_tail=forecast_tail, plot_url=plot_url, error=None)

    except Exception as e:
        logger.warning(f"Error during prediction for {symbol}: {e}")
        return render_template("form.html", symbol=symbol, error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
