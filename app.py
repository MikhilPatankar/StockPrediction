import logging
import io, os
import base64
import matplotlib
import traceback
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template
from prophet import Prophet
from yahooquery import Ticker
from datetime import datetime, date
from verboselogs import VerboseLogger, VERBOSE
from coloredlogs import install as Cloginstall

matplotlib.use('Agg')

#Initialize Logger
logger = VerboseLogger('APP')
Cloginstall(level=VERBOSE, fmt='[%(asctime)s] | [%(name)s] | %(levelname)-8s | %(message)s')

logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

app = Flask(__name__)

PORT = os.environ.get('PORT', 8080)

def get_stock_data(symbol):
    try:
        stock = Ticker(symbol)
        history = stock.history(period="max", interval="1d")
        if history.empty or isinstance(history, dict):
             raise ValueError(f"Could not retrieve data for symbol: {symbol}")

        history.reset_index(inplace=True)

        date_col = None
        if 'date' in history.columns:
            date_col = 'date'
        elif 'index' in history.columns:
            date_col = 'index'
        else:
            potential_cols = [col for col in history.columns if 'date' in col.lower()]
            if potential_cols:
                date_col = potential_cols[0]
            else:
                 raise ValueError("Could not automatically identify the date column after reset_index().")

        logger.info(f"Identified date column: {date_col}")

        history[date_col] = pd.to_datetime(history[date_col])

        if date_col != 'date':
            history.rename(columns={date_col: 'date'}, inplace=True)


        if 'adjclose' not in history.columns:
             if 'close' in history.columns:
                 logger.error("Warning: 'adjclose' not found, using 'close' price instead.")
                 history.rename(columns={'close': 'adjclose'}, inplace=True)
             else:
                 raise ValueError("Neither 'adjclose' nor 'close' column found in historical data.")

        data = history[['date','adjclose']].copy()

        data.ta.ema(close='adjclose', length=21, append=True)

        data.dropna(subset=['adjclose'], inplace=True)
        if data.empty:
             raise ValueError(f"Not enough historical data after processing for: {symbol}")

        data.sort_values(by='date', inplace=True)
        return data
    except Exception as e:
        logger.warning(f"Error getting stock data for {symbol}: {e}")
        raise


def prepare_for_prophet(data):
    df_train = data[['date','adjclose']].copy()
    df_train.rename(columns={'date': 'ds', 'adjclose': 'y'}, inplace=True)
    df_train['ds'] = pd.to_datetime(df_train['ds'])
    return df_train

def train_and_forecast(df_train, future_date):
    last_historical_date = df_train['ds'].max().date()
    future_date_obj = future_date
    periods = (future_date_obj - last_historical_date).days

    if periods < 1:
        periods = 1

    logger.info(f"Last historical date: {last_historical_date}")
    logger.info(f"Future date requested: {future_date_obj}")
    logger.info(f"Forecasting {periods} periods ahead.")


    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df_train)

    future_df = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future_df)
    return model, forecast


def create_plot(model, forecast, symbol, target_date):
    try:
        fig = model.plot(forecast)
        from prophet.plot import add_changepoints_to_plot
        ax = fig.gca()
        add_changepoints_to_plot(ax, model, forecast)

        target_forecast = forecast[forecast['ds'].dt.date == target_date]
        if not target_forecast.empty:
            ax.plot(target_forecast['ds'], target_forecast['yhat'], 'ro', markersize=8, label=f'Prediction for {target_date}')
            ax.legend()

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
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        logger.warning(f"Error creating plot: {e}")
        try: plt.close(fig)
        except: pass
        return None

def create_components_plot(model, forecast):
    fig = None
    try:
        fig = model.plot_components(forecast)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"Error creating components plot: {e}")
        traceback.format_exc()
        if fig:
             try: plt.close(fig)
             except: pass
        return None
    

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict')
def predict():
    symbol = request.args.get('symbol')
    target_date_str = request.args.get('target_date')

    if not symbol:
        return render_template("form.html", symbol="N/A", target_date_str="N/A", error="No stock symbol provided.")
    if not target_date_str:
        return render_template("form.html", symbol=symbol, target_date_str="N/A", error="No target date provided.")

    try:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()

        if target_date <= date.today():
             logger.error(f"Warning: Target date {target_date_str} is today or in the past.")

        logger.info(f"Fetching data for {symbol}...")
        stock_data = get_stock_data(symbol)
        logger.info("Data fetched.")

        last_hist_date = stock_data['date'].max().date()
        if target_date <= last_hist_date:
             logger.error(f"Warning: Target date {target_date_str} is within historical data range.")

        logger.info("Preparing data for Prophet...")
        prophet_data = prepare_for_prophet(stock_data)
        logger.info("Data prepared.")

        logger.info(f"Training model and forecasting up to {target_date_str}...")
        model, forecast = train_and_forecast(prophet_data, target_date)
        logger.success("Forecast complete.")
        
        prediction_row = forecast[forecast['ds'].dt.date == target_date]

        specific_prediction = None
        if not prediction_row.empty:
            specific_prediction = prediction_row.iloc[0]
            logger.success(f"Prediction found for {target_date_str}: {specific_prediction['yhat']:.2f}")
        else:
            logger.info(f"No specific prediction found for {target_date_str} in the forecast output.")

        logger.info("Generating plot...")
        plot_url = create_plot(model, forecast, symbol, target_date)
        logger.success("Plot generated.")

        print("Generating components plot...")
        components_plot_url = create_components_plot(model, forecast)
        print("Components plot generated.")
        
        return render_template(
            "form.html",
            symbol=symbol,
            target_date_str=target_date_str,
            prediction=specific_prediction,
            plot_url=plot_url,
            components_plot_url=components_plot_url,
            error=None
        )

    except ValueError as ve:
        logger.warning(f"Value Error during prediction for {symbol} on {target_date_str}: {ve}")
        logger.warning(traceback.format_exc())
        return render_template("form.html", symbol=symbol, target_date_str=target_date_str, error=str(ve))
    except Exception as e:
        logger.warning(f"Unexpected Error during prediction for {symbol} on {target_date_str}: {e}")
        logger.warning(traceback.format_exc())
        return render_template("form.html", symbol=symbol, target_date_str=target_date_str, error=f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=PORT)
