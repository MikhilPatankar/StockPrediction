# Stock Price Predictor
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) Repository: [https://github.com/MikhilPatankar/StockPrediction](https://github.com/MikhilPatankar/StockPrediction)

## Overview

This project is a web application built with Flask that predicts future stock prices using the Prophet forecasting model by Meta. Users can input a stock ticker symbol (e.g., AAPL, GOOG, WEGE3.SA) and a future date to get a predicted adjusted closing price for that day, along with a visualization of the forecast and its components.

## Features

* **Stock Data Fetching:** Retrieves historical daily stock data using the `yahooquery` library.
* **Time Series Forecasting:** Utilizes the Prophet library to model trends, weekly and yearly seasonality, and forecast future prices.
* **Specific Date Prediction:** Allows users to select a specific future date for prediction.
* **Web Interface:** Provides a simple web UI built with Flask to input parameters and view results.
* **Visualization:** Displays plots of the historical data, forecast, and forecast components (trend, seasonality) using Matplotlib, embedded within the web page.

## Technologies Used

* **Backend:** Python, Flask
* **Forecasting:** Prophet (fbprophet)
* **Data Retrieval:** yahooquery
* **Data Handling:** pandas
* **Plotting:** Matplotlib
* **Frontend:** HTML, CSS (embedded in Flask templates/strings)

## Deploy on Heroku

### [![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://www.heroku.com/deploy?template=https://github.com/MikhilPatankar/StockPrediction)


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MikhilPatankar/StockPrediction.git
    cd StockPrediction
    ```

2.  **Create and activate a virtual environment (Recommended):**
    * **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    * **Prophet Installation Note:** Installing Prophet can sometimes be tricky, especially on Windows, as it requires a C++ compiler. If you encounter issues, refer to the official Prophet installation guide: [https://facebook.github.io/prophet/docs/installation.html](https://facebook.github.io/prophet/docs/installation.html). Using Conda (as mentioned in the guide) might be easier for managing Prophet's dependencies.
    * **Install using pip:**
        ```bash
        pip install -r requirements.txt
        ```
        *(If you had issues with `lxml` previously, ensure it's installed correctly, potentially using a pre-compiled wheel as discussed before running this command).*

## Usage
1.  **Run the Flask application:**
    ```bash
    python app.py
    ```

2.  **Access the web interface:**
    Open your web browser and navigate to the address provided by Flask (usually `http://127.0.0.1:5000/` or `http://0.0.0.0:5000/` if run with `host='0.0.0.0'`).

3.  **Enter Parameters:**
    * Input a valid stock ticker symbol recognized by Yahoo Finance (e.g., `AAPL`, `MSFT`, `GOOG`, `WEGE3.SA`).
    * Select the future date for which you want the prediction.

4.  **Submit:** Click the "Get Forecast" button.

5.  **View Results:** The application will display the predicted price for the selected date, along with the forecast plot and the components plot.

## Future Improvements

* Add support for different forecast horizons (e.g., predict next week/month).
* Incorporate confidence intervals visually in the prediction display.
* Allow selection of different forecasting models.
* Improve UI/UX design.
* Add more robust error handling for invalid tickers or data fetching issues.
* Explore using intraday data for time-specific predictions (requires significant changes and potentially different data sources/models).
* Deploy the application to a cloud platform (e.g., Heroku, PythonAnywhere, Google Cloud Run).

## License

This project is licensed under the MIT License.
