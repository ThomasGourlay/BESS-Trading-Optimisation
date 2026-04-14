# Battery Dispatch Model

Inspired by my mathematical optimisation (Operations Research) class, I realised that simple battery trading can be implemented as a linear program. This project implements a linear optimisation model for battery energy storage system (BESS) dispatch, along with a backtesting framework.

We also use a simple xgboost tree model to create a prediction for energy prices. We use some data engineering to allow the model to predict prices for multiple horizons, giving us essentially time series data. This model isn't the best at forecasting but I wanted to see whether the optimisation model would still profit with shaky forecasts, which it passes with flying colours.

## Contents

* `writeup.pdf`
  Contains the mathematical formulation of the LP optimisation model.

* `energy.ipynb`
  Main analysis notebook. Includes:

  * model implementation using OOP
  * training and using the forecaster
  * rolling optimisation backtest
  * visualisations of PnL in backtests

* `energypredictionmodel.py`
  Contains the data side of the project including:
  * wrangling data from csv files provided by AEMO
  * modifying data where necessary
  * functions to train and test the xgboost model


## Overview

The model optimises charging and discharging decisions to maximise profit given electricity prices, using a rolling horizon approach.

We compare a perfect forecast and a forecast using the xgboost model, and observe that, unsurprisingly, the perfect forecast is a lot better. 

## Usage

Open the notebook and run the cells sequentially to reproduce results.
