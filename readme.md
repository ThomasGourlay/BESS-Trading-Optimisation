# Battery Dispatch Model

This project implements a linear optimisation model for battery energy storage system (BESS) dispatch, along with a simple backtesting framework.

## Contents

* `writeup.pdf`
  Contains the mathematical formulation of the model.

* `notebook.ipynb`
  Main analysis notebook. Includes:

  * model implementation
  * rolling optimisation backtest
  * visualisations

* `price_data.csv`
  Input price data used for simulation. From AEMO website.

## Overview

The model optimises charging and discharging decisions to maximise profit given electricity prices, using a rolling horizon approach.

In a backtest, the model was able to generate over 470k AUD in profit over the month of February 2026. This is unrealistic as the model has perfect foresight of prices, and does not consider the price-setting abilities of a BESS, however it can be viewed as an upper bound for plausible energy arbitrage profits. 

## Usage

Open the notebook and run the cells sequentially to reproduce results.

