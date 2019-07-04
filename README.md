*Aim:*<br>
This notebook contains data cleaning, data exploration and modelling of Vilnius' public transportation passenger flows in 2016. The data is open publicly at: https://github.com/vilnius/keleiviu-srautai

The aim of the model is to predict how full the selected bus will be at a certain place/time, for trip planning purposes. <br>

*Predictors:*<br>
By default, the data contains sensor data of bus occupancy in many different stops/times/lines.

We augmented the data by adding:
1. Distance of each bus-stop from city-center, Kudirka square (/coords/coords.csv)
2. Weather history at 30min granularity (/weather/weather.csv)
3. Time-lagged covariates for time-series predictions
<br>


*Modelling:*<br>

This data is then used to GridSearch over RandomForest and AdaBoost models and train them to predict passenger loads in specific bus lines for 2017 January, based on 2016 data. TimeSeriesSplit is used throughout the models (both, for train-test split (80/20) and cross-validation).

Due to memory limitations, the full data set has been limited to 15 bus lines, model training being conducted on subsets of data. The resulting models are therefore to be improved.<br>


*Structure:*<br>
- /srautai_2016/      -- raw data from github (csv)<br>
- /tmp/               -- narrowed-down dataset before feature creation<br>
- /train_test_data/   -- train and test data subsets (sparse matrices)<br>
- /transporto_uz~app/ -- web application based on the best model in this notebook<br>
- /best_model*.pkl    -- best models based on subsample CV (\*sample.pkl) and on the entire training data (best_model*.pkl)
- data_categ~map.csv  -- dictionary with categorical variable labels
- data_cleaned.csv    -- final data after feature creation, before transforming to sparse matrices
- get_weather.R       -- R script for weather data download
