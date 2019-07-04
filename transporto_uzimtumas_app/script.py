import pandas as pd
import numpy as np
from datetime import datetime
import ast
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import flask
from flask import Flask, render_template, request

app=Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')



# Function that loads the model, predicts observations one-by-one,
# replacing lagged-outcome variables with predictions and using them
# in subsequent predictions

def ValuePredictor(to_predict):
    predicted_outcome = []
    final_model = joblib.load("best_model_rf.pkl")
    for i in range(0,len(to_predict)):
        if i == 1:
            to_predict.loc[i, "relat_filling_tmin1"] = 0 + float(predicted_outcome[i-1][0])
        if i > 1:
            to_predict.loc[i, "relat_filling_tmin1"] = to_predict.loc[i-1, "relat_filling_tmin1"] + \
                                                                float(predicted_outcome[i-1][0])
        if i >= 2:
            to_predict.loc[i, "relat_filling_tmin2"] = to_predict.loc[i-1, "relat_filling_tmin1"]
        if i >= 3:
            to_predict.loc[i, "relat_filling_tmin3"] = to_predict.loc[i-1, "relat_filling_tmin2"]
        try:
            to_predict["relat_filling_prevstop"] = to_predict.loc[i-20, "relat_filling_tmin1"]
        except:
            pass
        if to_predict.loc[i, "relat_filling_tmin1"] < 0:
            to_predict.loc[i, "relat_filling_tmin1"] = 0
        if to_predict.loc[i, "relat_filling_tmin2"] < 0:
            to_predict.loc[i, "relat_filling_tmin2"] = 0
        if to_predict.loc[i, "relat_filling_tmin3"] < 0:
            to_predict.loc[i, "relat_filling_tmin3"] = 0
        if to_predict.loc[i, "relat_filling_prevstop"] < 0:
            to_predict.loc[i, "relat_filling_prevstop"] = 0
        predicted_outcome.append(final_model.predict(np.array(to_predict)[i].reshape(1,-1)))
        
        result = np.array(predicted_outcome)[-1] + to_predict["relat_filling_tmin1"].tail(1)
        return "Autobusas bus užimtas " + str(result.values*100) + "% (+-7%)" if result.values >= 0 else 0


@app.route('/result',methods = ['POST'])
def result():
    
    bus_trips = pd.read_csv("./static/bus_trips.csv", 
                        parse_dates=[0,2], infer_datetime_format=True)
    if request.method == 'POST':
        list_from_form = request.form.to_dict()
        list_from_form = list(list_from_form.values())
        (list_from_form[0], list_from_form[2]) = (int(list_from_form[0]), int(list_from_form[2]))
        date_to_predict = datetime.strptime(str(datetime.now().date())+" "+list_from_form[1], '%Y-%m-%d %H:%M')
        to_predict = bus_trips.loc[(bus_trips["line"]==list_from_form[2]) & 
                           (bus_trips["day"] <= datetime(2016,1,list_from_form[0]))].reset_index(drop=True)
        try: 
            closest_result = pd.Series(abs(to_predict.loc[(to_predict["day"] == datetime(2016,1,list_from_form[0])) & 
                   (to_predict["stop_name"] == list_from_form[3]) & 
                   (to_predict["direction"] == list_from_form[4]), "time"] - date_to_predict)).idxmin()
        except:
            return f"Pasirinktu metu autobusas {list_from_form[2]} nevažiuoja. Peržiūrėkite autobuso grafiką ir pasirinkite tinkamą laiką."
        to_predict = to_predict.iloc[:closest_result+1]
        to_predict["time"] = to_predict["time"].apply(lambda x: x.time())
        # Minute of the day (to test a polynomial of continuous time)
        to_predict["minute_of_day"] = to_predict["time"].apply(lambda x: (x.hour * 60) + x.minute)

        # Half-hour of the day (to test one-hot encoding)
        to_predict["30min_of_day"] = to_predict["time"].apply(lambda x: (x.hour * 2) + ((x.minute // 30) + 1))

        # Day of the week
        to_predict["weekday"] = to_predict["day"].apply(lambda x: x.weekday() + 1)

        # Day of the year
        to_predict["day_of_year"] = to_predict["day"].apply(lambda x: x.day + 1)

        # Week of the year
        to_predict["week"] = to_predict["day"].apply(lambda x: x.week)

        # Month of the year
        to_predict["month"] = to_predict["day"].apply(lambda x: x.month)

        to_predict["vehicle"] = 21 # 953 vehicle (most obs for January 2016)
        to_predict["stop_name_unique"] = to_predict["stop_name"] + to_predict["direction"]
        to_predict["is_holiday"] = 0
        to_predict.loc[to_predict["day"] == datetime(2016,1,1), "is_holiday"] = 1
        to_predict["holiday_number"] = -1
        to_predict.loc[to_predict["day"] == datetime(2016,1,1), "holiday_number"] = 0

        stop_numbers = pd.read_csv("./static/stop_numbers.csv", header=None, 
                                   names=["stop_name", "stop_number"])
        to_predict = to_predict.merge(stop_numbers, how="left", on="stop_name")

        dist = pd.read_csv("./static/coords.csv")
        to_predict = to_predict.merge(dist, on="stop_number", how="left")
        to_predict["stop_name"] = to_predict["stop_name_x"]
        to_predict.drop(["stop_name_x", "stop_name_y"], axis=1, inplace=True)


        weather = pd.read_csv('./static/weather.csv', parse_dates=[0])
        to_predict = to_predict.dropna(axis=0)
        # 2. Sort values in both DFs
        to_predict["day_time"] = to_predict.apply(lambda x : pd.datetime.combine(x['day'], x['time']),axis=1)
        weather.sort_values("day", inplace=True)

        # 3. For each obs in data, get the closest weather obs (30min measurements)
        to_predict = pd.merge_asof(to_predict, weather, left_on="day_time", right_on="day", direction="nearest")

        # Last 3 stops were marked as <1% on 31st Dec for all lines
        to_predict["relat_filling_tmin1"] = 0
        to_predict["relat_filling_tmin2"] = 0
        to_predict["relat_filling_tmin3"] = 0
        to_predict["relat_filling_prevstop"] = 0

        to_predict.columns = ['day_x', 'line', 'time', 'direction', 'minute_of_day', '30min_of_day', 
                              'weekday', 'day_of_year', 'week', 'month', 'vehicle',  'stop_name_unique', 
                              'is_holiday', 'holiday_number', 'stop_number', 'lon', 'lat', 'distance_from_center', 
                              'stop_name', 'day_time', 'day_y', 'temp', 'humidity', 'wind_speed', 'pressure', 
                              'visibility', 'relat_filling_tmin1', 'relat_filling_tmin2', 'relat_filling_tmin3', 
                              'relat_filling_prevstop']
        to_predict = to_predict[["vehicle", "line", "direction", "stop_name_unique", "holiday_number",
                                "is_holiday", "distance_from_center", "minute_of_day", "30min_of_day",
                                "weekday", "week", "month", "day_of_year",
                                 "relat_filling_tmin1", "relat_filling_tmin2", 
                                "relat_filling_tmin3", "relat_filling_prevstop",
                                "temp", "humidity", "wind_speed", "pressure", "visibility"]]

        category_labels = pd.read_csv("./static/data_categorical_valuemap.csv", header=None,
                             index_col=0)

        line_cat = ast.literal_eval(category_labels[1][1])
        dir_cat = ast.literal_eval(category_labels[1][2])
        stop_cat = ast.literal_eval(category_labels[1][3])
      
        to_predict["line"] = to_predict.line.map(lambda x: line_cat[str(x)])
        to_predict["direction"] = to_predict.direction.map(lambda x: dir_cat[str(x)])
        to_predict["stop_name_unique"] = to_predict.stop_name_unique.map(lambda x: stop_cat[str(x)])
        to_predict = to_predict.dropna(axis=0)
        to_predict = to_predict.reset_index(drop=True)
#        sample = np.random.choice(to_predict.shape[0], len(to_predict)//50, replace=False)
#        to_predict = to_predict.loc[sample,:].reset_index(drop=True)
        result = ValuePredictor(to_predict)

        return render_template("result.html",prediction=str(result))
