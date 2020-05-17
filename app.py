from flask import Flask, jsonify
import pickle
import pickle as pckl
import pandas as pd
import datetime as dt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.cluster import SpectralClustering
import numpy as np
import csv
from shapely.geometry import MultiPoint, Point
from sklearn.preprocessing import StandardScaler

model = {}
values_array = []


# --------------------CLUSTER ELEMENTS FETCHING-------------------------------
def cluster_indices(clust_num, labels_array):
    return np.where(labels_array == clust_num)[0]


# --------------------CENTROID CALCULATION------------------------------------
def cluster_centroid(arr):
    a = []
    for t in arr:
        a.append(values_array[t])
    points = MultiPoint(a)
    return points.centroid.distance(Point(0, 0))


def parser(x):
    return dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def dump_tsa_model():
    # json req

    # json to csv

    series = pd.read_csv('AEP_hourly.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    values = [x for x in series.values]
    sarimax = SARIMAX(values)
    model_fit = sarimax.fit(disp=False)
    pckl.dump(model_fit, open('model.pkl', 'wb'))


def dump_cluster_model():
    # ------------------------DATA PREPARATION----------------------------------------

    values = pd.read_csv('DATA.csv')
    formatted_values = StandardScaler().fit_transform(values)

    with open("DATA.csv") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            values_array.append(row)

    # ------------------------MODEL FITTING-------------------------------------------

    spectral_model = SpectralClustering(n_clusters=3, affinity='rbf')
    label_rbf = spectral_model.fit_predict(formatted_values)

    cluster_0 = cluster_indices(0, label_rbf)
    cluster_1 = cluster_indices(1, label_rbf)
    cluster_2 = cluster_indices(2, label_rbf)

    d0 = cluster_centroid(cluster_0)
    d1 = cluster_centroid(cluster_1)
    d2 = cluster_centroid(cluster_2)

    if d0 > d1:
        if d0 > d2:
            if d2 > d1:
                dump_cluster_model.high = cluster_0
                dump_cluster_model.moderate = cluster_2
                dump_cluster_model.low = cluster_1
            else:
                dump_cluster_model.high = cluster_0
                dump_cluster_model.moderate = cluster_1
                dump_cluster_model.low = cluster_2
        else:
            dump_cluster_model.high = cluster_2
            dump_cluster_model.moderate = cluster_0
            dump_cluster_model.low = cluster_1
    elif d1 > d2:
        if d2 > d0:
            dump_cluster_model.high = cluster_1
            dump_cluster_model.moderate = cluster_2
            dump_cluster_model.low = cluster_0
        else:
            dump_cluster_model.high = cluster_1
            dump_cluster_model.moderate = cluster_0
            dump_cluster_model.low = cluster_2
    elif d0 > d1:
        dump_cluster_model.high = cluster_2
        dump_cluster_model.moderate = cluster_0
        dump_cluster_model.low = cluster_1
    else:
        dump_cluster_model.high = cluster_2
        dump_cluster_model.moderate = cluster_1
        dump_cluster_model.low = cluster_0


# load_model
dump_tsa_model()
sarimax_model = pickle.load(open('model.pkl', 'rb'))

dump_cluster_model()
spectral_model = pickle.load(open('model.pkl', 'rb'))

# app
app = Flask(__name__)


# routes
@app.route('/tsa', methods=['POST'])
def predict():
    dump_tsa_model()
    result = sarimax_model.forecast(steps=10)

    # array to json
    output = {'1st_hour': int(result[0]),
              '2nd_hour': int(result[1]),
              '3rd_hour': int(result[2]),
              '4th_hour': int(result[3]),
              '5th_hour': int(result[4]),
              '6th_hour': int(result[5]),
              '7th_hour': int(result[6]),
              '8th_hour': int(result[7]),
              '9th_hour': int(result[8]),
              '10th_hour': int(result[9]),
              }

    return jsonify(output)


@app.route('/dc', methods=['POST'])
def cluster():
    dump_cluster_model()
    output = {'High Users': dump_cluster_model.high.tolist(),
              'Moderate Users': dump_cluster_model.moderate.tolist(),
              'Low Users': dump_cluster_model.low.tolist(),
              }
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
