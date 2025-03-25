import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
import math
import os
import pickle


def haversine_distance(p1, p2):
    p1 = [math.radians(_) for _ in p1]
    p2 = [math.radians(_) for _ in p2]
    dist = haversine_distances([p1, p2])
    return 6371000*dist[0][1]


def feet_min_to_meters_sec(s):
    return s/18.288


def feet_to_meters(f):
    return f*0.3048


coordinates_runway = {
    '18R/36L': [40.530218500159428, -3.574838439918973],
    '14R/32L': [40.45647777995444, -3.547191670444303],
    '14L/32R': [40.470008330215407, -3.532580559771673],
    '18L/36R': [40.532622220224376, -3.55938056019154]
}


def runway_heading(runway):
    if runway[:2] == '18':
        return 180
    elif runway[4:6] == '32':
        return 320
    else:
        return 0


wake_vortex_ICAO = {
    'n/a': 0,
    'High performance': 0,
    'High vortex': 1,
    'Heavy': 2,
    '<34,000kg': 3,
    '<136,000kg': 4,
    '<7000kg': 5
}

df = pd.read_csv('predictions_YourTeamName.csv')

for index, row in df.iterrows():
    # Read appropiate scenario and its appropiate values
    scenario = pd.read_parquet(
        './scenarios/' + row['id_scenario'] + ".parquet", engine="pyarrow")
    icao = row['icao24']
    runway = row['runway']
    # Order the data frame by timestamp so that we know which is the last known point
    filtered = scenario[scenario['icao24'] ==
                        icao].sort_values(by='ts', ascending=True)
    last_position_message = filtered[filtered['bds']
                                     == '05'].iloc[-1].dropna()

    # Model variables (heading, ts1, point1, point2, distance_3d, wake_vortex)
    heading = runway_heading(runway)
    altitude = last_position_message['altitude']
    timestamp = last_position_message['ts']
    wake_vortex = filtered['wake_vortex'].dropna().iloc[0]
    wake_vortex = wake_vortex_ICAO[wake_vortex]
    point_ini = [
        last_position_message['lat_deg'],
        last_position_message['lon_deg']
    ]
    point_end = [
        coordinates_runway[runway][0],
        coordinates_runway[runway][1]
    ]
    is_alt_in_meters = True
    if 9 <= last_position_message['tc'] <= 18:
        is_alt_in_meters = False
    altitude_point = altitude if is_alt_in_meters else feet_to_meters(
        altitude)   # altitude in meters

    distance_2d = haversine_distance(point_ini, point_end)
    distance_3d = math.sqrt(
        altitude_point*altitude_point + distance_2d*distance_2d)

    # FALTA CALCULAR EL TIME_DIFF AMB EL MODEL
    # time_diff = ...(...)

    with open('models/model_lr_bias.plk', 'rb') as f:
        model = pickle.load(f)

    data = {
        'hdg': heading,
        'ts1': timestamp,
        'lat_deg1': point_ini[0],
        'lon_deg1': point_ini[1],
        'altitude1': altitude,
        'lat_deg2': point_end[0],
        'lon_deg2': point_end[1],
        'altitude2': 592,
        'distance_3d': distance_3d,
        'wake_code': wake_vortex
    }

    input = pd.DataFrame([data])

    print(input)

    output = model.predict(input)

    print(output)

    df.loc[index, 'seconds_to_threshold'] = output

    print(df)

df.to_csv('./metrics/' + 'predictions_Fly-by-coders.csv', index=False)
