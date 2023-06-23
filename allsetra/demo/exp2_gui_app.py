import streamlit as st
import numpy as np
import pandas as pd
import movingpandas as mpd
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from haversine import haversine, Unit
import seaborn as sns
from datetime import datetime
from visualz import plot

import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Allsetra Trip Engine ", page_icon=":car:")

st.sidebar.title(":blue[Allsetra] Trip Engine :pager: :car: :pushpin: :rotating_light: :world_map: ")

# Custom styling
custom_style = """
<style>
footer {visibility: hidden;}
}
</style>
"""


sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'gray', 'figure.figsize': (15, 10)})

geolocator = Nominatim(user_agent="Allsetra-Tracking")


def to_seconds(timestamp_str):
    timestamp_obj = datetime.strptime(timestamp_str, "%H:%M:%S")
    total_seconds = timestamp_obj.hour * 3600 + timestamp_obj.minute * 60 + timestamp_obj.second
    return total_seconds


#Convert Longs and Lats to floats.

def cnvrtCoord(val: str) -> float:
    sgn = +1
    if val[0] == 'S' or val[0] == 'W':
        sgn = -1
    return float(val[1:]) * sgn


def get_address_from_GeoPoint(point):
    query = ",".join([str(point.y), str(point.x)])
    location = geolocator.reverse(query)
    # print(f"Point: {point}, Query: {query}, Addr: {location.address}")
    return location.address


def further_clean(df, duration):
    # remove those trip with trip_duration less than 3 minutes

    return df[(df["trip_duration"].dt.seconds > duration) & ((df["trip_distance_in_km"] * 1000) >= 900) ]


def post_process_df(df):
    dfcpy = df.copy()
    dfcpy["trip_duration"] = df.trip_duration.apply(lambda x: str(x).split("days")[-1].split(".")[0])
    return dfcpy

def add_more_data_into_df(cleaned_df):
    # Iterate through the DataFrame rows
    for i in range(1, len(cleaned_df.index)):
        # Get the coordinates of the current and previous points
        lon1, lat1 = cleaned_df["long"].iloc[i], cleaned_df["lat"].iloc[i]
        lon2, lat2 = cleaned_df["long"].iloc[i - 1], cleaned_df["lat"].iloc[i - 1]

        # Calculate the Haversine distance between the points
        dist = haversine((lon1, lat1), (lon2, lat2), unit=Unit.KILOMETERS)
        duration = cleaned_df["DateTimeOfPosition"].iloc[i] - cleaned_df["DateTimeOfPosition"].iloc[i - 1]
        # Update the Haversine distance in the DataFrame
        cleaned_df.at[i, 'haversine_distance'] = dist
        cleaned_df.at[i, 'duration'] = duration

    cleaned_df["duration"] = cleaned_df.duration.apply(lambda x: x.seconds)
    cleaned_df["speed"] = cleaned_df["haversine_distance"] / (cleaned_df["duration"] / 3600)
    return cleaned_df


def clean_df_using_ignition_values(df):
    toggles = []

    for i in range(1, len(df)):
        if df.IgnitionOn.iloc[i] != df.IgnitionOn.iloc[i - 1]:
            if df.IgnitionOn.iloc[i] == 0:
                toggles.append(i)

    keep = [toggle for toggle in toggles]
    to_remove = list()
    for val in df[df.IgnitionOn == 0].index.values:
        if val == 0:
            continue
        if val not in keep:
            to_remove.append(val)

    cleaned_df = df.drop(index=to_remove)
    cleaned_df.reset_index(inplace=True)
    cleaned_df.drop("index", axis=1, inplace=True)

    return cleaned_df


def get_final_df_from_traj_collection(traj_collection):
    data = {}
    start_addrs = []
    end_addrs = []
    start_timestamps = []
    end_timestamps = []
    start_longs_and_lats = []
    end_longs_and_lats = []
    distances = []
    durations = []

    debug_durations = []

    for idx, trajectory in enumerate(traj_collection.trajectories):
        start_time = trajectory.df.index[0]
        end_time = trajectory.df.index[-1]

        start_timestamps.append(start_time)
        end_timestamps.append(end_time)

        diff = end_time - start_time

        debug_durations.append(diff)

        #durations.append(str(diff).split("days")[-1].split(".")[0])
        durations.append(diff)

        start_xy = (trajectory.df["geometry"].x[0], trajectory.df["geometry"].y[0])
        end_xy = (trajectory.df["geometry"].x[-1], trajectory.df["geometry"].y[-1])

        #start_addrs.append(get_address_from_GeoPoint(trajectory.df["geometry"].iloc[0]))
        #end_addrs.append(get_address_from_GeoPoint(trajectory.df["geometry"].iloc[-1]))

        start_longs_and_lats.append(start_xy)
        end_longs_and_lats.append(end_xy)

        distances.append(haversine(start_xy, end_xy, unit=Unit.KILOMETERS))

    # data["start_trip_addr"] = start_addrs
    # data["end_trip_addr"] = end_addrs

    data["start_trip_time"] = start_timestamps
    data["end_trip_time"] = end_timestamps

    data["start_gps_point"] = start_longs_and_lats
    data["end_gps_point"] = end_longs_and_lats

    data["trip_distance_in_km"] = distances

    data["trip_duration"] = durations

    return pd.DataFrame(data)


def remove_outliers(df):
    df = df[df["speed"] < 250]
    df = df.reset_index()
    df = df.drop("index", axis=1)
    return df.reset_index()


def get_trips_using_gap_splitter(df, duration):

    traj = mpd.Trajectory(df[["DateTimeOfPosition", "long", "lat"]], "Allsetra", x='lat', y='long',
                          t='DateTimeOfPosition', crs=4326)
    splitted_traj = mpd.ObservationGapSplitter(traj).split(gap=timedelta(seconds=duration))
    # dfs = [splitted.df for splitted in splitted_traj.trajectories]
    #
    # combined_dfs = pd.concat(dfs)
    final_gap_splitter_df = get_final_df_from_traj_collection(splitted_traj)
    return final_gap_splitter_df
#
# def plot_on_folium(df):
#
#     lat_mean = df["start_gps_point"].apply(lambda x:x[0]).mean()
#     long_mean = df["start_gps_point"].apply(lambda x:x[1]).mean()
#
#     m = folium.Map(location=[lat_mean, long_mean],
#                    zoom_start=4, control_scale=True)
#
#
#
#     # Loop through each row in the dataframe
#     for i, row in df.iterrows():
#         # Setup the content of the popup
#         iframe = folium.IFrame('Allsetra')
#
#         # Initialise the popup using the iframe
#         popup = folium.Popup(iframe, min_width=300, max_width=300)
#
#         # Add each row to the map
#         folium.Marker(location=[row['latitude'], row['longitude']],
#                       popup=popup, c=row['Well Name']).add_to(m)
#
#     st_data = folium_static(m, width=700)
#

st.warning("Upload the Positiondump utc.xlsx. or another xlsx file but with same schema.")

uploaded_file = None

if "df" not in st.session_state:
    uploaded_file = st.file_uploader("Upload Excel file")


if "df" in st.session_state or uploaded_file is not None:

    df = pd.read_excel(uploaded_file) if "df" not in st.session_state else st.session_state["df"]

    if "gpspoint" in df.columns.tolist() or "df" not in st.session_state:
        df = df.rename( columns={"gpspoint": "GPSpoint", "datetimeofposition": "DateTimeOfPosition", "ignitionon": "IgnitionOn"})

    st.write(f""":red[According to this data, the Car Ignition value was ON  {round(((len(df[df["IgnitionOn"] == 1]) / len(df)) * 100))}% times.] """)

    st.session_state["df"] = df.copy()

    duration = st.sidebar.slider(
        "Min Duration to consider for detecting gaps in the data", min_value=180, max_value=900, step=10, value=500
    )

    st.header(":blue[New Trip Detection Algo]")

    # Get Long/Lat from data.
    df["long"] = df["GPSpoint"].apply(lambda x: x.split(",")[0])
    df["lat"] = df["GPSpoint"].apply(lambda x: x.split(",")[-1])

    # Convert string to float
    df['long'] = [cnvrtCoord(x) for x in df['long'].tolist()]
    df['lat'] = [cnvrtCoord(x) for x in df['lat'].tolist()]

    df.drop("GPSpoint", axis=1, inplace=True)

    df['DateTimeOfPosition'] = pd.to_datetime(df['DateTimeOfPosition'])

    df.sort_values("DateTimeOfPosition", inplace=True)

    df.drop_duplicates(subset="DateTimeOfPosition", inplace=True)
    df.reset_index(inplace=True)

    cleaned_df = clean_df_using_ignition_values(df)

    cleaned_df = add_more_data_into_df(cleaned_df)

    with st.spinner("Detecting trips.."):
        final_df = get_trips_using_gap_splitter(remove_outliers(cleaned_df), duration)

    # num_data_points = st.sidebar.slider(
    #     "Number of data points to visualize", min_value=10, max_value=len(final_df), step=10, value=len(final_df)
    # )
    #
    # st.session_state.num_data_points = num_data_points
    #
    # print(num_data_points)

    # Number of trips detected
    final_df = further_clean(final_df, duration)
    final_df = final_df.reset_index().drop("index", axis=1)
    st.subheader(f":violet[Was able to detect #{len(final_df)} trips.] ")
    df = post_process_df(final_df)
    st.dataframe(df)

    st.markdown("---")

    # Summary
    st.subheader(f":violet[Descriptive Statistics] ")
    st.warning("Please note that the trip duration here is shown in seconds. ")

    final_df["trip_duration"] = final_df.trip_duration.apply(lambda x: x.seconds)
    final_df["speed"] = final_df["trip_distance_in_km"] / (final_df["trip_duration"] / 3600)

    st.dataframe(final_df[["trip_distance_in_km", "trip_duration" , "speed"]].describe())

    st.markdown("---")

    #
    # text_input = st.text_input(f"Index of the record that u want to display on map. :red[Choose from: 0->{len(final_df)-1}]", 0)
    # st.write(final_df.iloc[int(text_input)])

    #plot(final_df, st.session_state.num_data_points)



