import streamlit as st
import numpy as np
import pandas as pd
import movingpandas as mpd
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from haversine import haversine, Unit
import seaborn as sns
from datetime import datetime


import warnings
warnings.filterwarnings('ignore')
#plot_defaults = {'linewidth':5, 'capstyle':'round', 'figsize':(9,3), 'legend':True}
#opts.defaults(opts.Overlay(active_tools=['wheel_zoom'], frame_width=300, frame_height=500))
#hvplot_defaults = {'tiles':None, 'cmap':'Viridis', 'colorbar':True}


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

    return pd.DataFrame(data), debug_durations


def post_process_df(df):
    dfcpy = df.copy()
    dfcpy["trip_duration"] = df.trip_duration.apply(lambda x: str(x).split("days")[-1].split(".")[0])
    return dfcpy


def get_cleaned_and_potential_trips_df(final_df, min_duration):
    cleaned_df = final_df[final_df['trip_duration'].dt.seconds >= min_duration]
    potential_df = final_df[final_df['trip_duration'].dt.seconds < min_duration]
    return cleaned_df, potential_df



def main():

    diameter = st.sidebar.slider(
        "Maximum Diameter to consider (in meter)", min_value=10, max_value=800, step=10, value=800
    )

    duration = st.sidebar.slider(
        "Min Duration to consider (in seconds)", min_value=60, max_value=600, step=10, value=180
    )

    st.header(":blue[New Trip Detection Algo]")

    df = pd.read_excel("positiondump utc .xlsx", index_col=[0])

    # Get Long/Lat from data.
    df["long"] = df["GPSpoint"].apply(lambda x: x.split(",")[0])
    df["lat"] = df["GPSpoint"].apply(lambda x: x.split(",")[-1])

    # Convert string to float
    df['long'] = [cnvrtCoord(x) for x in df['long'].tolist()]
    df['lat'] = [cnvrtCoord(x) for x in df['lat'].tolist()]

    df.drop("GPSpoint", axis=1, inplace=True)

    df['DateTimeOfPosition'] = pd.to_datetime(df['DateTimeOfPosition'])

    # Calculate Trajectory
    traj = mpd.Trajectory(df[["DateTimeOfPosition", "long", "lat"]], "Allsetra", x='lat', y='long',
                          t='DateTimeOfPosition')

    # Calculate Stop-Trajectory Splitter
    traj_collection = mpd.StopSplitter(traj).split(max_diameter=diameter, min_duration=timedelta(seconds=duration))

    with st.spinner("Detecting trips.."):

        final_df, debug_duration = get_final_df_from_traj_collection(traj_collection)


    final_df, potential_trip_df = get_cleaned_and_potential_trips_df(final_df, duration)

    num_data_points = st.sidebar.slider(
        "Number of data points to visualize", min_value=10, max_value=len(final_df), step=10, value=len(final_df)
    )

    st.session_state.num_data_points = num_data_points

    print(num_data_points)

    # Number of trips detected
    st.subheader(f":violet[Was able to detect #{len(final_df)} trips.] ")
    st.dataframe(post_process_df(final_df))

    st.markdown("---")

    # Number of Potential trips detected
    st.subheader(f":violet[The following #{len(potential_trip_df)} trips are potential candidates as well] ")
    st.dataframe(post_process_df(potential_trip_df))
    st.markdown("---")

    # Summary
    st.subheader(f":violet[Descriptive Statistics] ")
    st.warning("Please note that the trip duration here is shown in seconds. ")
    final_df["trip_duration"] = final_df.trip_duration.apply(lambda x: x.seconds)
    st.dataframe(final_df[["trip_distance_in_km", "trip_duration"]].describe())

    st.markdown("---")


    st.markdown("---")


    # --------------------------------------------------------  OLD ALGO --------------------------------------------------



    st.warning("Read the 20230614_204050_Ritregistratie_awgx file")
    df = pd.read_excel("20230614_204050_Ritregistratie_awgx.xlsx")
    df = df.rename(columns={"Ritduur": "trip_duration", "Afstand (km)": "trip_distance_in_km", "Van": "start_gps_point",
                            "Naar": "end_gps_point", "Starttijd": "start_trip_time", "Einddtijd": "end_trip_time"})
    df = df[["start_trip_time", "end_trip_time", "start_gps_point", "end_gps_point", "trip_distance_in_km",
             "trip_duration"]]

    st.subheader(f":red[Old Trip Algo Results]: Was able to detect #{len(df)} trips.")
    df["trip_duration_in_seconds"] = df.trip_duration.apply(lambda x: to_seconds(x))
    st.dataframe(df.sort_values("trip_distance_in_km"))

    st.markdown("---")

    # Summary
    st.subheader(f":violet[Descriptive Statistics] ")
    st.warning("Please note that the trip duration here is shown in seconds. ")
    df["trip_duration"] = df.trip_duration.apply(lambda x: to_seconds(x))
    st.dataframe(df[["trip_distance_in_km", "trip_duration"]].describe())

    st.markdown("---")


    # Plot trip_durations
    #st.subheader(":red[Visualize trip_durations]")
    plt = sns.scatterplot(y=final_df.index[:num_data_points], x=final_df["trip_duration"].iloc[:num_data_points], s=150)
    fig = plt.get_figure()
    #fig.savefig("out.png")
    #st.image("out.png")

    # Plot trip_durations
    st.subheader(":red[Visualize trip_durations]")
    plt = sns.scatterplot(y=df.index[:st.session_state.num_data_points],
                          x=df["trip_duration"].iloc[:st.session_state.num_data_points], s=150)
    fig = plt.get_figure()
    leg = fig.legend(labels=['NewAlgo', 'OldAlgo'], fontsize='20')
    for text in leg.get_texts():
        text.set_color("white")
    fig.savefig("old-trip-duration.png")
    st.image("old-trip-duration.png")

    plt.clear()

    st.markdown("---")




    # Sorted Plot trip_durations
    st.subheader(":red[Visualize Sorted trip_durations]")
    plt = sns.scatterplot(y=final_df.index[:num_data_points],
                          x=sorted(final_df["trip_duration"].iloc[:num_data_points], reverse=True), s=150)
    #fig = plt.get_figure()
    #fig.savefig("sortedout.png")
    #st.image("sortedout.png")

    # Sorted Plot trip_durations
    plt = sns.scatterplot(y=df.index[:st.session_state.num_data_points],
                          x=sorted(df["trip_duration"].iloc[:st.session_state.num_data_points], reverse=True), s=150)
    fig = plt.get_figure()
    leg = fig.legend(labels=['NewAlgo', 'OldAlgo'], fontsize='20')
    for text in leg.get_texts():
        text.set_color("white")
    fig.savefig("old-trip-duration-sorted.png")
    st.image("old-trip-duration-sorted.png")

    plt.clear()

    st.markdown("---")



    # Plot trip_distance_in_km
    st.subheader(":green[Visualize trip_distance_in_km]")
    plt = sns.scatterplot(y=final_df.index[:num_data_points], x=final_df["trip_distance_in_km"].iloc[:num_data_points],
                          s=150)
    #fig = plt.get_figure()
    #fig.savefig("trip_distance_in_km.png")
    #st.image("trip_distance_in_km.png")

    plt = sns.scatterplot(y=df.index[:st.session_state.num_data_points],
                          x=df["trip_distance_in_km"].iloc[:st.session_state.num_data_points], s=150)

    fig = plt.get_figure()

    leg = fig.legend(labels=['NewAlgo', 'OldAlgo'], fontsize='20')
    for text in leg.get_texts():
        text.set_color("white")

    fig.savefig("old-trip_distance_in_km.png")
    st.image("old-trip_distance_in_km.png")
    plt.clear()

    st.markdown("---")

    # Sorted Plot trip_distance_in_km
    st.subheader(":green[Visualize Sorted trip_distance_in_km]")
    plt = sns.scatterplot(y=final_df.index[:num_data_points],
                          x=sorted(final_df["trip_distance_in_km"].iloc[:num_data_points], reverse=True), s=150)
    fig = plt.get_figure()
    #fig.savefig("sorted-trip_distance_in_km.png")
    #st.image("sorted-trip_distance_in_km.png")

    #old
    plt = sns.scatterplot(y=df.index[:st.session_state.num_data_points],
                          x=sorted(df["trip_distance_in_km"].iloc[:st.session_state.num_data_points], reverse=True), s=150)
    fig = plt.get_figure()

    leg = fig.legend(labels=['NewAlgo', 'OldAlgo'], fontsize='20')
    for text in leg.get_texts():
        text.set_color("white")

    fig.savefig("old-trip_distance_in_km-sorted.png")
    st.image("old-trip_distance_in_km-sorted.png")

    plt.clear()
    st.markdown("---")


main()




