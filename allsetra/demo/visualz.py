import streamlit as st
import numpy as np
import pandas as pd
import movingpandas as mpd
from datetime import datetime, timedelta

import seaborn as sns
from datetime import datetime

def to_seconds(timestamp_str):
    timestamp_obj = datetime.strptime(timestamp_str, "%H:%M:%S")
    total_seconds = timestamp_obj.hour * 3600 + timestamp_obj.minute * 60 + timestamp_obj.second
    return total_seconds



def plot(final_df, num_data_points):
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
    # st.subheader(":red[Visualize trip_durations]")
    plt = sns.scatterplot(y=final_df.index[:num_data_points], x=final_df["trip_duration"].iloc[:num_data_points], s=150)
    fig = plt.get_figure()
    # fig.savefig("out.png")
    # st.image("out.png")

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
    # fig = plt.get_figure()
    # fig.savefig("sortedout.png")
    # st.image("sortedout.png")

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
    # fig = plt.get_figure()
    # fig.savefig("trip_distance_in_km.png")
    # st.image("trip_distance_in_km.png")

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
    # fig.savefig("sorted-trip_distance_in_km.png")
    # st.image("sorted-trip_distance_in_km.png")

    # old
    plt = sns.scatterplot(y=df.index[:st.session_state.num_data_points],
                          x=sorted(df["trip_distance_in_km"].iloc[:st.session_state.num_data_points], reverse=True),
                          s=150)
    fig = plt.get_figure()

    leg = fig.legend(labels=['NewAlgo', 'OldAlgo'], fontsize='20')
    for text in leg.get_texts():
        text.set_color("white")

    fig.savefig("old-trip_distance_in_km-sorted.png")
    st.image("old-trip_distance_in_km-sorted.png")

    plt.clear()
    st.markdown("---")
