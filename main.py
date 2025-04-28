import json
import pandas as pd
import streamlit as st
from pathlib import Path
from utils import create_interactive_shot_plot, team_of_interest, opponents
import warnings

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

data_dir = Path("data")

# Reduce top margin
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def ms_to_min_sec(ms):
    """Convert milliseconds to mm:ss format"""
    seconds = ms // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


@st.cache_data
def return_team_shot_df(shot_df, team):
    if team == opponents:
        team_shot_df = shot_df[shot_df.teamName != team]
    else:
        team_shot_df = shot_df[shot_df.teamName == team]

    # team of interest from bottom to top

    # For start positions
    mask = team_shot_df["startPosXM"] < 0
    if team != team_of_interest:
        mask = team_shot_df["startPosXM"] > 0
    team_shot_df.loc[mask, "startPosXM"] = -team_shot_df.loc[mask, "startPosXM"]
    team_shot_df.loc[mask, "startPosYM"] = -team_shot_df.loc[mask, "startPosYM"]

    # For end positions
    mask = team_shot_df["endPosXM"] < 0
    if team != team_of_interest:
        mask = team_shot_df["endPosXM"] > 0

    team_shot_df.loc[mask, "endPosXM"] = -team_shot_df.loc[mask, "endPosXM"]
    team_shot_df.loc[mask, "endPosYM"] = -team_shot_df.loc[mask, "endPosYM"]

    return team_shot_df


def return_shortname(name):
    names = name.split()
    if len(names) < 2:
        return name  # or raise an exception if you prefer
    first_name = names[0]
    last_name = names[-1]  # this handles cases with middle names
    return f"{first_name[0]}. {last_name}"

@st.cache_data  # Changed from @st.static() which doesn't exist
def parse_data():
    matches = []
    shot_df_list = []
    # sp_df_list = []
    for file_path in data_dir.glob("*"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)  # Append each file's content to data list
            match_name = data["metaData"]["name"]
            matches.append(match_name)
            events_df = pd.DataFrame(data["events"])
            events_df.drop(
                columns=[
                    "eventId",
                    "playerId",
                    "groupId",
                    "teamId",
                    "receiverId",
                    "receiverName",
                    "receiverTeamId",
                    "receiverTeamName",
                    "baseTypeId",
                    "subTypeId",
                    "resultId",
                    "bodyPartId",
                    "shotTypeId",
                    "foulTypeId",
                    "positionTypeId",
                    "formationTypeId",
                    "partId",
                    "possessionTypeId",
                    "synced",
                ],
                inplace=True,
            )
            events_df["match"] = match_name
            shot_events_df = events_df[
                events_df.subTypeName.isin(["BLOCKED_SHOT", "SHOT"])
            ]
            # sp_events_df = events_df[
            #     events_df.subTypeName.isin(
            #         [
            #             "THROW_IN",
            #             "FREE_KICK_AWARDED",
            #             "FREE_KICK",
            #             "THROW_IN_CROSSED",
            #             "GOAL_KICK",
            #             "CORNER_CROSSED",
            #             "CROSS",
            #             "CORNER_SHORT",
            #         ]
            #     )
            # ]
            shot_df_list.append(shot_events_df)
            # sp_df_list.append(sp_events_df)

    shot_df = pd.concat(shot_df_list, ignore_index=True)
    # sp_df = pd.concat(sp_df_list, ignore_index=True)
    half = {
        "FIRST_HALF": "One",
        "SECOND_HALF": "Two",
        "FIRST_OVERTIME": "Three",
        "SECOND_OVERTIME": "Four",
    }
    shot_df["resultName"] = shot_df["resultName"].apply(lambda x: x.lower().title())
    shot_df["bodyPartName"] = shot_df["bodyPartName"].apply(
        lambda x: x.replace("_", " ").lower().title()
    )
    shot_df["possessionTypeName"] = shot_df["possessionTypeName"].apply(
        lambda x: x.replace("_", " ").lower().title()
    )
    shot_df["partName"] = shot_df["partName"].apply(lambda x: half[x])
    shot_df["time"] = shot_df["startTimeMs"].apply(ms_to_min_sec)
    shot_df["playerName"] = shot_df["playerName"].apply(return_shortname) 
    return matches, shot_df  # Return the collected data


matches, shot_df = parse_data()

# Sidebar dropdown
selected_match = st.sidebar.selectbox(
    "Select a Match:",
    options=["all"] + matches,
    index=0,  # Default to first match
    help="Choose which match to analyze",
)

if selected_match == "all":
    team_options = [team_of_interest, opponents]
    interest_team_df = return_team_shot_df(shot_df, team_of_interest)
    opponent_df = return_team_shot_df(shot_df, opponents)
else:
    interest_team_df = return_team_shot_df(
        shot_df[shot_df["match"] == selected_match], team_of_interest
    )
    opponent_df = return_team_shot_df(
        shot_df[shot_df["match"] == selected_match], opponents
    )
    team_options = [team_of_interest, list(opponent_df.teamName.unique())[0]]

selected_team = st.sidebar.selectbox(
    "Team:",
    options=team_options,
    index=0,  # Default to first match
    help="Choose which team to analyze",
)

color_col = st.sidebar.selectbox(
    "Color:",
    options=["None", "Player", "Phase", "Body Part"],
    index=0,  # Default to first match
    help="Choose a column as color",
)

col2type = {
    "Player": "playerName",
    "Phase": "possessionTypeName",
    "Body Part": "bodyPartName",
}
color_type = col2type[color_col] if color_col != "None" else "None"

# Get team data
if selected_team == team_of_interest:
    fig = create_interactive_shot_plot(interest_team_df, team_of_interest, color_type)
else:
    if selected_match == "all":
        fig = create_interactive_shot_plot(opponent_df, opponents, color_type)
    else:
        fig = create_interactive_shot_plot(opponent_df, selected_team, color_type)
st.plotly_chart(fig, use_container_width=True)
