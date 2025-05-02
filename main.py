import json
import pandas as pd
import streamlit as st
from pathlib import Path
from utils import create_interactive_shot_plot, team_of_interest, opponents, show_kpis, calculate_xg, get_ai_analysis
import warnings

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# st.set_page_config(page_title="Recovery Status", page_icon="", layout="wide")

st.set_page_config(
    page_title="Impact",
    page_icon="static/logo.png",
    layout="wide",
)
data_dir = Path("data")

# Reduce top margin
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.8rem;
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

    team_shot_df["xg"] = team_shot_df.apply(lambda row: calculate_xg(row.startPosXM, row.startPosYM, team), axis=1)
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
    shot_df["date"] = shot_df["match"].str.split(" ").str[0]
    return matches, shot_df  # Return the collected data


matches, shot_df = parse_data()

# Sidebar dropdown
selected_match = st.sidebar.selectbox(
    "🎯 Select a Match:",
    options=["all"] + matches,
    index=0,  # Default to first match
)

if selected_match == "all":
    team_options = [team_of_interest, opponents]
    interest_team_df = return_team_shot_df(shot_df, team_of_interest)
    opponent_df = return_team_shot_df(shot_df, opponents)
else:
    match_shot_df = shot_df[shot_df["match"] == selected_match]
    interest_team_df = return_team_shot_df(match_shot_df, team_of_interest
    )
    opponent_df = return_team_shot_df(
        match_shot_df, opponents
    )
    temp = " ".join(shot_df["match"].iloc[0].split(" ")[1:])
    teamA, teamB = temp.split(" vs ")
    team_options = [teamA.strip(), teamB.strip()]
selected_team = st.sidebar.selectbox(
    "👥 Team:",
    options=team_options,
    index=0,  # Default to first match
)

if selected_team == opponents:
    color_options = ["Same", "Team", "Phase", "Body Part"]
else:
    color_options = ["Same", "Player", "Phase", "Body Part"]

color_col = st.sidebar.selectbox(
    "🎨 Color:",
    options=color_options,
    index=0,  # Default to first match
    help="Choose a column as color",
)

show_heatmap = st.sidebar.selectbox(
    "🔥 Heatmap:",
    options=[False, True],
    index=0,  # Default to first match
    help="Choose a column as color",
)

ai_attack_columns = ['playerName','resultName', 'bodyPartName', 
              'shotTypeName','startPosXM', 'startPosYM', 'possessionTypeName', 
              'match', 'time']

ai_defense_columns = ['teamName','resultName', 'bodyPartName', 
              'shotTypeName','startPosXM', 'startPosYM', 'possessionTypeName', 
              'match', 'time']

col2type = {
    "Player": "playerName",
    "Phase": "possessionTypeName",
    "Body Part": "bodyPartName",
    "Team": "teamName",
}
color_type = col2type[color_col] if color_col != "Same" else "Same"

# Get team data
if selected_team == team_of_interest:
    show_kpis(interest_team_df, selected_team)
    fig = create_interactive_shot_plot(interest_team_df, team_of_interest, color_type, show_heatmap)
    df_json = interest_team_df[ai_attack_columns].to_json(orient="records")
    ai_matchanalyst_message = get_ai_analysis(df_json, mode="attack")
else:
    show_kpis(opponent_df, selected_team)
    if selected_match == "all":
        fig = create_interactive_shot_plot(opponent_df, opponents, color_type, show_heatmap)
    else:
        fig = create_interactive_shot_plot(opponent_df, selected_team, color_type, show_heatmap)
    df_json = opponent_df[ai_defense_columns].to_json(orient="records")
    ai_matchanalyst_message = get_ai_analysis(df_json, mode="defense")

st.plotly_chart(fig, use_container_width=True)

st.subheader("AI Match Analyst's Opinion: 💻")
st.write(ai_matchanalyst_message)

st.sidebar.markdown("**Laws of the Game (#10):**")  
st.sidebar.write("The team scoring the greater number of goals is the winner ...")