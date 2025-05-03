import json
import pandas as pd
import streamlit as st
from pathlib import Path
from utils import create_interactive_shot_plot, team_of_interest, opponents, show_kpis, calculate_xg, get_ai_analysis
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# st.set_page_config(page_title="Recovery Status", page_icon="", layout="wide")

blue = "#4D7DBF"
red = "#C55D57"

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
    if team == team_of_interest:
        team_shot_df = shot_df[shot_df.teamName == team_of_interest]
    else:
        team_shot_df = shot_df[shot_df.teamName != team_of_interest]

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
            shot_df_list.append(shot_events_df)

    shot_df = pd.concat(shot_df_list, ignore_index=True)
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

selected_view = st.sidebar.selectbox(
    "🔎 Select a View:",
    options=["Case Explorer", "Season"],
    index=0,  # Default to first match
)

if selected_view == "Case Explorer":
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
else:
    interest_team_df = return_team_shot_df(shot_df, team_of_interest)
    opponent_df = return_team_shot_df(shot_df, opponents)

    df = pd.concat([interest_team_df, opponent_df])
    df['match_date'] = pd.to_datetime(df['date'])

    # Sort by date
    df = df.sort_values('match_date')

    xg_for = df[df['teamName'] == team_of_interest].groupby(['match','match_date'])['xg'].sum().reset_index().rename(columns={'xg': 'xG_for'})

    # Calculate xG against (opponent's xG in same matches)
    xg_against = df[df['teamName'] != team_of_interest].groupby(['match','match_date'])['xg'].sum().reset_index().rename(columns={'xg': 'xG_against'})

    # Merge results
    match_xg = pd.merge(xg_for, xg_against, on=['match','match_date'], how='outer').fillna(0)

    match_xg.sort_values("match_date", inplace=True)

    match_xg['y_for'] = match_xg['xG_for']
    match_xg['y_against'] = match_xg['xG_against']
    ylabel = 'xG per Match'

    # Create the plot with Plotly
    fig = go.Figure()

    # Add traces for xG For and xG Against
    fig.add_trace(go.Scatter(
        x=match_xg['match_date'],
        y=match_xg['y_for'],
        mode='lines+markers',
        name='xG For',
        line=dict(color=blue, dash='dash'),
        marker=dict(color=blue, size=8),
        hovertemplate='xG For: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=match_xg['match_date'],
        y=match_xg['y_against'],
        mode='lines+markers',
        name='xG Against',
        line=dict(color=red, dash='dash'),
        marker=dict(color=red, size=8),
        hovertemplate='xG Against: %{y:.2f}<extra></extra>'  # No match name here
    ))

    # Add annotations for the last point
    last_date = match_xg.iloc[-1]['match_date']
    last_for = match_xg.iloc[-1]['y_for']
    last_against = match_xg.iloc[-1]['y_against']

    fig.add_annotation(
        x=last_date,
        y=last_for,
        text=f'{last_for:.2f}',
        showarrow=True,
        arrowhead=1,
        ax=20,
        ay=0,
        font=dict(color=blue)
    )

    fig.add_annotation(
        x=last_date,
        y=last_against,
        text=f'{last_against:.2f}',
        showarrow=True,
        arrowhead=1,
        ax=20,
        ay=0,
        font=dict(color=red)
    )

    # Update layout
    fig.update_layout(
        title=f'xG Timeline for {team_of_interest}',
        xaxis_title='Match Date',
        yaxis_title=ylabel,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white', # Changed to white
        xaxis=dict(
            gridcolor='rgba(200,200,200,0.5)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(200,200,200,0.5)',
            showgrid=True
        )
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)