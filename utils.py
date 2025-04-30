import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from millify import millify

team_of_interest = "University of North Carolina Wilmington"
opponents = "opponents"

interest_team_color = "#008080"
opponent_color = "#D48681"
shot_color = "#7A9DCF"
marker_size = 9
base_marker = 5

tableau_colors = [
    "#4E79A7",
    "#A0CBE8",
    "#F28E2B",
    "#FFBE7D",
    "#59A14F",
    "#8CD17D",
    "#B6992D",
    "#F1CE63",
    "#499894",
    "#86BCB6",
    "#E15759",
    "#FF9D9A",
    "#79706E",
    "#BAB0AC",
    "#D37295",
    "#FABFD2",
    "#B07AA1",
    "#D4A6C8",
    "#9D7660",
    "#D7B5A6",
]

import numpy as np

def calculate_xg(x, y, team):
    """
    Calculate expected goals (xG) based on shot location and target goal.
    
    Args:
        x (float): x-coordinate of shot (range -52.5 to 52.5)
        y (float): y-coordinate of shot (range -34 to 34)
        target (str): "top" or "bottom" goal
    
    Returns:
        float: xG value between 0 and 1
    """
    if team == team_of_interest:
        target = "top"
    else:
        target = "bottom"
    
    goal_center = (52.5, 0) if target == "top" else (-52.5, 0)
    goal_half_width = 3.66
    
    # Relative coordinates
    dx = x - goal_center[0]
    dy = y - goal_center[1]
    
    # Distance (meters) and angle (radians)
    distance = np.sqrt(dx**2 + dy**2)
    angle_left = np.arctan2(abs(dy) - goal_half_width, abs(dx))
    angle_right = np.arctan2(abs(dy) + goal_half_width, abs(dx))
    angle = angle_right - angle_left
    
    # Trained coefficients (approximated from StatsBomb-like models)
    intercept = -1.45
    dist_coef = -0.08  # More negative = sharper distance penalty
    angle_coef = 0.02   # Positive = higher angle improves xG
    
    logit = intercept + dist_coef * distance + angle_coef * np.degrees(angle)
    xg = 1 / (1 + np.exp(-logit))
    
    return round(xg,3)

def show_kpis(shot_df):
    n_shots = shot_df.shape[0]
    n_goals = shot_df[shot_df.resultName=="Successful"].shape[0]

    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)

    kpi_1.metric(
        r"\# Shots",  # Include the number in the text
        millify(n_shots),
        border=True,
    )

    kpi_2.metric(
        r"\# Goals",
        millify(n_goals),
        border=True,
    )

    kpi_3.metric(
        "xG",
        millify(shot_df.xg.sum(), 2),
        border=True,
    )

    kpi_4.metric(
        "Goal Per Shot",
        f'{millify(100*(n_goals/n_shots), 1)}%',
        border=True,
    )

    kpi_5, kpi_6, kpi_7, kpi_8 = st.columns(4)
    n_very_good_chances = shot_df[shot_df.xg>=0.3].shape[0]
    n_good_chances = shot_df[(shot_df.xg<0.3)&(shot_df.xg>=0.15)].shape[0]
    n_fair_chances = shot_df[(shot_df.xg<0.15)&(shot_df.xg>=0.07)].shape[0]
    n_poor_chances = shot_df[shot_df.xg<0.07].shape[0]
    
    kpi_5.metric(
        r"# :green-badge[Very Good Chances]",
        millify(n_very_good_chances),
        help="xG >= 0.3",
        border=True,
    )

    kpi_6.metric(
        r"# :blue-badge[Good Chances]",
        millify(n_good_chances),
        help="0.15 <= xG < 0.3",
        border=True,
    )

    kpi_7.metric(
        r"# :orange-badge[Fair Chances]",
        millify(n_fair_chances),
        help="0.07 <= xG < 0.15",
        border=True,
    )

    kpi_8.metric(
        r"# :red-badge[Poor Chances]",
        f'{n_poor_chances}',
        help="xG < 0.07",
        border=True,
    )

def show_secondary_kpis(shot_df, team):
    _, _, kpi_1, kpi_2, _, _ = st.columns(6)
    n_shots = shot_df.shape[0]

    if team == team_of_interest:
        n_shots_within_penalty_area = len(
            shot_df[
                (shot_df['startPosXM'] >= 52.5 - 16.5) & 
                (shot_df['startPosYM'].abs() <= 23.82)
                ])
    else:
        n_shots_within_penalty_area = len(
            shot_df[
                (shot_df['startPosXM'] <= 16.5 - 52.5) & 
                (shot_df['startPosYM'].abs() <= 23.82)
            ])

    n_shots_outside_penalty_area = n_shots - n_shots_within_penalty_area
    kpi_1.metric(
        r"# :green-badge[# Shots Inside Penalty Area]",
        millify(n_shots_within_penalty_area),
        border=True,
    )
    kpi_2.metric(
        r"# :red-badge[# Shots Outside Penalty Area]",
        millify(n_shots_outside_penalty_area),
        border=True,
    )

def create_interactive_shot_plot(df, team, color_type):
    """
    Create an interactive shot plot with:
    - Thicker goal net visualization
    - Corrected penalty arcs showing only outside portion
    """
    # Create figure
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter"}]])

    # FIFA standard dimensions
    pitch_length = 105
    pitch_width = 68
    penalty_box_depth = 16.5
    penalty_box_width = 40.32
    goal_area_depth = 5.5
    goal_area_width = 18.32
    center_circle_radius = 9.15
    penalty_spot_distance = 11
    goal_width = 7.32
    goal_depth = 1.3  # Increased net depth

    # Determine y-axis range
    y_range = (
        [0, pitch_length / 2] if team == team_of_interest else [-pitch_length / 2, 0]
    )

    # Filter successful shots
    successful_shots = df[df["resultName"] == "Successful"]
    if color_type != "Same":
        categories = df[color_type].unique()

        for i, category in enumerate(categories):
            category_shots = df[df[color_type] == category]
            color = tableau_colors[i % len(tableau_colors)]

            fig.add_trace(
                go.Scatter(
                    x=category_shots.startPosYM,
                    y=category_shots.startPosXM,
                    mode="markers",
                    marker=dict(
                        color=color, size=base_marker + (marker_size*category_shots.xg), line=dict(color="black", width=0.5)
                    ),
                    name=str(category),
                    legendgroup="categories",
                    showlegend=True,
                    customdata=category_shots[
                        [
                            "playerName",
                            "teamName",
                            "bodyPartName",
                            "possessionTypeName",
                            "partName",
                            "time",
                            "xg",
                            "date"
                        ]
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>" +
                        ("<b>Team:</b> %{customdata[1]}<br>" if team != team_of_interest else "<b>Date:</b> %{customdata[7]}<br>") +
                        "<b>Body Part:</b> %{customdata[2]}<br>"
                        "<b>Phase:</b> %{customdata[3]}<br>"
                        "<b>Half:</b> %{customdata[4]}<br>"
                        "<b>Time:</b> %{customdata[5]}<br>"
                        "<b>xG:</b> %{customdata[6]}"
                        "<extra></extra>"
                    ),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.startPosYM,
                y=df.startPosXM,
                mode="markers",
                marker=dict(
                    color=shot_color, size=base_marker + (marker_size*df.xg), line=dict(color="black", width=0.5)
                ),
                showlegend=False,
                customdata=df[
                    [
                        "playerName",
                        "teamName",
                        "bodyPartName",
                        "possessionTypeName",
                        "partName",
                        "time",
                        "xg",
                        "date"
                    ]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>" +
                    ("<b>Team:</b> %{customdata[1]}<br>" if team != team_of_interest else "<b>Date:</b> %{customdata[7]}<br>") +
                    "<b>Body Part:</b> %{customdata[2]}<br>"
                    "<b>Phase:</b> %{customdata[3]}<br>"
                    "<b>Half:</b> %{customdata[4]}<br>"
                    "<b>Time:</b> %{customdata[5]}<br>"
                    "<b>xG:</b> %{customdata[6]}"
                    "<extra></extra>"
                ),
            )
        )

    # Plot successful shots as football emoji
    fig.add_trace(
        go.Scatter(
            x=successful_shots.startPosYM,
            y=successful_shots.startPosXM,
            mode="text",
            text="⚽",
            textfont=dict(size=11, color="black"),
            showlegend=False,
            customdata=successful_shots[
                [
                    "playerName",
                    "teamName",
                    "bodyPartName",
                    "possessionTypeName",
                    "partName",
                    "time",
                    "xg",
                    "date"
                ]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>" +
                ("<b>Team:</b> %{customdata[1]}<br>" if team != team_of_interest else "<b>Date:</b> %{customdata[7]}<br>") +
                "<b>Body Part:</b> %{customdata[2]}<br>"
                "<b>Phase:</b> %{customdata[3]}<br>"
                "<b>Half:</b> %{customdata[4]}<br>"
                "<b>Time:</b> %{customdata[5]}<br>"
                "<b>xG:</b> %{customdata[6]}"
                "<extra></extra>"
            ),
        )
    )

    showlegend = True if color_type != "None" else False
    # Layout configuration
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=showlegend,
        hoverlabel=dict(
            align="right",  # Align text to the right within the box
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            bordercolor="black"
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",  # Fully transparent background
            bordercolor="lightgray",   # Border color (e.g., lightgray, black, #D3D3D3)
            borderwidth=0.1,         # Border thickness (adjust as needed)
            y=0.9,  # Moves legend down (negative values move it below the plot)
            yanchor="top",  # Anchors the top of the legend at the y position
            x=0.89 if color_type!="teamName" else 1.15,  # Centers horizontally
            xanchor="center",
            font=dict(  # Font customization
                size=14,  # Increase font size
                family="Arial",  # Optional font family
                color="black",  # Optional font color
            ),
        ),
    )

    # Axes configuration
    fig.update_xaxes(
        range=[-pitch_width / 2, pitch_width / 2],
        scaleanchor="y",
        scaleratio=1,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
    )
    fig.update_yaxes(
        range=y_range, showgrid=False, zeroline=False, showticklabels=False
    )

    # Pitch markings style
    line_color = "#888888"
    line_width = 0.5

    # 1. Pitch outline
    fig.add_shape(
        type="rect",
        x0=-pitch_width / 2,
        y0=y_range[0],
        x1=pitch_width / 2,
        y1=y_range[1],
        line=dict(color=line_color, width=line_width),
    )

    # 2. Halfway line (only for full pitch)
    if team is None:
        fig.add_shape(
            type="line",
            x0=-pitch_width / 2,
            y0=0,
            x1=pitch_width / 2,
            y1=0,
            line=dict(color=line_color, width=line_width),
        )

    # 3. Center circle and spot
    if team == team_of_interest:
        theta = np.linspace(0, np.pi, 50)  # Top semicircle
    elif team is None:
        theta = np.linspace(0, 2 * np.pi, 100)  # Full circle
    else:
        theta = np.linspace(np.pi, 2 * np.pi, 50)  # Bottom semicircle

    fig.add_trace(
        go.Scatter(
            x=center_circle_radius * np.cos(theta),
            y=center_circle_radius * np.sin(theta),
            mode="lines",
            line=dict(color=line_color, width=line_width),
            showlegend=False,
            hoverinfo="none",
        )
    )

    if (team == team_of_interest and 0 <= y_range[1]) or (
        team != team_of_interest and y_range[0] <= 0
    ):
        fig.add_shape(
            type="circle",
            x0=-0.15,
            y0=-0.15,
            x1=0.15,
            y1=0.15,
            line=dict(color=line_color, width=line_width),
            fillcolor=line_color,
        )

    # 4. Penalty and goal areas
    # Bottom areas
    if y_range[0] <= -pitch_length / 2 + penalty_box_depth:
        fig.add_shape(
            type="rect",
            x0=-penalty_box_width / 2,
            y0=-pitch_length / 2,
            x1=penalty_box_width / 2,
            y1=-pitch_length / 2 + penalty_box_depth,
            line=dict(color=line_color, width=line_width),
        )
        fig.add_shape(
            type="rect",
            x0=-goal_area_width / 2,
            y0=-pitch_length / 2,
            x1=goal_area_width / 2,
            y1=-pitch_length / 2 + goal_area_depth,
            line=dict(color=line_color, width=line_width),
        )

    # Top areas
    if y_range[1] >= pitch_length / 2 - penalty_box_depth:
        fig.add_shape(
            type="rect",
            x0=-penalty_box_width / 2,
            y0=pitch_length / 2 - penalty_box_depth,
            x1=penalty_box_width / 2,
            y1=pitch_length / 2,
            line=dict(color=line_color, width=line_width),
        )
        fig.add_shape(
            type="rect",
            x0=-goal_area_width / 2,
            y0=pitch_length / 2 - goal_area_depth,
            x1=goal_area_width / 2,
            y1=pitch_length / 2,
            line=dict(color=line_color, width=line_width),
        )

    # 5. Penalty spots
    # Bottom spot
    if y_range[0] <= -pitch_length / 2 + penalty_spot_distance:
        fig.add_shape(
            type="circle",
            x0=-0.15,
            y0=-pitch_length / 2 + penalty_spot_distance - 0.15,
            x1=0.15,
            y1=-pitch_length / 2 + penalty_spot_distance + 0.15,
            line=dict(color=line_color, width=line_width),
            fillcolor=line_color,
        )

    # Top spot
    if y_range[1] >= pitch_length / 2 - penalty_spot_distance:
        fig.add_shape(
            type="circle",
            x0=-0.15,
            y0=pitch_length / 2 - penalty_spot_distance - 0.15,
            x1=0.15,
            y1=pitch_length / 2 - penalty_spot_distance + 0.15,
            line=dict(color=line_color, width=line_width),
            fillcolor=line_color,
        )

    # 6. Penalty arcs (only outside penalty box)
    def add_penalty_arc(fig, arc_center_y, direction):
        radius = 9.15
        theta = (
            np.linspace(0, np.pi, 100)
            if direction == "bottom"
            else np.linspace(np.pi, 2 * np.pi, 100)
        )
        x = radius * np.cos(theta)
        y = arc_center_y + radius * np.sin(theta)

        # REVERSED LOGIC - only show parts outside penalty box
        if direction == "bottom":
            valid = y > -pitch_length / 2 + penalty_box_depth  # Only above penalty box
        else:
            valid = y < pitch_length / 2 - penalty_box_depth  # Only below penalty box

        # Find continuous segments
        segments = []
        current_segment = []
        for i in range(len(theta)):
            if valid[i]:
                current_segment.append(i)
            elif current_segment:
                segments.append(current_segment)
                current_segment = []
        if current_segment:
            segments.append(current_segment)

        # Add each valid segment
        for seg in segments:
            if len(seg) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=x[seg],
                        y=y[seg],
                        mode="lines",
                        line=dict(color=line_color, width=line_width),
                        showlegend=False,
                        hoverinfo="none",
                    )
                )

    # Bottom arc
    if y_range[0] <= -pitch_length / 2 + penalty_spot_distance + 9.15:
        add_penalty_arc(fig, -pitch_length / 2 + penalty_spot_distance, "bottom")

    # Top arc
    if y_range[1] >= pitch_length / 2 - penalty_spot_distance - 9.15:
        add_penalty_arc(fig, pitch_length / 2 - penalty_spot_distance, "top")

    # Bottom goal
    if y_range[0] <= -pitch_length / 2:
        # Enhanced net visualization
        # Main net area (thicker border)
        fig.add_shape(
            type="rect",
            x0=-goal_width / 2,
            y0=-pitch_length / 2 - goal_depth,
            x1=goal_width / 2,
            y1=-pitch_length / 2,
            line=dict(color="black", width=2),  # Thicker border
            fillcolor="black",
        )

    # Top goal
    if y_range[1] >= pitch_length / 2:
        # Main net area (thicker border)
        fig.add_shape(
            type="rect",
            x0=-goal_width / 2,
            y0=pitch_length / 2,
            x1=goal_width / 2,
            y1=pitch_length / 2 + goal_depth,
            line=dict(color="black", width=2),  # Thicker border
            fillcolor="black",
        )

    # Reduce space between plot and modebar
    fig.update_layout(
        autosize=True,
        hovermode="closest",
        margin=dict(t=0, b=0, l=0, r=0),  # Increased left margin
        modebar=dict(
            orientation='h',
        )
    )
    return fig
