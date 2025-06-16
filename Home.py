import streamlit as st
import pandas as pd
from PIL import Image
import re
import plotly.graph_objs as go

favicon = Image.open("Logo/tf.png")  # or "favicon.ico"
st.set_page_config(
    page_title="T&F Odds Dashboard",
    page_icon=favicon,
    layout="wide"
)

st.sidebar.markdown("Use this to toggle between pages.")

df = pd.read_csv("final_results_naveen.csv")


# Custom event order
custom_event_order = [
    "Men's 100 Metres", "Men's 200 Metres", "Men's 400 Metres",
    "Men's 800 Metres", "Men's 1500 Metres", "Men's 3000 Metres Steeplechase",
    "Men's 5000 Metres", "Men's 10,000 Metres",
    "Women's 100 Metres", "Women's 200 Metres", "Women's 400 Metres",
    "Women's 800 Metres", "Women's 1500 Metres", "Women's 3000 Metres Steeplechase",
    "Women's 5000 Metres", "Women's 10,000 Metres"
]

# Sidebar or main page filters
st.markdown(f"<h1 style='text-align: center;'>Track and Field Odds Dashboard</h1>", unsafe_allow_html=True)

full_results = pd.read_csv("test_track_meet_results.csv")

end = ["Men's 100 Metres", "Men's 200 Metres", "Men's 400 Metres",
       "Men's 800 Metres", "Men's 400 Metres Hurdles",
       "Men's 3000 Metres Steeplechase", "Women's 100 Metres",
       "Women's 400 Metres", "Women's 800 Metres",
       "Women's 400 Metres Hurdles", "Women's 200 Metres",
       "Men's 5000 Metres", "Women's 3000 Metres", "Men's 1500 Metres",
       "Men's 110 Metres Hurdles", "Women's 1500 Metres",
       "Women's 100 Metres Hurdles", "Men's 3000 Metres",
       "Women's 5000 Metres", "Men's 60 Metres Hurdles",
       "Women's 60 Metres Hurdles", "Men's 60 Metres",
       "Men's 1000 Metres", "Women's 60 Metres", "Men's Mile",
       "Women's Mile", "Women's 3000 Metres Steeplechase",
       "Women's 10,000 Metres", "Men's 10,000 Metres",
       "Men's 60m Hurdles", "Women's 1000 Metres", "Women's 2 Miles",
       "Men's 2 Miles"]

full_results = full_results[full_results['Event'].isin(end)]

# Example: assume df['Date'] contains the raw date strings
def parse_date(date_str):
    date_str = date_str.strip()

    # Handle ranges with two different months (e.g., "29 SEP–05 OCT 2023")
    if '–' in date_str:
        # Use regex to handle both same- and cross-month cases
        match = re.match(r'(\d{2})\s*([A-Z]{3})?(?:–(\d{2})\s*([A-Z]{3}))?\s+(\d{4})', date_str)
        if match:
            start_day, start_month, end_day, end_month, year = match.groups()
            # If start_month is missing (e.g., from earlier code), reuse end_month
            if not start_month:
                start_month = end_month
            if not end_month:
                end_month = start_month
            # Rebuild start date only (you can also return a tuple if you want the full range)
            clean_date_str = f"{start_day} {start_month} {year}"
        else:
            return pd.NaT
    else:
        clean_date_str = date_str

    return pd.to_datetime(clean_date_str, format='%d %b %Y', errors='coerce').date()

# Apply parsing function to your DataFrame
full_results['Date'] = full_results['Date'].apply(parse_date)

full_results2 = pd.read_csv('preProcessed_betting.csv')

full_results = pd.concat([full_results, full_results2], ignore_index=True)

full_results = full_results[['MeetName', 'Event', 'Date', 'Final', 'Name', 'Place', 'AthleteURL']]

full_results['Final'] = full_results['Final'].fillna('Idk')

full_results = full_results.rename(columns={'Date': 'Meet_Date'})

df['Meet_Date'] = pd.to_datetime(df['Meet_Date'], errors='coerce').dt.date
full_results['Meet_Date'] = pd.to_datetime(full_results['Meet_Date'], errors='coerce').dt.date

will_df = pd.read_parquet("plackettLuceScores.parquet", engine="pyarrow")
del will_df['Athlete']

will_df.drop_duplicates(subset=['Event', 'Date', 'Name'], inplace=True)

will_df['Date'] = pd.to_datetime(will_df['Date'], errors='coerce').dt.date

will_df = will_df.rename(columns={'Date': 'Meet_Date'})

will_df['Event'] = will_df['Sex'] + ' ' + will_df['Event']

del will_df['Sex']

df = pd.merge(df, full_results, on=['MeetName', 'Event', 'Final', 'Meet_Date', 'Name'], how='outer')

df = df.loc[df['Meet_Date'] >= pd.to_datetime('2024-12-01').date()].reset_index(drop=True)

df = pd.merge(
    df,
    will_df,
    on=['Event', 'Meet_Date', 'Name'],
    how='outer'
)

df.dropna(subset=['Name'], inplace=True)

# Count non-null 'Pre-Race Odds' per (MeetName, Event, Final)
odds_counts = (
    df[df['American Odds'].notna()]
    .groupby(['MeetName', 'Event', 'Final'])
    .size()
    .reset_index(name='odds_count')
)

# Keep only combinations where odds are available for at least 5 athletes
valid_combinations = odds_counts[odds_counts['odds_count'] >= 5]

df = df.merge(valid_combinations[['MeetName', 'Event', 'Final']], 
                        on=['MeetName', 'Event', 'Final'], how='inner')

filtered_event_set = set(df['Event'].dropna().unique())
filtered_meet_set = set(df['MeetName'].dropna().unique())

meet_options = ["Select a Meet"] + sorted(filtered_meet_set)

event_options = [e for e in custom_event_order if e in filtered_event_set]

with st.container():
    st.markdown(f"<h4 style='text-align: center;'>Select a Meet and Event</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        selected_meet = st.selectbox("", meet_options, index=0)
    if selected_meet == "Select a Meet":
        st.warning("Please select a meet to proceed.")
        st.stop()

    # Filter events based on selected meet
    meet_filtered_df = df[df['MeetName'] == selected_meet]
    event_set_for_meet = set(meet_filtered_df['Event'].dropna().unique())
    event_options = ["Select an Event"] + [e for e in custom_event_order if e in event_set_for_meet]

    with col2:
        selected_event = st.selectbox("", event_options, index=0)
    
    if selected_event == "Select an Event":
        st.warning("Please select an event to proceed.")
        st.stop()

# --- Filter the DataFrame ---
# Filter for meet + event
sub_df = df[(df['MeetName'] == selected_meet) & (df['Event'] == selected_event)]

# --- Identify unique races ---
# Assuming 'Final' column identifies races like 'Heat 1', 'Final', 'Prelims', etc.
race_options = sub_df['Final'].dropna().unique()

# IF I don't drop this then we get an error
#sub_df.dropna(subset=['Target_Mark'], inplace=True)

# If multiple races exist, let the user choose
if len(race_options) > 1:
    selected_race = st.selectbox("Select Race", race_options)
    filtered_df = sub_df[sub_df['Final'] == selected_race]
else:
    filtered_df = sub_df


filtered_df = filtered_df[['Place', 'Name', 'American Odds', 'Target_Mark', 'Predicted_Mark', 'Worth']]

filtered_df.rename(columns={
    'Name': 'Athlete Name',
    'American Odds': 'Pre-Race Odds',
    'Target_Mark': 'Actual Time',
    'Predicted_Mark': 'Predicted Time'}, inplace=True)

filtered_df['Pre-Race Odds'] = pd.to_numeric(filtered_df['Pre-Race Odds'], errors='coerce')

# Format as a string with "+" for positive odds
filtered_df['Pre-Race Odds'] = filtered_df['Pre-Race Odds'].apply(
    lambda x: f"+{int(x)}" if pd.notnull(x) and x >= 0 else f"{int(x)}" if pd.notnull(x) else ""
).astype(str)

filtered_df['Place'] = filtered_df['Place'].astype(str).str.replace('.', '', regex=False)
filtered_df['Place'] = pd.to_numeric(filtered_df['Place'], errors='coerce').astype('Int64')

filtered_df = filtered_df[filtered_df['Place'].notna()]

scatter_df = filtered_df.copy()

filtered_df = filtered_df.set_index("Place").sort_index()

filtered_df['Pred. Place (Time)'] = filtered_df['Predicted Time'].rank(method='min').astype('Int64')
filtered_df['Pred. Place (Worth)'] = filtered_df['Worth'].rank(method='min', ascending=False).astype('Int64')

filtered_df['Diff_Time'] = (filtered_df['Pred. Place (Time)'] - filtered_df.index).abs()
filtered_df['Diff_Worth'] = (filtered_df['Pred. Place (Worth)'] - filtered_df.index).abs()

#del filtered_df['Predicted Time'], filtered_df['Worth']

# --- Show Results ---
st.markdown(f"<h3 style='text-align: center;'>{selected_meet} - {selected_event} Results</h3>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 16px; line-height: 1.6;'>
  <div style="background-color: lightgreen; display: inline-block; padding: 4px 10px; margin: 4px; border-radius: 5px;">Exact Place Match</div>
  <div style="background-color: khaki; display: inline-block; padding: 4px 10px; margin: 4px; border-radius: 5px;">Within +/- 2 Places</div>
  <div style="background-color: lightcoral; display: inline-block; padding: 4px 10px; margin: 4px; border-radius: 5px;">Over 2 Places Off</div>
</div>        
<br>
""", unsafe_allow_html=True)


# Highlighting logic
def highlight_better_prediction(row):
    styles = [''] * len(display_cols)

    time_col = display_cols.index('Pred. Place (Time)')
    worth_col = display_cols.index('Pred. Place (Worth)')

    # Handle Time highlighting
    diff_time = filtered_df.loc[row.name, 'Diff_Time']
    if pd.notna(diff_time):
        if diff_time == 0:
            styles[time_col] = 'background-color: lightgreen'
        elif diff_time <= 2:
            styles[time_col] = 'background-color: khaki'
        else:
            styles[time_col] = 'background-color: lightcoral'

    # Handle Worth highlighting (only if value exists)
    diff_worth = filtered_df.loc[row.name, 'Diff_Worth']
    if pd.notna(diff_worth):
        if diff_worth == 0:
            styles[worth_col] = 'background-color: lightgreen'
        elif diff_worth <= 2:
            styles[worth_col] = 'background-color: khaki'
        else:
            styles[worth_col] = 'background-color: lightcoral'

    return styles

# Apply formatting
display_cols = [
    'Athlete Name', 'Pre-Race Odds', 'Actual Time',
    'Predicted Time', 'Worth',
    'Pred. Place (Time)', 'Pred. Place (Worth)'
]

# Apply formatting
styled_df = filtered_df[display_cols].style\
    .apply(highlight_better_prediction, axis=1)\
    .format({
        'Actual Time': '{:.2f}',
        'Worth': '{:.3f}',
        'Predicted Time': '{:.2f}'
    })


st.dataframe(styled_df, use_container_width=True)

st.markdown(""" <br><br> """, unsafe_allow_html=True)

scatter_df['Actual Time'] = pd.to_numeric(scatter_df['Actual Time'], errors='coerce')
scatter_df['Predicted Time'] = pd.to_numeric(scatter_df['Predicted Time'], errors='coerce')
scatter_df = scatter_df.dropna(subset=['Actual Time', 'Predicted Time'])

avg_actual = scatter_df['Actual Time'].mean()
avg_pred = scatter_df['Predicted Time'].mean()

def format_time(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:05.2f}"

if not any(sprint in selected_event for sprint in ['400', '200', '100']):
    # Create formatted columns for tooltips
    scatter_df['Actual (m:s)'] = scatter_df['Actual Time'].apply(format_time)
    scatter_df['Predicted (m:s)'] = scatter_df['Predicted Time'].apply(format_time)

    # Generate tick labels for both axes
    def generate_ticks(values, event_name):
        min_val = values.min()
        max_val = values.max()
        if any(s in event_name.lower() for s in ['800', '1500']):
            step = 1  # Short sprints: 1-second ticks
        elif any(s in event_name.lower() for s in ['3000', '5000']):
            step = 5  # Mid-distance: 5-second ticks
        else:
            step = 10  # Long-distance or default
        ticks = list(range(int(min_val) - step, int(max_val) + step, step))
        tick_labels = [format_time(t) for t in ticks]
        return ticks, tick_labels

    x_ticks, x_labels = generate_ticks(scatter_df['Predicted Time'], selected_event)
    y_ticks, y_labels = generate_ticks(scatter_df['Actual Time'], selected_event)

    # Format average lines
    avg_actual_label = format_time(avg_actual)
    avg_pred_label = format_time(avg_pred)
else:
    scatter_df['Actual (m:s)'] = scatter_df['Actual Time'].round(2)
    scatter_df['Predicted (m:s)'] = scatter_df['Predicted Time'].round(2)

    # Use default raw seconds for ticks and labels
    x_ticks, x_labels = None, None
    y_ticks, y_labels = None, None
    avg_actual_label = f"{avg_actual:.2f}"
    avg_pred_label = f"{avg_pred:.2f}"

scatter_df['Color'] = scatter_df.apply(
    lambda row: 'green' if row['Actual Time'] < row['Predicted Time'] else 'red',
    axis=1
)

# Plot
fig = go.Figure()

# Add scatter points
fig.add_trace(go.Scatter(
    x=scatter_df['Predicted Time'],
    y=scatter_df['Actual Time'],
    mode='markers',
    marker=dict(size=10, color=scatter_df['Color']),
    customdata=scatter_df[['Athlete Name', 'Predicted (m:s)', 'Actual (m:s)']],
    hovertemplate="<b>%{customdata[0]}</b><br>Predicted: %{customdata[1]}<br>Actual: %{customdata[2]}<extra></extra>",
    name='Athletes'
))

# Add vertical average line (Predicted)
fig.add_shape(
    type="line", x0=avg_pred, x1=avg_pred,
    y0=scatter_df['Actual Time'].min(), y1=scatter_df['Actual Time'].max(),
    line=dict(color="gray", dash="dash")
)

# Add horizontal average line (Actual)
fig.add_shape(
    type="line", x0=scatter_df['Predicted Time'].min(), x1=scatter_df['Predicted Time'].max(),
    y0=avg_actual, y1=avg_actual,
    line=dict(color="gray", dash="dash")
)

# Add annotations for averages
fig.add_annotation(
    x=avg_pred, y=scatter_df['Actual Time'].max(),
    text=f"Avg Pred: {avg_pred_label}",
    showarrow=False, yanchor='bottom', xanchor='left', font=dict(size=10)
)
fig.add_annotation(
    x=scatter_df['Predicted Time'].max(), y=avg_actual,
    text=f"Avg Act: {avg_actual_label}",
    showarrow=False, yanchor='bottom', xanchor='right', font=dict(size=10)
)

# Axis formatting
fig.update_layout(
    title=dict(
        text=f'Predicted vs Actual Times for {selected_event} at {selected_meet}',
        x=0.5,
        xanchor='center',
        font=dict(size=16)
    ),
    height=400,
    width=400,
    margin=dict(l=20, r=20, t=40, b=30),
    xaxis=dict(
        title=f'Predicted Time for {selected_event}',
        tickvals=x_ticks if x_ticks else None,
        ticktext=x_labels if x_labels else None,
        showline=True
    ),
    yaxis=dict(
        title=f'Actual Time for {selected_event}',
        tickvals=y_ticks if y_ticks else None,
        ticktext=y_labels if y_labels else None,
        showline=True
    ),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)