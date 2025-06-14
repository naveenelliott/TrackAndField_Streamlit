import streamlit as st
import pandas as pd
from PIL import Image
import re

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

event_options = [e for e in custom_event_order if e in filtered_event_set]
meet_options = sorted(filtered_meet_set)

with st.container():
    st.markdown(f"<h4 style='text-align: center;'>Select a Meet and Event</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        selected_meet = st.selectbox("Meet", meet_options)

    meet_filtered_df = df[df['MeetName'] == selected_meet]
    event_set_for_meet = set(meet_filtered_df['Event'].dropna().unique())
    event_options = [e for e in custom_event_order if e in event_set_for_meet]

    with col2:
        selected_event = st.selectbox("Event", event_options)

# --- Filter the DataFrame ---
# Filter for meet + event
sub_df = df[(df['MeetName'] == selected_meet) & (df['Event'] == selected_event)]

# --- Identify unique races ---
# Assuming 'Final' column identifies races like 'Heat 1', 'Final', 'Prelims', etc.
race_options = sub_df['Final'].dropna().unique()

sub_df.dropna(subset=['Target_Mark'], inplace=True)

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
    try:
        diff_time = filtered_df.loc[row.name, 'Diff_Time']
        diff_worth = filtered_df.loc[row.name, 'Diff_Worth']

        time_col = display_cols.index('Pred. Place (Time)')
        worth_col = display_cols.index('Pred. Place (Worth)')

        if pd.isna(diff_time) or pd.isna(diff_worth):
            return styles

        if diff_time == 0:
            styles[time_col] = 'background-color: lightgreen'
        elif diff_time <= 2:
            styles[time_col] = 'background-color: khaki'
        else:
            styles[time_col] = 'background-color: lightcoral'

        if diff_worth == 0:
            styles[worth_col] = 'background-color: lightgreen'
        elif diff_worth <= 2:
            styles[worth_col] = 'background-color: khaki'
        else:
            styles[worth_col] = 'background-color: lightcoral'
    except:
        pass
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
