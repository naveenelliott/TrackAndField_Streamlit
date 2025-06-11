import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.markdown("""
    <style>
    .about-title {
        text-align: center;
        font-size: 2rem;
        color: #2c3e50;
        margin-top: 1rem;
    }

    .about-text {
        font-size: 1.1rem;
        line-height: 1.6;
        max-width: 900px;
        margin: auto;
        padding-top: 1rem;
        color: #444;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='about-title'>About This Dashboard</div>", unsafe_allow_html=True)

st.markdown("""
<div class='about-text'>
This Track & Field Odds Dashboard helps visualize and analyze pre-race betting odds, predicted race times, and performance metrics 
for major athletics competitions. Built using Python, Streamlit, and Pandas, this tool integrates data from multiple sources, including 
pre-processed betting lines and post-race results.

<br>
<br>
The goal is to provide both predictive insight and retrospective evaluation of athlete performance, ranking models using both 
predicted time and value-based metrics such as "worth". Features include sortable tables, highlight comparisons, and race-specific filters.

<br><br>
🛠️ Built by: <strong>Naveen Elliott and Will Sorg</strong>  
<br>
📬 Contact: naveenelliott1@gmail.com, wsorg2004@gmail.com
</div>
""", unsafe_allow_html=True)
