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
This Track & Field Odds Dashboard helps visualize pre-race betting odds and predicted race times for major athletics competitions. 
Built using Python, Streamlit, and R, this tool integrates data from World Athletics to analyze athletics results.

<br>
<br>
The goal is to provide both predictive insight and retrospective evaluation of athlete performance, using both predicted time and value-based metrics such as "worth". 
In a sport built upon numbers, we believed there was a lack of predictive analytics. Our hope is that the insights shown in this dashboard increase the accessibility of the sport. 

<br>
<br>
            
Since this is our first iteration, we are open to feedback on how to improve this dashboard in the future.

<br><br>
🛠️ Built by: <strong>Naveen Elliott and Will Sorg</strong>  
<br>
📬 Contact: naveenelliott1@gmail.com, wsorg2004@gmail.com
</div>
""", unsafe_allow_html=True)
