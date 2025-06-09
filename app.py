# Save this as app.py
import streamlit as st
import joblib
import numpy as np

model = joblib.load('random_forest_personality.pkl')

st.title("Introvert vs Extrovert Predictor")

# Example input fields (change these as per your dataset)
hours_spent_alone = st.slider("Hours Spent Alone", 0.0, 10.0, 5.0)
stage_fear = st.selectbox("Stage Fear (0 = No, 1 = Yes)", [0, 1])
event_attendance = st.slider("Social Event Attendance", 0, 10, 2)
going_outside = st.slider("Going Outside (times/week)", 0, 7, 3)
friends_size = st.slider("Friend Circle Size", 0, 20, 5)
post_freq = st.slider("Post Frequency", 0.0, 1.0, 0.2)
drained_after_social = st.selectbox("Drained After Socializing (0/1)", [0, 1])

input_data = np.array([[hours_spent_alone, stage_fear, event_attendance,
                        going_outside, friends_size, post_freq,
                        drained_after_social]])

if st.button("Predict"):
    result = model.predict(input_data)
    st.success(f"Predicted Personality: {'Extrovert' if result[0]==1 else 'Introvert'}")
