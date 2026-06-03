import streamlit as st
import pandas as pd
import joblib

model_lr = joblib.load("saved_models/logistic_model.pkl")
model_dt = joblib.load("saved_models/decision_tree_model.pkl")
model_xgb = joblib.load("saved_models/xgboost_model.pkl")
scaler = joblib.load("saved_models/scaler.pkl")
columns = joblib.load("saved_models/columns.pkl")

st.set_page_config(layout="wide")

cola, colb,colc = st.columns(3)

with cola:
    pass
with colc:
    pass
with colb:
    st.title("Student Performance Prediction")

    st.write("""
    This application predicts student exam performance based on academic,
    social, and personal factors.
        """)

st.divider() 

# -------------------- NUMERIC INPUTS (SLIDERS) --------------------
st.header("Numeric Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    hours_studied = st.slider("Hours Studied", 0, 12, 4)
    sleep_hours = st.slider("Sleep Hours", 0, 12, 7)

with col2:
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    previous_scores = st.slider("Previous Scores", 0, 100, 60)

with col3:
    tutoring_sessions = st.slider("Tutoring Sessions", 0, 10, 2)
    physical_activity = st.slider("Physical Activity (hrs/week)", 0, 10, 3)

st.divider()


# -------------------- CATEGORICAL INPUTS (SELECTBOXES) --------------------
st.header("Categorical Inputs")

col4, col5, col6 = st.columns(3)

with col4:
    parental_involvement = st.selectbox(
        "Parental Involvement",
        ["Low", "Medium", "High"]
    )

    motivation_level = st.selectbox(
        "Motivation Level",
        ["Low", "Medium", "High"]
    )

    family_income = st.selectbox(
        "Family Income",
        ["Low", "Medium", "High"]
    )

    learning_disabilities = st.selectbox(
        "Learning Disabilities",
        ["No", "Yes"]
    )

with col5:
    access_to_resources = st.selectbox(
        "Access to Resources",
        ["Low", "Medium", "High"]
    )

    internet_access = st.selectbox(
        "Internet Access",
        ["No", "Yes"]
    )

    teacher_quality = st.selectbox(
        "Teacher Quality",
        ["Low", "Medium", "High"]
    )

    parental_education_level = st.selectbox(
        "Parental Education Level",
        ["High School", "College", "Postgraduate"]
    )

with col6:
    extracurricular_activities = st.selectbox(
        "Extracurricular Activities",
        ["No", "Yes"]
    )

    school_type = st.selectbox(
        "School Type",
        ["Public", "Private"]
    )

    peer_influence = st.selectbox(
        "Peer Influence",
        ["Low", "Medium", "High"]
    )

    distance_from_home = st.selectbox(
        "Distance from Home",
        ["Close", "Moderate", "Far"]
    )
col7 = st.columns(1)
with col7[0]:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )


# -------------------- PREDICT BUTTON --------------------
st.divider()

if st.button("Predict Exam Score", use_container_width=True):
   
   user_data = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Tutoring_Sessions": tutoring_sessions,
        "Physical_Activity": physical_activity,

        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Extracurricular_Activities": extracurricular_activities,
        "Motivation_Level": motivation_level,
        "Internet_Access": internet_access,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education_level,
        "Distance_from_Home": distance_from_home,
        "Gender": gender
    }
   
   df = pd.DataFrame([user_data])

   df = pd.get_dummies(df)
   df = df.reindex(columns=columns, fill_value=0)

   df_raw = df
   df_scaled = scaler.transform(df)

   prediction_lr = model_lr.predict(df_scaled)[0]
   prediction_dt = model_dt.predict(df_raw)[0]
   prediction_xgb = model_xgb.predict(df_raw)[0]

   st.subheader("Predicted Exam Performance")

   colx, coly, colz = st.columns(3)
   with colx:
       if prediction_lr == 1:
           st.success("Logistic Regression Prediction: Pass")
       else:
           st.error("Logistic Regression Prediction: Fail")

   with coly:
       if prediction_dt == 1:
           st.success("Decision Tree Prediction: Pass")
       else:
           st.error("Decision Tree Prediction: Fail")

   with colz:
       if prediction_xgb == 1:
           st.success("XGBoost Prediction: Pass")
       else:
           st.error("XGBoost Prediction: Fail")