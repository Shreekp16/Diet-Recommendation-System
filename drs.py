import streamlit as st
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import base64
import random

# ---------- CONFIG ----------
st.set_page_config(page_title="NutriPredict AI",page_icon="🥗",layout="wide")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
BG_PATH = os.path.join(BASE_DIR, "assets", "images", "bg1.png")


# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


model = load_model()
if model is None:
    st.error("❌ Model failed to load")
    st.stop()


# ---------- BACKGROUND ----------
def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


bg = get_base64_image(BG_PATH)

if bg:
    st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{bg}") no-repeat center center fixed;
        background-size: cover;
    }}
    .stApp::before {{
        content:"";
        position:fixed;
        inset:0;
        background:rgba(0,0,0,0.55);
        backdrop-filter:blur(3px);
        z-index:-1;
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------- CSS ----------
st.markdown("""
<style>
html, body, [class*="css"] {
    color: white !important;
}

/* Labels */
label, .stRadio label, .stSelectbox label, .stNumberInput label {
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

div[role="radiogroup"] label,
div[role="radiogroup"] p,
.stRadio label {
    color: white !important;
    opacity: 1 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ff7e00, #ff3c00);
    color: white !important;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
    width: 100%;
}
.stButton > button:hover {
    transform: scale(1.02);
}

/* Meal Cards */
.card{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding:20px;
    border-radius:20px;
    margin-bottom:15px;
    border:1px solid rgba(255,255,255,0.10);
}

.meal-head{
    font-size:24px;
    font-weight:700;
    color:white !important;
    margin-bottom:10px;
}

.meal-text{
    font-size:20px;
    font-weight:600;
    color:#FFD580 !important;
    line-height:1.9;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,126,0,0.12);
    border: 1px solid rgba(255,126,0,0.35);
    border-radius: 16px;
    padding: 14px;
    text-align:center;
}
[data-testid="stMetricLabel"] {
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    color: #ffb347 !important;
    font-size: 34px !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- SESSION ----------
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# ---------- HEADER ----------
st.markdown("""
<div style="
text-align:center;
padding:20px;
margin-bottom:20px;
background:#ff7e00;
border-radius:20px;">
<h1 style="
font-size:48px;
font-weight:800;
color:#3a3a3a;
margin:0;">
🥗 NutriPredict AI
</h1>
<p style="color:#4a4a4a;font-size:18px;margin-top:8px;">
AI-powered personalized nutrition system
</p>
</div>
""", unsafe_allow_html=True)

# ---------- INPUT ----------
if not st.session_state.submitted:

    st.markdown("""
    <h2 style="font-weight:800;margin-top:10px;margin-bottom:15px;">
    <span>👤</span><span style="color:white;"> Enter Your Details</span>
    </h2>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 10, 100)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        diet_pref = st.selectbox("Diet Preference", ["Veg", "Non-Veg"])

    with c2:
        height = st.number_input("Height (cm)", 100.0, 250.0)
        weight = st.number_input("Weight (kg)", 30.0, 200.0)
        region = st.selectbox("Food Style", ["South Indian", "North Indian"])

    with c3:
        activity = st.radio("Activity Level", ["Low", "Moderate", "High"], horizontal=True)
        goal = st.radio("Goal", ["Weight Loss", "Maintain", "Muscle Gain"], horizontal=True)
        budget = st.selectbox("Budget", ["Budget", "Premium"])

    sugar = st.number_input("Sugar Level", 50.0, 300.0)
    cholesterol = st.number_input("Cholesterol", 100.0, 400.0)

    if st.button("🚀 Generate Plan"):
        st.session_state.submitted = True
        st.session_state.data = {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "diet_pref": diet_pref,
            "region": region,
            "budget": budget,
            "activity": activity,
            "goal": goal,
            "sugar": sugar,
            "cholesterol": cholesterol
        }
        st.rerun()

# ---------- OUTPUT ----------
if st.session_state.submitted:

    d = st.session_state.data
    bmi = d["weight"] / ((d["height"] / 100) ** 2)

    gender_map = {"Male": 0, "Female": 1}
    activity_map = {"Low": 0, "Moderate": 1, "High": 2}
    goal_map = {"Weight Loss": 0, "Maintain": 1, "Muscle Gain": 2}

    input_data = np.array([[
        d["age"],
        gender_map[d["gender"]],
        d["height"],
        d["weight"],
        bmi,
        activity_map[d["activity"]],
        d["sugar"],
        d["cholesterol"],
        goal_map[d["goal"]]
    ]])

    pred = model.predict(input_data)[0]

    diet_names = {
        0: "Low Carb Diet",
        1: "Diabetic Diet",
        2: "Heart Healthy Diet",
        3: "Balanced Diet",
        4: "High Protein Diet"
    }

    diet_name = diet_names.get(pred, "Balanced Diet")

    st.markdown(f"""
    <div style="text-align:center;font-size:38px;font-weight:800;color:#ffb347;margin:15px 0;">
    🥗 {diet_name}
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("BMI", f"{bmi:.2f}")
    c2.metric("Sugar", f'{d["sugar"]:.1f}')
    c3.metric("Cholesterol", f'{d["cholesterol"]:.1f}')

    # BMI Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi,
        title={"text": "BMI"},
        gauge={
            "axis": {"range": [10, 50]},
            "steps": [
                {"range": [10, 18], "color": "lightblue"},
                {"range": [18, 25], "color": "green"},
                {"range": [25, 30], "color": "orange"},
                {"range": [30, 50], "color": "red"},
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Daily Plan Heading
    st.markdown("""
    <h2 style="color:white;font-weight:800;margin-top:20px;margin-bottom:15px;">
    🍽️ Your Daily Plan
    </h2>
    """, unsafe_allow_html=True)

    foods = {
        "Breakfast": {
            "Veg": ["🫐 Oats", "🥞 Dosa", "🥪 Veg Sandwich"],
            "Non-Veg": ["🥚 Eggs", "🍗 Chicken Sandwich", "🍳 Omelette"]
        },
        "Lunch": {
            "Veg": ["🍚 Rice + Dal", "🥗 Salad", "🫓 Roti + Curry"],
            "Non-Veg": ["🍗 Chicken Rice", "🐟 Fish Curry", "🍖 Egg Rice"]
        },
        "Dinner": {
            "Veg": ["🥣 Soup", "🫓 Roti + Paneer", "🍲 Khichdi"],
            "Non-Veg": ["🍗 Grilled Chicken", "🐟 Fish Fry", "🥚 Egg Curry"]
        },
        "Snacks": {
            "Veg": ["🥜 Nuts", "🍎 Apple"],
            "Non-Veg": ["🥚 Boiled Eggs", "🥜 Nuts"]
        },
        "Drinks": {
            "Veg": ["🥤 Buttermilk", "🍵 Green Tea"],
            "Non-Veg": ["🥤 Protein Shake", "🥛 Milk"]
        }
    }

    for meal in ["Breakfast", "Lunch", "Dinner"]:
        options = foods[meal][d["diet_pref"]]
        st.markdown(f"""
        <div class="card">
            <div class="meal-head">{meal}</div>
            <div class="meal-text">
                {options[0]} <br>
                {options[1]} <br>
                {options[2]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Snacks + Drinks
    snack = random.choice(foods["Snacks"][d["diet_pref"]])
    drink = random.choice(foods["Drinks"][d["diet_pref"]])

    cc1, cc2 = st.columns(2)

    with cc1:
        st.markdown(f"""
        <div class="card">
            <div class="meal-head">🥜 Snacks</div>
            <div class="meal-text">{snack}</div>
        </div>
        """, unsafe_allow_html=True)

    with cc2:
        st.markdown(f"""
        <div class="card">
            <div class="meal-head">🥤 Drink</div>
            <div class="meal-text">{drink}</div>
        </div>
        """, unsafe_allow_html=True)

    # Calories Graph
    st.markdown("""
<h3 style="
color:#ffb347;
font-weight:800;
margin-top:15px;
margin-bottom:10px;
">
📊 Daily Calories Overview
</h3>
""", unsafe_allow_html=True)

    calories_data = {
        "Breakfast": random.randint(250, 400),
        "Lunch": random.randint(450, 700),
        "Dinner": random.randint(350, 600),
        "Snacks": random.randint(100, 250)
    }

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Bar(
        x=list(calories_data.keys()),
        y=list(calories_data.values()),
        text=list(calories_data.values()),
        textposition="outside",
        marker=dict(
            color=["#00ff88", "#ffb347", "#ff7e00", "#66ccff"],
            line=dict(color="white", width=1)
        )
    ))

    fig_cal.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="white", size=14),
        title=dict(
            text="Estimated Calories Per Meal",
            font=dict(size=22, color="white")
        ),
        xaxis=dict(title="Meals", showgrid=False),
        yaxis=dict(title="Calories", gridcolor="rgba(255,255,255,0.08)"),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig_cal, use_container_width=True)

    if st.button("🔄 Try Again"):
        st.session_state.submitted = False
        st.rerun()
