import streamlit as st
import joblib
import numpy as np
import time

# --- WINDOW CONFIG ---
st.set_page_config(page_title="Ship Finder AI", layout="centered")

# --- CUSTOM "DEEP SEA" DESIGN ---
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle, #001d29 0%, #00050a 100%); color: #00ffd0; }
    
    /* Input Container */
    .stVerticalBlock > div:nth-child(2) {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 30px; border-radius: 20px;
        border: 1px solid rgba(0, 255, 208, 0.2);
    }

    /* Radar Button */
    .stButton>button {
        background: linear-gradient(90deg, #00ffd0, #008f7a);
        color: #001d24 !important; font-weight: 900 !important;
        border-radius: 12px !important; height: 3.5em !important;
        border: none !important; width: 100%; letter-spacing: 2px;
    }
    
    /* Result Box with Glow */
    .result-box {
        border: 2px solid; padding: 30px; border-radius: 20px;
        text-align: center; margin-top: 20px;
        background: rgba(0,0,0,0.4);
        box-shadow: 0 0 20px rgba(0, 255, 208, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Load AI
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("Missing AI Files! Please run the training script first.")

# --- UI HEADER ---
st.markdown("<h1 style='text-align: center; color: #00ffd0;'>🚢 SHIP <span style='color:white;'>IDENTIFIER</span></h1>", unsafe_allow_html=True)
st.write("---")

# --- SIMPLE INPUTS (No Complex Words) ---
st.markdown("### 📋 Ship Description")
col1, col2 = st.columns(2)

with col1:
    length = st.number_input("How long is the ship? (Meters)", 5, 1000, 100)
    speed = st.slider("How fast is it moving? (km/h)", 5, 120, 25)
    noise = st.select_slider("How loud is the engine?", options=list(range(1, 11)), value=5, 
                             help="1 is very quiet, 10 is very loud")

with col2:
    people = st.number_input("How many people can it carry?", 1, 5000, 50)
    age = st.number_input("How old is the ship? (Years)", 0, 50, 5)

st.markdown("<br>", unsafe_allow_html=True)

# --- SCANNING PROCESS ---
if st.button("🔍 SCAN THE OCEAN"):
    # Simulated Scanning Log (Added Element)
    log = st.empty()
    status_msgs = ["📡 Sending Sonar Ping...", "🌊 Reading Water Ripples...", "💻 Checking 1,000 Ship Records...", "✅ Match Confirmed!"]
    for msg in status_msgs:
        log.markdown(f"<p style='color:#00ffd0; text-align:center;'>{msg}</p>", unsafe_allow_html=True)
        time.sleep(0.5)
    log.empty()

    # AI Prediction
    data = np.array([[length, speed, noise, people, age]])
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0]
    
    # Simple Visual Categories
    if prediction == 0:
        color, name, icon = "#00ffd0", "Personal Boat", "🚤"
        note = "Looks like a fishing boat or a private yacht. Safe and small."
    elif prediction == 1:
        color, name, icon = "#fffb00", "Luxury Cruise Ship", "🛳️"
        note = "Huge ship full of people. Moving at a medium speed."
    else:
        color, name, icon = "#ff4b4b", "Industrial Cargo Ship", "🚢"
        note = "Giant ship carrying containers. Very loud and slow."

    # --- ATTRACTIVE RESULT BOX ---
    st.markdown(f"""
        <div class="result-box" style="border-color: {color};">
            <h5 style="color: grey; margin:0;">AI CLASSIFICATION</h5>
            <h1 style="color: {color}; margin:10px 0;">{icon} {name}</h1>
            <p style="font-size: 18px; color: white;">{note}</p>
        </div>
    """, unsafe_allow_html=True)

# --- SYSTEM FOOTER (Added Decorative Element) ---
st.write("<br>", unsafe_allow_html=True)
st.markdown("""
    <div style="background: rgba(0,255,208,0.1); padding: 10px; border-radius: 8px; font-family: monospace; font-size: 10px; text-align: center;">
        STATUS: ONLINE // ENCRYPTION: ACTIVE
    </div>
""", unsafe_allow_html=True)