import streamlit as st
import pandas as pd
import joblib
import re

# ==========================================
# 1. THE CLASS DEFINITION (Must match your Colab code)
# ==========================================
class IndieSuccessPredictor:
    def __init__(self):
        self.pipeline = None
        
    def _clean_text_tags(self, tag_string):
        if pd.isna(tag_string): return "indie"
        clean_str = re.sub(r'[^a-zA-Z\s,]', '', str(tag_string))
        tags = [t.strip().lower() for t in clean_str.split(',') if t.strip()]
        return ",".join(tags)

    def load_model(self, path):
        self.pipeline = joblib.load(path)
        
    def predict_success(self, game_dict):
        game_dict['tags'] = self._clean_text_tags(game_dict.get('tags', ''))
        df_input = pd.DataFrame([game_dict])
        return self.pipeline.predict_proba(df_input)[0][1]

# ==========================================
# 2. STREAMLIT WEB APP UI
# ==========================================
st.set_page_config(page_title="Indie Hit Predictor", page_icon="🎮", layout="centered")

@st.cache_resource
def load_brain():
    model = IndieSuccessPredictor()
    model.load_model("indie_model.pkl")
    return model

predictor = load_brain()

st.title("🎮 Indie Game Success Predictor")
st.markdown("Enter your game pitch below to see if it matches current Steam player trends for a breakout hit.")

# --- NEW UI ELEMENT: Month Mapping ---
MONTHS = [
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]

# --- NEW UI ELEMENT: Popular Steam Tags ---
POPULAR_TAGS = [
    "Indie", "Action", "Adventure", "RPG", "Simulation", "Strategy", "Puzzle", 
    "Pixel Graphics", "2D", "3D", "Story Rich", "Atmospheric", "Multiplayer", 
    "Co-op", "Singleplayer", "Rogue-like", "Rogue-lite", "Platformer", 
    "Early Access", "Horror", "Survival", "Open World", "Visual Novel", 
    "Casual", "Turn-Based", "Sci-Fi", "Fantasy", "Anime", "First-Person", 
    "Third-Person", "Shooter", "Management", "Base Building", "Farming Sim", 
    "Sandbox", "Physics", "Funny", "Difficult", "Cute", "Relaxing", "Retro", 
    "Arcade", "Local Co-Op", "Online Co-Op", "PvP", "Crafting", "Exploration"
]

with st.form("game_pitch_form"):
    st.subheader("Game Details")
    
    col1, col2 = st.columns(2)
    with col1:
        price = st.number_input("Game Price ($)", min_value=0.0, max_value=100.0, value=14.99, step=1.0)
        # Replaced the number input with a month name dropdown (Defaults to October / index 9)
        selected_month = st.selectbox("Target Release Month", MONTHS, index=9) 
    with col2:
        achievements = st.number_input("Number of Achievements", min_value=0, value=20)
        release_year = st.selectbox("Target Release Year", [2024, 2025, 2026], index=1)
        
    # Replaced the text input with a searchable multiselect dropdown
    selected_tags = st.multiselect(
        "Steam Tags (Select all that apply)", 
        options=POPULAR_TAGS, 
        default=["Indie", "Pixel Graphics", "Action"] # What is selected by default
    )
    
    st.subheader("Platform Support")
    col3, col4, col5 = st.columns(3)
    windows = col3.checkbox("Windows", value=True)
    mac = col4.checkbox("Mac", value=False)
    linux = col5.checkbox("Linux", value=False)
    
    submitted = st.form_submit_button("Predict Success Probability")

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
if submitted:
    
    # Convert the month name back into a number for the ML model (e.g., "January" -> 1)
    month_number = MONTHS.index(selected_month) + 1
    
    # Convert the list of tags back into a comma-separated string for the ML model
    tags_string = ", ".join(selected_tags)
    
    user_data = {
        'price': price, 'required_age': 0, 'dlc_count': 0, 'achievements': achievements,
        'windows': windows, 'mac': mac, 'linux': linux, 
        'release_year': release_year, 'release_month': month_number, 
        'tags': tags_string
    }
    
    with st.spinner('Analyzing Steam market trends...'):
        prob = predictor.predict_success(user_data)
    
    st.divider()
    st.subheader("Prediction Results")
    st.progress(prob)
    st.metric(label="Probability of becoming an Indie Hit", value=f"{prob * 100:.2f}%")
    
    if prob >= 0.20:
        st.success("✅ GREEN LIGHT! This game matches the metadata signature of highly successful indie games.")
    elif prob >= 0.05:
        st.warning("⚠️ MODERATE RISK. The market is saturated. Consider refining your genres or adjusting your price point.")
    else:
        st.error("❌ HIGH RISK. Games with this exact profile historically struggle to find an audience on Steam.")
