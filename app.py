import streamlit as st
import pandas as pd
import joblib
import re

# ==========================================
# 1. THE CLASS DEFINITION (Required for unpickling)
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

# Load the model once into memory
@st.cache_resource
def load_brain():
    model = IndieSuccessPredictor()
    model.load_model("indie_model.pkl")
    return model

predictor = load_brain()

# Build the Website Header
st.title("🎮 Indie Game Success Predictor")
st.markdown("Enter your game pitch below to see if it matches current Steam player trends for a breakout hit.")

# Build the Input Form
with st.form("game_pitch_form"):
    st.subheader("Game Details")
    
    col1, col2 = st.columns(2)
    with col1:
        price = st.number_input("Game Price ($)", min_value=0.0, max_value=100.0, value=14.99, step=1.0)
        release_month = st.selectbox("Target Release Month", range(1, 13), index=9) # Default Oct
    with col2:
        achievements = st.number_input("Number of Achievements", min_value=0, value=20)
        release_year = st.selectbox("Target Release Year", [2024, 2025, 2026], index=1)
        
    tags = st.text_input("Steam Tags (comma separated)", "Indie, Pixel Graphics, Action, Rogue-like")
    
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
    # Package the inputs into the JSON dictionary our model expects
    user_data = {
        'price': price, 'required_age': 0, 'dlc_count': 0, 'achievements': achievements,
        'windows': windows, 'mac': mac, 'linux': linux, 
        'release_year': release_year, 'release_month': release_month, 'tags': tags
    }
    
    # Run the prediction
    with st.spinner('Analyzing Steam market trends...'):
        prob = predictor.predict_success(user_data)
    
    # Display the results beautifully
    st.divider()
    st.subheader("Prediction Results")
    
    st.progress(prob)
    st.metric(label="Probability of becoming an Indie Hit", value=f"{prob * 100:.2f}%")
    
    if prob >= 0.20: # Our tuned threshold!
        st.success("✅ GREEN LIGHT! This game matches the metadata signature of highly successful indie games. Proceed with development!")
    elif prob >= 0.05:
        st.warning("⚠️ MODERATE RISK. The market is saturated. Consider refining your genres or adjusting your price point.")
    else:
        st.error("❌ HIGH RISK. Games with this exact profile historically struggle to find an audience on Steam.")