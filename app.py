import streamlit as st
import pandas as pd
import joblib
import re
import datetime

# ==========================================
# 1. THE CLASS DEFINITION
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

MONTHS = [
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]

POPULAR_TAGS = [
    "Indie", "Action", "Adventure", "RPG", "Simulation", "Strategy", "Puzzle", 
    "Pixel Graphics", "2D", "3D", "Story Rich", "Atmospheric", "Multiplayer", 
    "Co-op", "Singleplayer", "Rogue-like", "Rogue-lite", "Platformer", 
    "Early Access", "Horror", "Survival", "Open World", "Visual Novel", 
    "Casual", "Turn-Based", "Sci-Fi", "Fantasy", "Anime", "First-Person", 
    "Third-Person", "Shooter", "Management", "Base Building", "Farming Sim", 
    "Sandbox", "Physics", "Funny", "Difficult", "Cute", "Relaxing", "Retro"
]

current_year = datetime.datetime.now().year
dynamic_years = [current_year, current_year + 1, current_year + 2]

with st.form("game_pitch_form"):
    st.subheader("Game Details")
    
    col1, col2 = st.columns(2)
    with col1:
        price = st.number_input("Game Price ($)", min_value=0.0, max_value=100.0, value=14.99, step=1.0)
        selected_month = st.selectbox("Target Release Month", MONTHS, index=9) 
    with col2:
        achievements = st.number_input("Number of Achievements", min_value=0, value=20)
        release_year = st.selectbox("Target Release Year", dynamic_years)
        
    selected_tags = st.multiselect("Steam Tags (Select all that apply)", options=POPULAR_TAGS, default=["Indie", "Action"])
    
    st.subheader("Platform Support")
    col3, col4, col5 = st.columns(3)
    windows = col3.checkbox("Windows", value=True)
    mac = col4.checkbox("Mac", value=False)
    linux = col5.checkbox("Linux", value=False)
    
    submitted = st.form_submit_button("Predict Success Probability")

# ==========================================
# 3. PREDICTION LOGIC & DYNAMIC INSIGHTS
# ==========================================
if submitted:
    month_number = MONTHS.index(selected_month) + 1
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
    
    # --- UPGRADED: Much larger context note! ---
    st.markdown("##### 📊 Market Context")
    st.markdown("> **Note:** The global success rate for new Indie games on Steam is roughly **6%**. A score above **10%** means your baseline metadata is severely outperforming the market!")
    
    if prob >= 0.20:
        st.success("✅ **GREEN LIGHT!** This game matches the metadata signature of highly successful indie games.")
    elif prob >= 0.08:
        st.warning("⚠️ **MODERATE RISK.** You have a solid foundation, but the market is competitive.")
    else:
        st.error("❌ **HIGH RISK.** Games with this exact profile historically struggle to find an audience.")

    st.divider()
    st.subheader("💡 How to improve your score")
    
    advice_given = False
    
    if not mac or not linux:
        st.info("💻 **Platform Support:** Your game is missing Mac/Linux support. Adding alternative OS support historically boosts visibility and model confidence.")
        advice_given = True
        
    if price > 20.0:
        st.info("💰 **Pricing Strategy:** Games priced over $20 face fierce competition from AAA studios. Consider lowering to the $10-$15 sweet spot.")
        advice_given = True
        
    if len(selected_tags) < 5:
        st.info("🏷️ **Tagging:** The Steam algorithm relies heavily on tags. Aim for at least 5-7 accurate tags.")
        advice_given = True
        
    if achievements == 0:
        st.info("🏆 **Achievements:** Games with 0 achievements often see lower player retention. Adding even 10 basic achievements can boost your math.")
        advice_given = True
        
    if not advice_given:
        st.success("🌟 Your metadata is fully optimized! At this point, success comes down to your marketing, art style, and gameplay loop.")

    # --- BRAND NEW: Inspiration Note ---
    st.divider()
    st.markdown("### ❤️ A Note to Developers")
    st.markdown("> *\"True art ascends even the most accurate predictions.\"*")
    st.markdown("Machine learning models only know the past, but **you are building the future**. This algorithm looks at spreadsheets, but it cannot measure your passion, your unique art style, the tightness of your gameplay loop, or the story you are trying to tell. Use this tool to optimize your store page, but **never let a machine tell you to give up on your masterpiece.** Keep building!")
