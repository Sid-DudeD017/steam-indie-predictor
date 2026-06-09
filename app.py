import streamlit as st
import pandas as pd
import joblib
import datetime
import re 

# ==========================================
# 1. STREAMLIT WEB APP UI & MODEL LOADING
# ==========================================
st.set_page_config(page_title="Indie Hit Predictor", page_icon="🎮", layout="centered")

def clean_tags_series(X_col):
    """Takes a pandas Series of tags, cleans them, and returns a Series."""
    def _clean(tag_string):
        if pd.isna(tag_string): return "indie"
        clean_str = re.sub(r'[^a-zA-Z\s,]', '', str(tag_string))
        tags = [t.strip().lower() for t in clean_str.split(',') if t.strip()]
        return ",".join(tags)
    return X_col.apply(_clean)
# ------------------------------------------------------------------------------

@st.cache_resource
def load_brain():
    return joblib.load("indie_model_optimized.pkl")

# Load the optimized model
pipeline = load_brain()

# Helper function to format the dictionary for the sklearn pipeline
def get_prediction(data_dict):
    df_input = pd.DataFrame([data_dict])
    return pipeline.predict_proba(df_input)[0][1]

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
# 2. PREDICTION & AI OPTIMIZER LOGIC
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
        # Pass the raw data directly to our helper function
        base_prob = get_prediction(user_data)
    
    st.divider()
    st.subheader("Prediction Results")
    st.progress(float(base_prob)) # Ensure progress bar receives a standard float
    st.metric(label="Probability of becoming an Indie Hit", value=f"{base_prob * 100:.2f}%")
    
    st.markdown("##### 📊 Market Context")
    st.markdown("> **Note:** The global success rate for new Indie games on Steam is roughly **6%**. A score above **15%** means your baseline metadata is severely outperforming the market!")

    # ==========================================
    # 3. THE AI CONSULTANT (Counterfactual Optimizer)
    # ==========================================
    st.divider()
    st.subheader("💡 AI Consultant: Optimization Path")
    
    target_prob = 0.15 # Our 15% Baseline goal
    
    if base_prob >= target_prob:
        st.success("🌟 **Your metadata is fully optimized!** You are already above the mathematical baseline. Keep building your masterpiece!")
    else:
        st.markdown(f"Your score is currently below the **15% baseline**. Our AI has simulated thousands of alternate realities and found a mathematical path to boost your success chances:")
        
        simulated_data = user_data.copy()
        current_sim_prob = base_prob
        
        # Define the AI's "Bag of Tricks"
        def try_platforms(d):
            if not d['mac'] or not d['linux']:
                new_d = d.copy()
                new_d['mac'] = True; new_d['linux'] = True
                return new_d, "💻 **Platforms:** Build for Mac & Linux"
            return None, ""
            
        def try_price(d):
            if d['price'] > 19.99:
                new_d = d.copy(); new_d['price'] = 14.99
                return new_d, "💰 **Pricing:** Drop price to $14.99 to avoid AAA competition"
            elif d['price'] > 9.99:
                new_d = d.copy(); new_d['price'] = 9.99
                return new_d, "💰 **Pricing:** Drop price to $9.99 (The indie sweet-spot)"
            return None, ""
            
        def try_achievements(d):
            if d['achievements'] < 30:
                new_d = d.copy(); new_d['achievements'] = 30
                return new_d, "🏆 **Achievements:** Add at least 30 achievements for player retention"
            return None, ""
            
        def try_month(d):
            if d['release_month'] in [9, 10, 11, 12]:
                new_d = d.copy(); new_d['release_month'] = 5
                return new_d, "📅 **Release Window:** Delay launch from Q4 Holiday rush to May"
            return None, ""
            
        def try_tags(d):
            new_d = d.copy()
            if 'Singleplayer' not in new_d['tags']:
                new_d['tags'] += ", Singleplayer"
                return new_d, "🏷️ **Tags:** Add specific playstyle tags (e.g., 'Singleplayer')"
            elif 'Atmospheric' not in new_d['tags']:
                new_d['tags'] += ", Atmospheric"
                return new_d, "🏷️ **Tags:** Add mood-based tags (e.g., 'Atmospheric')"
            return None, ""

        tricks = [try_platforms, try_price, try_achievements, try_month, try_tags]
        used_tricks = set()
        
        # The Greedy Optimizer Loop
        while current_sim_prob < target_prob and len(used_tricks) < len(tricks):
            best_trick = None
            best_data = None
            best_msg = ""
            best_new_prob = current_sim_prob
            
            # Test every unused trick to find the one that gives the BIGGEST boost
            for trick in tricks:
                if trick in used_tricks: continue
                
                test_data, msg = trick(simulated_data)
                if test_data is not None:
                    test_prob = get_prediction(test_data)
                    if test_prob > best_new_prob:
                        best_new_prob = test_prob
                        best_data = test_data
                        best_msg = msg
                        best_trick = trick
            
            # If the best trick actually helped, apply it and show the user!
            if best_data is not None:
                impact = best_new_prob - current_sim_prob
                st.info(f"{best_msg}  \n*(Boosts score by **+{impact*100:.2f}%** ➔ New Projected Score: **{best_new_prob*100:.2f}%**)*")
                
                simulated_data = best_data
                current_sim_prob = best_new_prob
                used_tricks.add(best_trick)
            else:
                break # Mathematically impossible to push the score any higher with our tricks
        
        # Closing statements based on how the optimization went
        if current_sim_prob >= target_prob:
            st.success(f"🎉 **Goal Reached!** Following these exact steps puts your game mathematically above the top-tier indie baseline.")
        else:
            st.warning(f"📈 **Max Optimized Score: {current_sim_prob*100:.2f}%**. The AI has exhausted all metadata optimizations and cannot reach 15%. However, remember the note below!")

    # --- INSPIRATIONAL ENCOURAGEMENT ---
    st.divider()
    st.markdown("### ❤️ A Note to Developers")
    st.markdown("> *\"True art ascends even the most accurate predictions.\"*")
    st.markdown("Machine learning models only know the past, but **you are building the future**. This algorithm looks at cold spreadsheets, but it cannot measure your passion, your unique art style, the tightness of your gameplay loop, or the story you are trying to tell. Use this tool to optimize your store page, but **never let a machine tell you to give up on your masterpiece.** Keep building!")
