# 🎮 Steam Indie Hit Predictor & AI Consultant

[![Steam Indie Hit Predictor](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://steam-indie-predictor-jju8srpp8q6f55td75rzbq.streamlit.app/)

## 📖 Overview
The **Steam Indie Hit Predictor** is an end-to-end Machine Learning web application designed to help independent game developers optimize their Steam store metadata before launch. 

The global success rate for new indie games on Steam is roughly **6%**. This project uses historical data from tens of thousands of Steam games to calculate the mathematical probability of a new game becoming a breakout hit. More importantly, it features an **AI Consultant (Counterfactual Optimizer)** that simulates thousands of alternate realities to provide developers with actionable, step-by-step advice on how to improve their score.

## ✨ Key Features
* **Live Probability Engine:** Predicts success based on Price, Release Date, Platform Support, Achievements, and Steam Tags.
* **Counterfactual AI Optimizer:** A custom "Greedy Optimizer" algorithm that runs background simulations to find the exact metadata tweaks needed to mathematically boost a game's success rate above the global baseline.
* **Empathetic UX:** Designed specifically for indie developers, acknowledging the limitations of data science and encouraging artistic vision alongside mathematical optimization.

---

## 🧠 Under the Hood (Machine Learning Pipeline)

This project transitions from raw Data Exploration to a fully deployable MLOps pipeline.

### 1. Data Engineering & Preprocessing
* **Raw Data:** Parsed over 80,000 unmoderated JSON/CSV records from the Steam API.
* **Tag Extraction:** Utilized Regular Expressions (Regex) to strip broken characters and upvote counts from Kaggle data, converting dictionary strings into clean lists.
* **Vectorization:** Replaced bulky `MultiLabelBinarizer` arrays with an optimized `CountVectorizer` inside a Scikit-Learn Pipeline to handle dynamically typed tags from the web UI.

### 2. Handling Severe Imbalance (SMOTE)
Because 94% of games in the dataset are "Average" or "Flops", a standard algorithm would succumb to the **Accuracy Paradox** (guessing "Average" every time to achieve 94% accuracy). 
* Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to mathematically synthesize realistic "Hits" in the training data, forcing the algorithm to learn the nuanced boundaries between a $15 generic platformer and a $15 hit platformer.

### 3. The Model
* **Algorithm:** Random Forest Classifier.
* **Optimization:** Trees were pruned (`max_depth=15`, `n_estimators=30`) and the vocabulary size was capped to the top 100 most influential Steam tags. This reduced the pickled model size by over **90%**, allowing it to be instantly loaded into a lightweight Streamlit Cloud container.

---

## 🚀 Installation & Local Deployment

If you want to run this application on your local machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/steam-indie-predictor.git](https://github.com/your-username/steam-indie-predictor.git)
cd steam-indie-predictor
