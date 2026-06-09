# Steam Indie Hit Predictor & AI Consultant

[![Steam Indie Hit Predictor](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://steam-indie-predictor-jju8srpp8q6f55td75rzbq.streamlit.app/)

## Overview
The **Steam Indie Hit Predictor** is an end-to-end Machine Learning web application designed to help independent game developers optimize their Steam store metadata before launch. 

The global success rate for new indie games on Steam is roughly **6%**. This project uses historical data from tens of thousands of Steam games to calculate the mathematical probability of a new game becoming a breakout hit. More importantly, it features an **AI Consultant (Counterfactual Optimizer)** that simulates thousands of alternate realities to provide developers with actionable, step-by-step advice on how to improve their score.

## Key Features
* **Live Probability Engine:** Predicts success based on Price, Release Date, Platform Support, Achievements, and Steam Tags.
* **Counterfactual AI Optimizer:** A custom "Greedy Optimizer" algorithm that runs background simulations to find the exact metadata tweaks needed to mathematically boost a game's success rate above the global baseline.
* **Empathetic UX:** Designed specifically for indie developers, acknowledging the limitations of data science and encouraging artistic vision alongside mathematical optimization.

---

## Tech Stack
* **Language:** Python 3.12
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (`imblearn`)
* **Data Processing:** Pandas, NumPy, Regex
* **Web Framework & Deployment:** Streamlit, Streamlit Community Cloud

---

## Under the Hood (Machine Learning Pipeline)

This project was built to demonstrate a production-grade Data Science lifecycle, transitioning from raw data exploration to a fully deployable MLOps pipeline.

### 1. Data Engineering & Native Pipelines
* **Tag Extraction:** Utilized Regular Expressions (Regex) to strip broken characters and upvote counts from raw dataset strings.
* **Native Scikit-Learn Architecture:** Instead of using custom wrapper classes, the text-cleaning logic is embedded directly into the pipeline using a `FunctionTransformer`. This creates a completely self-contained DAG (Directed Acyclic Graph) where the exported `.pkl` file inherently knows how to clean messy user web input before passing it to the `CountVectorizer`.

### 2. Handling Severe Imbalance 
Because 94% of games in the dataset are "Average" or "Flops", a standard algorithm would succumb to the **Accuracy Paradox** (achieving 94% accuracy simply by guessing "Flop" every time). 
* **SMOTE:** Applied the Synthetic Minority Over-sampling Technique to mathematically synthesize realistic "Hits" in the training data.
* **Class Weighting:** Implemented `class_weight='balanced'` to heavily penalize the model for missing actual hits, drastically reducing False Negatives.

### 3. Model Tuning & Optimization
* **Algorithm:** Random Forest Classifier.
* **Hyperparameter Tuning:** Utilized `RandomizedSearchCV` (with 3-Fold Cross Validation) to systematically find the best parameters. Crucially, the model was tuned to optimize the **F1-Score** rather than sheer Accuracy, ensuring a proper balance of Precision and Recall.
* **Compression:** Trees were pruned (`max_depth=15`, `n_estimators=30`) and the NLP vocabulary size was capped to the top 100 most influential Steam tags. This reduced the pickled model size by over **90%**, allowing it to be instantly loaded into a lightweight web container.

---

##  Installation & Local Deployment

If you want to run this application on your local machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/steam-indie-predictor.git](https://github.com/your-username/steam-indie-predictor.git)
cd steam-indie-predictor
