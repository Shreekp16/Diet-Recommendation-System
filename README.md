<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:111827,100:ff7e00&height=200&section=header&text=🥗%20NutriPredict%20AI&fontSize=46&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=ML-Powered%20Personalized%20Diet%20Recommendation%20System&descAlignY=58&descSize=18" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)[![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-KNN-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)[![Pandas](https://img.shields.io/badge/Pandas-EDA-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)[![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)[![Status](https://img.shields.io/badge/Status-Live-22c55e?style=for-the-badge)](https://diet-recommendation-system-2drzcfznrwqxxtr2ou6vny.streamlit.app/)

<br/>

> **Analyze health metrics → Predict your ideal diet → Live better.**
>
> An end-to-end machine learning system that recommends a fully personalized diet plan  > based on your BMI, sugar levels, cholesterol, activity, and fitness goals.

<br/>

[Live Demo](#live-demo) · [Problem](#problem-statement) · [ML Workflow](#ml-workflow) · [Evaluation](#model-evaluation) · [Features](#input-features) · [Diet Plans](#predicted-diet-classes) · [Challenges](#challenges-solved) · [Setup](#installation-guide) · [Author](#author)

</div>

---

##  Live Demo

<div align="center">


### 🔗 [diet-recommendation-system-qvrbhfmbtkcv9jklcxwfku.streamlit.app](https://diet-recommendation-system-qvrbhfmbtkcv9jklcxwfku.streamlit.app/)
*Fill in your health profile → Instantly get your AI-recommended diet plan*

</div>

---

##  Problem Statement

Most diet plans fail because they are **generic** — designed for the average person, not for *you*.

Two people with the same weight-loss goal can need completely different nutrition strategies depending on age, activity level, blood sugar, and cholesterol. **Arogya Plan AI** solves this by training a K-Nearest Neighbors model on diverse health profiles to match users with their most suitable diet category.

| Without NutriPredict AI | With NutriPredict AI ||---|---|| Generic meal plans | Personalized to your exact health profile || Ignores medical indicators | Sugar & cholesterol-aware || Same advice for everyone | Goal-specific: Loss / Maintain / Gain || Manual guesswork | Instant ML-powered prediction |

---


##  ML Workflow

```
User Health Input
       │
       ▼
┌──────────────────────────┐
│   Feature Engineering    │  BMI calc · encode gender, activity, goal
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│   KNN Model  (model.pkl) │  Finds k most similar health profiles
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  Diet Plan Recommendation│  Breakfast · Lunch · Dinner + visualizations
└──────────────────────────┘
       │
       ▼
  Personalized Result 
```

### Why KNN?

- **Similarity-based** — naturally suited to recommendation and user personalization problems
- **Interpretable** — easy to explain what "similar users eat" to non-technical stakeholders
- **No distribution assumptions** — non-parametric, flexible for health data
- **Fast inference** — real-time predictions suitable for model deployment in a web app

### Notebooks

| Notebook | Purpose |
|---|---|
| `Data Cleaning + EDA.ipynb` | Data preprocessing, missing value handling, feature inspection, class distribution analysis, outlier detection, correlation heatmaps |
| `Diet Recommendation Using KNN Algorithm.ipynb` | Full machine learning pipeline — feature engineering, SMOTE resampling, hyperparameter tuning via GridSearchCV, predictive analytics evaluation, model deployment via pickle |

---

##  Input Features

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | Age | Numeric | User age (years) |
| 2 | Gender | Categorical | Male → 0, Female → 1 |
| 3 | Height | Numeric | In centimetres |
| 4 | Weight | Numeric | In kilograms |
| 5 | BMI | Derived | Auto-calculated: weight / (height/100)² |
| 6 | Activity Level | Categorical | Low → 0, Moderate → 1, High → 2 |
| 7 | Sugar Level | Numeric | Blood sugar reading |
| 8 | Cholesterol | Numeric | Cholesterol level |
| 9 | Goal | Categorical | Weight Loss → 0, Maintain → 1, Muscle Gain → 2 |

---

##  Predicted Diet Classes

The trained KNN model predicts one of five categories:

| Class | Diet Plan | Recommended For |
|---|---|---|
| 0 |  Low Carb Diet | Active users targeting fat loss |
| 1 |  Diabetic Diet | High sugar levels or diabetic profiles |
| 2 |  Heart Healthy Diet | Elevated cholesterol, cardiovascular focus |
| 3 |  Balanced Diet | General maintenance and healthy living |
| 4 |  High Protein Diet | Muscle gain and strength goals |

**Sample meal output:**

```
Diet Predicted: High Protein Diet
──────────────────────────────────
 Breakfast  │  Paneer scramble
 Lunch      │  Grilled chicken + rice
 Dinner     │  Protein shake + salad
```

---

##  Model Evaluation

The project explored **5 pipeline variants** across forward/backward sequential feature selection, RFECV, SMOTE, SMOTEENN, and GridSearchCV tuning before landing on the final model. All evaluation used stratified 75/25 train/test splits with StratifiedKFold cross-validation to handle class imbalance (the Diabetic class dominated the 1,000-row dataset). Each pipeline was assessed on both **accuracy** and **macro F1** so minority diet classes weren't masked by the majority class.


### Pipeline Comparison

| # | Pipeline | Accuracy |
|---|---|---|
| 1 | Baseline KNN (RFECV + SMOTE) | 73.2% |
| 2 | Forward Feature Selection + SMOTE | 88.0% |
| 3 | Forward Feature Selection + GridSearchCV | 86.4% |
| 4 | **Backward Feature Selection + GridSearchCV**  | **90.8%** |
| 5 | Correlation-filtered (top 5) + GridSearchCV | 87.2% |

### Final Model — Best Configuration

| Parameter | Value |
|---|---|
| Algorithm | K-Nearest Neighbors |
| Feature Selection | Backward Sequential (5 features retained) |
| Final Features | BMI · Activity Level · Sugar Level · Cholesterol · Goal |
| Resampling | SMOTE (`k_neighbors=5`) |
| Best K | **3** |
| Distance Metric | Manhattan |
| Weight Function | Distance-weighted |
| Hyperparameter Search | GridSearchCV over 5-fold StratifiedKFold |

### Final Performance Metrics

```
Test Accuracy  :  90.8%
Macro F1 Score :  89.4%
Macro Recall   :  92.8%

Per-class breakdown (test set):
 Class              Precision   Recall   F1
 ─────────────────────────────────────────
 Balanced Diet        0.88      1.00     0.94
 Diabetic Diet        0.98      0.88     0.93
 Heart Healthy        0.91      0.95     0.93
 High Protein         0.72      0.87     0.79
 Low Carb             0.85      ~0.85    ~0.85
```

## 📸 App Preview

<p align="center">

<img src="assets/Project Screenshots/desktop view.png" width="48%" alt="Arogya Plan AI Input Screen" /><img src="assets/Project Screenshots/desktop-results 1.png" width="48%" alt="NutriPredict AI Results Dashboard" />

</p>

<p align="center"><i>Desktop View: Input screen and personalized results dashboard with BMI gauge, KPI cards, and meal recommendations.</i></p>

---

##  App Features

Built with Streamlit, the app delivers a polished, responsive experience:

- **Glassmorphism UI** — blurred card panels with a dark overlay background
- **BMI Gauge Chart** — real-time Plotly gauge with colour-coded health zones
- **KPI Metrics** — BMI, sugar, and cholesterol displayed as quick-read cards
- **Daily Meal Cards** — Breakfast, Lunch, and Dinner for the recommended diet
- **Try Again** — reset and re-run with new inputs without refreshing
- **Session state management** — smooth two-screen flow (input → results)

---

##  Challenges Solved

Building a working ML pipeline for a recommendation problem came with several real engineering obstacles:

**1. Class Imbalance** — The Diabetic Diet class dominated the dataset, causing the model to over-predict it. Solved by applying SMOTEENN (a combined oversampling + cleaning technique) and switching the primary evaluation metric to Macro F1, which weights all classes equally regardless of size.

**2. Feature Noise** — Training on all 9 features initially hurt performance. Correlation analysis revealed that Age, Gender, Height, and Weight had weak predictive signal for diet category. Dropping them and retaining only the top 5 features (BMI, Activity Level, Sugar, Cholesterol, Goal) improved the model's generalisation — a core **feature engineering** decision.

**3. Categorical Encoding for ML** — The KNN algorithm requires all-numeric input. Gender, Activity Level, and Goal were categorical strings in the raw dataset. These were mapped to ordinal integers in the preprocessing pipeline so the distance metric could operate correctly.

**4. Streamlit UI State Management** — Streamlit reruns the entire script on every interaction. Managing the two-screen flow (input form → results dashboard) without losing user data required explicit `st.session_state` handling, which is non-obvious for first-time Streamlit developers.

**5. Hyperparameter Tuning at Scale** — GridSearchCV across `n_neighbors` (3–31 odd), 2 weight options, 2 distance metrics, and 3 SMOTE settings created a large search space. Parallel execution (`n_jobs=-1`) and StratifiedKFold ensured results were both fast and statistically sound.

---

### Installation Guide

**Prerequisites:** Python 3.10+, pip

```bash
# 1. Clone the repository
git clone https://github.com/shreekp16/diet-recommendation-system.git
cd diet-recommendation-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run drs.py
```

Open `http://localhost:8501` in your browser.

> **Note:** The trained model file must be present at `model/model.pkl`. Run the KNN notebook first if you need to regenerate it.

---
## 📂 Project Structure

```text
diet-recommendation-system/
│
├──  .streamlit/
│   └── config.toml                    # Streamlit configuration
│
├──  Source/
│   └── diet_recommendation_dataset_1000.xls   # Dataset file
│
├──  assets/
│   ├──  Project Screenshots/
│   │   ├── desktop view.png
│   │   ├── desktop-results 1.png
│   │   ├── desktop-results 2.png
│   │   ├── mobile view.jpeg
│   │   ├── mobile-results 1.jpeg
│   │   └── mobile-results 2.jpeg
│   │
│   └──  images/
│       └── bg1.png                    # Background UI image
│
├──  model/
│   └── model.pkl                      # Trained KNN model
│
├──  Data Cleaning + EDA.ipynb
├──  Diet Recommendation Using KNN Algorithm.ipynb
│
├──  drs.py
├──  requirements.txt
├──  README.md
└──  .gitignore

```
---

##  Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (KNN) |
| Web App | Streamlit |
| Visualizations | Plotly |
| UI Styling | Custom HTML + CSS (glassmorphism) |
| Model Storage | Pickle (.pkl) |


##  Author

<div align="center">

**Shreenath Pangal**  
*Aspiring Data Scientist*

[![GitHub](https://img.shields.io/badge/GitHub-shreekp16-181717?style=for-the-badge&logo=github)](https://github.com/shreekp16)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/shreenath-pangal-82b4103a8)

*Open to Data Science, ML Engineering, and Analytics roles.*

</div>

---

<div align="center">

**Built with ❤️ for healthier living**

*Found this helpful? Drop a ⭐ — it keeps the project going!*

[![GitHub stars](https://img.shields.io/github/stars/shreekp16/Diet-Recommendation-System?style=social)]([https://github.com/shreekp16/Diet-recommendation-system](https://github.com/Shreekp16/Diet-Recommendation-System))
[![GitHub forks](https://img.shields.io/github/forks/shreekp16/Diet-Recommendation-System?style=social)]([https://github.com/Shreekp16/Diet-Recommendation-System](https://github.com/Shreekp16/Diet-Recommendation-System)/network/members)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:ff7e00,100:111827&height=120&section=footer" />

</div>
