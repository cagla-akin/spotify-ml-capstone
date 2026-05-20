<h1><img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg" height="28" alt="Spotify" style="vertical-align:middle; margin-right:8px"/> Spotify Hit Predictor</h1>

### Can data science predict what makes a song popular?

**Imperial College London | Executive Education in Data Science Programme | Capstone Project 2026**  
**Author:** Cagla Akin  
**Live site:** [cagla-akin.github.io/spotify-ml-capstone](https://cagla-akin.github.io/spotify-ml-capstone/)

---

## The Question

Spotify processes hundreds of thousands of tracks and must decide which artists to surface, promote, and invest in at scale. Music labels face the same challenge from the other side: which songs are worth signing?

This project builds a machine learning tool to predict a Spotify track's **popularity score (0–100)** from audio features and artist metadata, giving both Spotify and the wider music industry a data-driven lens for those decisions.

---

## The Short Answer

**Yes, and the result is surprising.**

The model predicts popularity with strong accuracy. But the more important finding is what actually drives popularity, which it is not the music itself.

> **Release year and artist reputation together dominate predictive power. All 13 audio features combined (tempo, energy, danceability, loudness, and more) account for less than 3% of popularity variance.**

In other words, who releases a song and when matters far more than what the song sounds like.

---

## The Data

| Property | Detail |
|---|---|
| Source | Spotify via Kaggle |
| Size | 169,909 tracks (142,552 after removing unscored tracks) |
| Year range | 1921 – 2020 |
| Target variable | `popularity` (0–100 Spotify score) |
| Audio features | 13 (danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, time_signature) |
| Engineered features | `year` (from release date), `artist_avg_popularity` (mean of artist's top 3 tracks by popularity, computed on training set only) |

---

## Approach

The analysis follows a supervised regression pipeline with strict data leakage prevention; the train/test split is performed before any feature engineering:

```
Raw Data → EDA → Train/Test Split (80/20) → Feature Engineering (fit on train only) → Model Training → Evaluation
```

Five models were trained in order of increasing complexity:

| Model | Purpose |
|---|---|
| **Dummy Regressor** | Predicts the mean; establishes the performance floor |
| **Linear Regression** | Tests linear relationships between features and popularity |
| **Random Forest** | Captures non-linear interactions between features |
| **XGBoost (Default)** | Gradient boosting with default parameters |
| **XGBoost (Tuned)** | RandomizedSearchCV across 6 hyperparameters, 50 iterations, 3-fold CV |

---

## Results

| Model | RMSE | R² | RMSE as % of mean popularity |
|---|---|---|---|
| Dummy Regressor | 18.17 | 0.00 | 48.2% |
| Linear Regression | 10.02 | 0.70 | 26.6% |
| Random Forest | 9.77 | 0.71 | 25.9% |
| **XGBoost (Default)** | **9.65** | **0.72** | **25.6%** |
| XGBoost (Tuned) | 9.72 | 0.71 | 25.8% |

**XGBoost (Default) is the best performing model** with RMSE = 9.65 and R² = 0.72 on 28,511 held-out tracks. Hyperparameter tuning via RandomizedSearchCV produced no meaningful improvement; default parameters were already near optimal. An RMSE of 9.65 means predictions are off by roughly ±10 points on a 0–100 scale, a ~47% improvement over the naive baseline.

---

## Key Finding: Audio Features Don't Matter (As Much)

Feature importance analysis reveals a counterintuitive result:

- `year` — strongest raw predictor (r = 0.82), driven by Spotify's recency bias: tracks from the 2020s average 66 popularity vs. 7 for the 1920s
- `artist_avg_popularity` — most actionable engineered feature; top artists (Billie Eilish, Harry Styles) average ~77 popularity vs. <2 for bottom-ranked artists
- All 13 audio features combined — less than 3% of popularity variance

**What does this mean?** For Spotify, the implication is significant. If editorial decisions, playlist placement, and promotional activity are a major driver of popularity independent of audio quality, then Spotify holds the power to cultivate emerging artists at lower royalty rates, shifting bargaining power away from established acts. For music labels, signing established artists remains the strongest lever for chart performance; optimising a song's sonic characteristics has negligible impact.

---

## Limitations

- Spotify popularity scores are dynamic and decay over time. This dataset is a snapshot; scores will have changed since collection.
- The dataset contains no genre information, which limits the model's ability to distinguish genre-specific patterns. For example, high speechiness may indicate rap or spoken word, but without a genre label the model cannot separate these cases.
- `artist_avg_popularity` falls back to the global mean for artists not seen in training, limiting accuracy for emerging or unknown artists.
- The model does not account for marketing spend, playlist placement, or social media virality; all known drivers of streaming performance that warrant deeper analysis.

---

## Technical Stack

```
Python 3.x
├── pandas / numpy          — data manipulation
├── scikit-learn            — preprocessing, Linear Regression, Random Forest
├── xgboost                 — gradient boosting + hyperparameter tuning
├── matplotlib / seaborn    — visualisation
└── statsmodels             — statistical analysis
```

**Notebook:** `Spotify_HitPredictor_Final.ipynb`

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/cagla-akin/spotify-ml-capstone.git
cd spotify-ml-capstone

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook Spotify_HitPredictor_Final.ipynb
```

The dataset (`spotify_data 2.csv`) must be placed in the project root directory.

---

## Repository Structure

```
spotify-ml-capstone/
├── Spotify_HitPredictor_Final.ipynb   # Main analysis notebook
├── spotify_data 2.csv                 # Dataset (169,909 tracks)
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## About the Project

This capstone was completed as part of the **Imperial College London Data Science Programme (2026)**. The case study brief posed the question: *"What makes a song popular?"*, approached from the perspective of both Spotify and music labels seeking data-driven decisions on discovery, production, and promotion.

The project covers the full data science lifecycle: exploratory analysis, feature engineering, model selection, hyperparameter tuning, and business interpretation.

---

*Built with Python · scikit-learn · XGBoost · Jupyter*
