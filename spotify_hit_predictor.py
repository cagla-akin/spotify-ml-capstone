import ast
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Hit Predictor",
    page_icon="🎵",
    layout="wide"
)

# ── Load data & train model (cached so it only runs once) ────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("spotify_data 2.csv")
    df = df.dropna().reset_index(drop=True)

    def primary_artist(val):
        try:
            parsed = ast.literal_eval(val)
            return parsed[0] if isinstance(parsed, list) and len(parsed) > 0 else val
        except:
            return val

    df["primary_artist"] = df["artists"].apply(primary_artist)

    top10_mean = (
        df[df["popularity"] > 0]
        .groupby("primary_artist")["popularity"]
        .apply(lambda x: x.nlargest(10).mean())
        .rename("artist_avg_popularity")
    )
    df = df.join(top10_mean, on="primary_artist")
    df["artist_avg_popularity"] = df["artist_avg_popularity"].fillna(df["popularity"].median())

    artist_lookup = (
        top10_mean
        .reset_index()
        .rename(columns={"primary_artist": "artist_name"})
    )
    artist_lookup["artist_lower"] = artist_lookup["artist_name"].str.lower()

    drop_cols = ["id", "name", "artists", "release_date", "popularity", "primary_artist"]
    X = df.drop(columns=drop_cols)
    y = df["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    r2_test = r2_score(y_test, model.predict(X_test))

    feature_cols = X.columns.tolist()
    global_median = df[feature_cols].median()

    return model, feature_cols, artist_lookup, global_median, r2_test, df

model, feature_cols, artist_lookup, global_median, r2_test, df = load_and_train()

# ── Helper: lookup artist avg popularity ─────────────────────────────────────
def get_artist_popularity(name):
    if not name.strip():
        return None, None
    name_lower = name.strip().lower()
    match = artist_lookup[artist_lookup["artist_lower"] == name_lower]
    if match.empty:
        match = artist_lookup[artist_lookup["artist_lower"].str.contains(name_lower, na=False)]
    if match.empty:
        return None, None
    row = match.iloc[0]
    return row["artist_name"], float(row["artist_avg_popularity"])

# ── Helper: popularity gauge chart ───────────────────────────────────────────
def make_gauge(score):
    if score >= 70:
        color = "#1DB954"
        label = "Hit Potential"
    elif score >= 50:
        color = "#FFA500"
        label = "Growing"
    elif score >= 30:
        color = "#FFD700"
        label = "Niche"
    else:
        color = "#E8115B"
        label = "Low Reach"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 52, "color": color}, "suffix": ""},
        title={"text": label, "font": {"size": 22}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 14}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 30],   "color": "#ffe5ec"},
                {"range": [30, 50],  "color": "#fff3cd"},
                {"range": [50, 70],  "color": "#fff8e1"},
                {"range": [70, 100], "color": "#e8f8f0"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
    return fig

# ── Sidebar: inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Artist")
    artist_input = st.text_input(
        "Artist name",
        placeholder="e.g. Taylor Swift, Drake, The Weeknd"
    )

    found_name, found_pop = get_artist_popularity(artist_input)

    if artist_input:
        if found_name:
            st.success(f"✓ {found_name} — avg popularity: {found_pop:.1f}")
            artist_avg = found_pop
        else:
            st.warning("Artist not found. Using dataset average.")
            artist_avg = float(global_median["artist_avg_popularity"])
    else:
        artist_avg = float(global_median["artist_avg_popularity"])

    st.subheader("Audio Features")
    st.caption("Drag the sliders to describe your song's sound.")

    acousticness     = st.slider("Acousticness",       0.0, 1.0, float(global_median["acousticness"]),     0.01)
    danceability     = st.slider("Danceability",       0.0, 1.0, float(global_median["danceability"]),     0.01)
    energy           = st.slider("Energy",             0.0, 1.0, float(global_median["energy"]),           0.01)
    valence          = st.slider("Valence (happiness)", 0.0, 1.0, float(global_median["valence"]),         0.01)
    instrumentalness = st.slider("Instrumentalness",   0.0, 1.0, float(global_median["instrumentalness"]), 0.01)
    speechiness      = st.slider("Speechiness",        0.0, 1.0, float(global_median["speechiness"]),      0.01)
    liveness         = st.slider("Liveness",           0.0, 1.0, float(global_median["liveness"]),         0.01)
    loudness         = st.slider("Loudness (dB)",     -60.0, 0.0, float(global_median["loudness"]),        0.1)
    tempo            = st.slider("Tempo (BPM)",        50.0, 220.0, float(global_median["tempo"]),         1.0)
    duration_ms      = st.slider("Duration (ms)",      30000, 600000, int(global_median["duration_ms"]),   1000)
    explicit         = st.selectbox("Explicit lyrics?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    key              = st.selectbox("Key", list(range(12)),
                                    format_func=lambda x: ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][x])
    mode             = st.selectbox("Mode", [0, 1], format_func=lambda x: "Major" if x else "Minor")
    year             = st.slider("Release Year", 1990, 2023, 2022, 1)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Spotify Hit Predictor")
st.markdown(
    "Powered by an **XGBoost model** trained on ~170,000 Spotify tracks and deployed on **AWS SageMaker**. "
    "Adjust the sliders on the left and see your prediction update instantly."
)
st.divider()

# Build prediction
input_data = {
    "danceability":     danceability,
    "energy":           energy,
    "key":              key,
    "loudness":         loudness,
    "mode":             mode,
    "speechiness":      speechiness,
    "acousticness":     acousticness,
    "instrumentalness": instrumentalness,
    "liveness":         liveness,
    "valence":          valence,
    "tempo":            tempo,
    "duration_ms":      duration_ms,
    "explicit":         explicit,
    "year":             year,
    "artist_avg_popularity": artist_avg
}

input_df = pd.DataFrame([input_data])[feature_cols]
predicted = float(model.predict(input_df)[0])
predicted = max(0.0, min(100.0, predicted))

col_gauge, col_scale = st.columns([1.2, 1], gap="large")

with col_gauge:
    st.subheader("Predicted Popularity")
    st.plotly_chart(make_gauge(predicted), use_container_width=True)

with col_scale:
    st.subheader("Popularity Scale")

    categories = [
        ("Low reach",      "0–30",   predicted < 30),
        ("Niche audience", "30–50",  30 <= predicted < 50),
        ("Growing track",  "50–70",  50 <= predicted < 70),
        ("Hit potential",  "70–100", predicted >= 70),
    ]

    for label, score_range, active in categories:
        if active:
            st.markdown(
                f"""<div style="padding:0.75rem 1rem; border:1px solid #1DB954;
                border-radius:4px; margin-bottom:0.5rem; background:#0d2b1a;">
                <strong style="color:#1DB954;">{label}</strong>
                <span style="color:#1DB954; float:right;">{score_range}</span></div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style="padding:0.75rem 1rem; border:1px solid #333;
                border-radius:4px; margin-bottom:0.5rem;">
                <span style="color:#ccc;">{label}</span>
                <span style="color:#666; float:right;">{score_range}</span></div>""",
                unsafe_allow_html=True
            )

st.caption(f"Model: XGBoost · ~170k tracks · AWS SageMaker")
