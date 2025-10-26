"""
Bot Detector ML for X (Twitter)
- Train on labeled data (CSV)
- Predict bot likelihood for live or offline accounts
- Handles timezone-aware datetimes
- Handles API rate limits
"""

import os
import json
import time
import argparse
from datetime import datetime, timezone
from statistics import mean

# Optional: Tweepy for API access
try:
    import tweepy
    from tweepy.errors import TooManyRequests
except ImportError:
    tweepy = None

# Optional: ML libraries
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib
except ImportError:
    pd = np = RandomForestClassifier = joblib = None

# ----------------------------
# Configuration
# ----------------------------
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
MODEL_PATH = "bot_model.pkl"
USE_API = bool(BEARER_TOKEN) and tweepy is not None

# ----------------------------
# Feature extraction
# ----------------------------
def extract_features_from_user(user, tweets=None):
    metrics = user.get("public_metrics", {})
    followers = metrics.get("followers_count", 0)
    following = metrics.get("following_count", 0)
    tweet_count = metrics.get("tweet_count", 0)

    created_at = (
        datetime.fromisoformat(user["created_at"].replace("Z", "+00:00"))
        if "created_at" in user
        else datetime.now(timezone.utc)
    )
    age_days = max((datetime.now(timezone.utc) - created_at).days, 1)

    features = {
        "followers": followers,
        "following": following,
        "tweet_count": tweet_count,
        "bio_length": len(user.get("description") or ""),
        "has_profile_pic": int(
            "default_profile" not in (user.get("profile_image_url") or "")
        ),
        "is_verified": int(user.get("verified", False)),
        "account_age_days": age_days,
        "follow_ratio": followers / (following + 1),
        "tweets_per_day": tweet_count / age_days,
    }

    # Optional tweet-level metrics
    if tweets:
        urls = sum("http" in (t.get("text") or "") for t in tweets)
        hashtags = sum("#" in (t.get("text") or "") for t in tweets)
        times = [
            datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
            for t in tweets
        ]
        if len(times) > 1:
            deltas = [(times[i] - times[i+1]).total_seconds() for i in range(len(times)-1)]
            avg_interval = mean(abs(d) for d in deltas)
        else:
            avg_interval = 0
        features.update({
            "url_ratio": urls / len(tweets),
            "hashtag_ratio": hashtags / len(tweets),
            "avg_post_interval_sec": avg_interval,
        })

    return features

# ----------------------------
# Fetch live account
# ----------------------------
def fetch_live(username):
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    user = client.get_user(
        username=username,
        user_fields=["created_at", "description", "public_metrics", "verified", "profile_image_url"]
    ).data

    for attempt in range(3):
        try:
            tweets = client.get_users_tweets(id=user.id, max_results=50).data or []
            break
        except TooManyRequests:
            wait = 60 * (attempt + 1)
            print(f"Rate limit hit. Waiting {wait} seconds...")
            time.sleep(wait)
    else:
        tweets = []
        print("Failed to fetch tweets after multiple retries.")

    user_json = user.data
    tweet_jsons = [t.data for t in tweets]
    return user_json, tweet_jsons

# ----------------------------
# Load offline JSON
# ----------------------------
def load_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["user"], data.get("tweets", [])

# ----------------------------
# Train ML model
# ----------------------------
def train_model(csv_path):
    if pd is None or RandomForestClassifier is None:
        raise ImportError("pandas or scikit-learn not installed")

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["is_bot"])
    y = df["is_bot"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}")

# ----------------------------
# Predict bot likelihood
# ----------------------------
def predict_user(username=None, json_path=None):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found. Train first using --train path/to/csv")

    model = joblib.load(MODEL_PATH)

    if username:
        if not USE_API:
            raise RuntimeError("API not available. Set X_BEARER_TOKEN and install tweepy.")
        user, tweets = fetch_live(username)
    elif json_path:
        user, tweets = load_from_json(json_path)
    else:
        raise ValueError("Provide either username or json_path.")

    features = extract_features_from_user(user, tweets)
    X = np.array([list(features.values())])
    proba = model.predict_proba(X)[0][1]

    print("\n--- Prediction Report ---")
    for k, v in features.items():
        print(f"{k:25}: {v}")
    print(f"\nBot likelihood (ML): {proba*100:.1f}%")
    return proba

# ----------------------------
# Main CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot Detector ML for X (Twitter)")
    parser.add_argument("--train", help="Path to labeled CSV for training")
    parser.add_argument("--username", help="X handle for prediction (API mode)")
    parser.add_argument("--json", help="Offline JSON file for prediction")
    args = parser.parse_args()

    if args.train:
        train_model(args.train)
    else:
        predict_user(username=args.username, json_path=args.json)
