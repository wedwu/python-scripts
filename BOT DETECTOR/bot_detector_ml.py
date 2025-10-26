# // bot_detector_ml.py

"""
Bot Detector for X (Twitter) - ML Version
Trains on labeled data and predicts bot likelihood.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

try:
    import tweepy
except ImportError:
    tweepy = None


BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
MODEL_PATH = "bot_model.pkl"


# ----------------------------
# Feature extraction (same as before)
# ----------------------------
def extract_features_from_user(user, tweets=None):
    metrics = user.get("public_metrics", {})
    followers = metrics.get("followers_count", 0)
    following = metrics.get("following_count", 0)
    tweet_count = metrics.get("tweet_count", 0)

    created_at = (
        datetime.fromisoformat(user["created_at"].replace("Z", "+00:00"))
        if "created_at" in user
        else datetime.utcnow()
    )
    age_days = max((datetime.utcnow() - created_at).days, 1)

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
    return user.data, []


# ----------------------------
# Train model
# ----------------------------
def train_model(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["is_bot"])
    y = df["is_bot"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}")


# ----------------------------
# Predict mode
# ----------------------------
def predict_user(username=None, json_path=None):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run training first with --train path/to/csv")

    model = joblib.load(MODEL_PATH)

    if username:
        user, _ = fetch_live(username)
    elif json_path:
        with open(json_path, "r") as f:
            data = json.load(f)
        user = data["user"]
    else:
        raise ValueError("Provide either username or json_path.")

    features = extract_features_from_user(user)
    X = np.array([list(features.values())])
    proba = model.predict_proba(X)[0][1]
    print(f"Bot likelihood for @{username or json_path}: {proba*100:.1f}%")
    return proba


# ----------------------------
# Main CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bot detector ML version.")
    parser.add_argument("--train", help="Path to labeled CSV for training")
    parser.add_argument("--username", help="X handle for prediction (API mode)")
    parser.add_argument("--json", help="Offline JSON file for prediction")
    args = parser.parse_args()

    if args.train:
        train_model(args.train)
    else:
        predict_user(username=args.username, json_path=args.json)
