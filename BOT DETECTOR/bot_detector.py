# // bot_detector.py

"""
Bot Detector for X (Twitter)
Supports:
 - Live API mode (requires bearer token)
 - Offline mode (reads from JSON file)
"""

import os
import json
from datetime import datetime, timezone
from statistics import mean

# --- Optional Tweepy import for API mode ---
try:
    import tweepy
except ImportError:
    tweepy = None


# ----------------------------
# Configuration
# ----------------------------
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")  # or paste your key here
USE_API = bool(BEARER_TOKEN)  # automatically switch based on availability


# ----------------------------
# Feature extraction functions
# ----------------------------
def extract_features_from_user(user, tweets=None):
    """Extract heuristic features from a user object (API or JSON)"""
    metrics = user.get("public_metrics", {})
    followers = metrics.get("followers_count", 0)
    following = metrics.get("following_count", 0)
    tweet_count = metrics.get("tweet_count", 0)

    created_at = (
        datetime.fromisoformat(user["created_at"].replace("Z", "+00:00"))
        if "created_at" in user
        else datetime.utcnow()
    )
    # age_days = max((datetime.utcnow() - created_at).days, 1)
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

    # Optional tweet-level stats
    if tweets:
        urls = sum("http" in (t.get("text") or "") for t in tweets)
        hashtags = sum("#" in (t.get("text") or "") for t in tweets)
        times = [datetime.fromisoformat(t["created_at"].replace("Z", "+00:00")) for t in tweets]
        if len(times) > 1:
            deltas = [(times[i] - times[i+1]).total_seconds() for i in range(len(times)-1)]
            avg_interval = mean(abs(d) for d in deltas)
        else:
            avg_interval = None
        features.update({
            "url_ratio": urls / len(tweets),
            "hashtag_ratio": hashtags / len(tweets),
            "avg_post_interval_sec": avg_interval or 0,
        })

    return features


# ----------------------------
# Simple rule-based scoring
# ----------------------------
def bot_likelihood(features):
    score = 0
    if features["tweets_per_day"] > 100: score += 1
    if features["follow_ratio"] < 0.1: score += 1
    if features["bio_length"] < 10: score += 1
    if not features["has_profile_pic"]: score += 1
    if not features["is_verified"]: score += 1
    if features.get("url_ratio", 0) > 0.5: score += 1

    likelihood = min(score / 6, 1.0)
    return likelihood


# ----------------------------
# Mode 1: Live API Mode
# ----------------------------
def fetch_live(username):
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    user = client.get_user(
        username=username,
        user_fields=["created_at", "description", "public_metrics", "verified", "profile_image_url"]
    ).data


    # tweets = client.get_users_tweets(id=user.id, max_results=50).data or []
    tweets = client.get_users_tweets(id=user.id, max_results=10).data or []
    user_json = user.data
    tweet_jsons = [t.data for t in tweets]
    return user_json, tweet_jsons


# ----------------------------
# Mode 2: Offline JSON Mode
# ----------------------------
def load_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["user"], data.get("tweets", [])


# ----------------------------
# Main entry point
# ----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detect likely bots on X.")
    parser.add_argument("--username", help="X handle (no @) for live mode")
    parser.add_argument("--json", help="Path to local JSON for offline mode")
    args = parser.parse_args()

    if USE_API and args.username:
        print(f"Fetching data for @{args.username} via API...")
        user, tweets = fetch_live(args.username)
    elif args.json:
        print(f"Loading from {args.json}...")
        user, tweets = load_from_json(args.json)
    else:
        print("Provide either --username (with API key) or --json path.")
        return

    features = extract_features_from_user(user, tweets)
    likelihood = bot_likelihood(features)

    print("\n--- Bot Likelihood Report ---")
    for k, v in features.items():
        print(f"{k:25}: {v}")
    print(f"\nBot likelihood score: {likelihood*100:.1f}%")
    print("(>70% = likely bot, <30% = likely human)")


if __name__ == "__main__":
    main()
