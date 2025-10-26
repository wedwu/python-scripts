# Python Scripts for X: Bot Detector

This script does not label accounts, it only computes a bot-likelihood score for analysis or research.

### Tech Stack
- **Python**

Mode 1: Uses the official X API via Tweepy for live lookups.

Mode 2: Works offline with a JSON file of account data or tweets you already have.

# Instrunctions

### Option 1 (Live API):
```bash
export X_BEARER_TOKEN="YOUR_TWITTER_BEARER_TOKEN"
python bot_detector.py --username elonmusk
```

### Option 2 (Offline JSON):
```bash
python bot_detector.py --json sample_user.json
```

### JSON:
```bash
{
  "user": {
    "created_at": "2020-06-01T00:00:00Z",
    "description": "Just another user",
    "verified": false,
    "profile_image_url": "http://pbs.twimg.com/profile_images/123/default_profile_normal.png",
    "public_metrics": {"followers_count": 10, "following_count": 100, "tweet_count": 500}
  },
  "tweets": [
    {"created_at": "2025-10-24T12:00:00Z", "text": "Check out my website http://example.com"},
    {"created_at": "2025-10-24T11:00:00Z", "text": "Another tweet"}
  ]
}
```

### ⚠️ Notes

- **This script does not label accounts, it only computes a bot-likelihood score for analysis or research.**
- **You must comply with X’s Developer Policy — no scraping or automated judgment without consent.**
- **The heuristics are intentionally simple; you can tune or replace them with ML models.**

~~~~
~~~~

# Python Scripts for X: Bot Detector ML (Machine Learning)

### Tech Stack

- **Python**

### This version will:

- **Read a CSV file of known accounts with features and labels.**
- **Train a model (Random Forest).**
- **Save it for reuse.**
- **Predict bot likelihood for new accounts (via API or offline JSON).**

### 1. Prepare your labeled data

You’ll need a CSV like training_data.csv with features and a label column is_bot:

```bash
followers,following,tweet_count,bio_length,has_profile_pic,is_verified,account_age_days,follow_ratio,tweets_per_day,is_bot
10,200,500,5,0,0,100,0.05,5,1
10000,5000,3000,100,1,1,2000,2.0,1.5,0
500,1000,20000,15,1,0,1000,0.5,20,1
```

(1 = bot, 0 = human)

```bash
python bot_detector_ml.py --train training_data.csv
```

```bash
export X_BEARER_TOKEN="YOUR_TWITTER_BEARER_TOKEN"
python bot_detector_ml.py --username elonmusk
```

```bash
python bot_detector_ml.py --json sample_user.json
```

```bash
              precision    recall  f1-score   support
           0       0.93      0.95      0.94       200
           1       0.91      0.89      0.90       150
    accuracy                           0.93       350
Bot likelihood for @someaccount: 78.2%
```

### 💡Tips for better accuracy

- **Add more behavioral features: time intervals, URL frequency, emoji ratio, etc.**
- **Try more models: XGBoost, Gradient Boosting, or LightGBM.**
- **Keep retraining — bot behaviors evolve over time.**




