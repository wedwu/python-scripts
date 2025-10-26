Instructions.md

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

This script does not label accounts, it only computes a bot-likelihood score for analysis or research.

You must comply with X’s Developer Policy — no scraping or automated judgment without consent.

The heuristics are intentionally simple; you can tune or replace them with ML models.