# ============================================================
# INSTAGRAM DATA ANALYSIS - ALFIDO TECH
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. FOLDER SETUP
# ------------------------------------------------------------

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD ALL 7 CSV FILES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INSTAGRAM DATA ANALYSIS - ALFIDO TECH")
print("=" * 60)

print("\nLoading datasets...")

users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
photos = pd.read_csv(os.path.join(DATA_DIR, "photos.csv"))
photo_tags = pd.read_csv(os.path.join(DATA_DIR, "photo_tags.csv"))
tags = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))
likes = pd.read_csv(os.path.join(DATA_DIR, "likes.csv"))
follows = pd.read_csv(os.path.join(DATA_DIR, "follows.csv"))
comments = pd.read_csv(os.path.join(DATA_DIR, "comments.csv"))

print("✓ users.csv loaded")
print("✓ photos.csv loaded")
print("✓ photo_tags.csv loaded")
print("✓ tags.csv loaded")
print("✓ likes.csv loaded")
print("✓ follows.csv loaded")
print("✓ comments.csv loaded")


# ------------------------------------------------------------
# 3. BASIC DATA INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

datasets = {
    "Users": users,
    "Photos": photos,
    "Photo Tags": photo_tags,
    "Tags": tags,
    "Likes": likes,
    "Follows": follows,
    "Comments": comments
}

for name, df in datasets.items():
    print(f"{name:15} : {df.shape[0]:,} rows × {df.shape[1]} columns")


# ------------------------------------------------------------
# 4. DATA CLEANING
# ------------------------------------------------------------

print("\nCleaning data...")

# Remove duplicate records
for name, df in datasets.items():
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)

    if before != after:
        print(f"{name}: removed {before - after} duplicates")


# ------------------------------------------------------------
# 5. DATE/TIME PARSING
# ------------------------------------------------------------

print("\nParsing dates...")

users["created time"] = pd.to_datetime(
    users["created time"],
    dayfirst=True,
    errors="coerce"
)

photos["created dat"] = pd.to_datetime(
    photos["created dat"],
    dayfirst=True,
    errors="coerce"
)

likes["created time"] = pd.to_datetime(
    likes["created time"],
    dayfirst=True,
    errors="coerce"
)

comments["created Timestamp"] = pd.to_datetime(
    comments["created Timestamp"],
    dayfirst=True,
    errors="coerce"
)

follows["created time"] = pd.to_datetime(
    follows["created time"],
    dayfirst=True,
    errors="coerce"
)

tags["created time"] = pd.to_datetime(
    tags["created time"],
    dayfirst=True,
    errors="coerce"
)


# ------------------------------------------------------------
# 6. CREATE POSTING TIME FEATURES
# ------------------------------------------------------------

photos["posting_hour"] = photos["created dat"].dt.hour
photos["posting_day"] = photos["created dat"].dt.day_name()
photos["posting_date"] = photos["created dat"].dt.date


# ------------------------------------------------------------
# 7. LIKE COUNT PER PHOTO
# ------------------------------------------------------------

like_counts = (
    likes.groupby("photo")
    .size()
    .reset_index(name="likes")
)

photos = photos.merge(
    like_counts,
    left_on="id",
    right_on="photo",
    how="left"
)

photos["likes"] = photos["likes"].fillna(0)

photos.drop(
    columns=["photo"],
    inplace=True,
    errors="ignore"
)


# ------------------------------------------------------------
# 8. COMMENT COUNT PER PHOTO
# ------------------------------------------------------------

comment_counts = (
    comments.groupby("Photo id")
    .size()
    .reset_index(name="comments")
)

photos = photos.merge(
    comment_counts,
    left_on="id",
    right_on="Photo id",
    how="left"
)

photos["comments"] = photos["comments"].fillna(0)

photos.drop(
    columns=["Photo id"],
    inplace=True,
    errors="ignore"
)


# ------------------------------------------------------------
# 9. FOLLOWER COUNTS
# ------------------------------------------------------------

follower_counts = (
    follows.groupby("followee ")
    .size()
    .reset_index(name="followers")
)

photos = photos.merge(
    follower_counts,
    left_on="user ID",
    right_on="followee ",
    how="left"
)

photos["followers"] = photos["followers"].fillna(0)

photos.drop(
    columns=["followee "],
    inplace=True,
    errors="ignore"
)


# ------------------------------------------------------------
# 10. ENGAGEMENT METRICS
# ------------------------------------------------------------

photos["total_engagement"] = (
    photos["likes"] +
    photos["comments"]
)

photos["engagement_rate"] = np.where(
    photos["followers"] > 0,
    (photos["total_engagement"] / photos["followers"]) * 100,
    0
)


# ------------------------------------------------------------
# 11. SAVE POST LEVEL ANALYSIS
# ------------------------------------------------------------

photos.to_csv(
    os.path.join(CSV_DIR, "post_level_analysis.csv"),
    index=False
)

print("\n✓ Post-level analysis saved")


# ------------------------------------------------------------
# 12. CONTENT TYPE ANALYSIS
# ------------------------------------------------------------

content_analysis = (
    photos.groupby("photo type")
    .agg(
        posts=("id", "count"),
        total_likes=("likes", "sum"),
        avg_likes=("likes", "mean"),
        total_comments=("comments", "sum"),
        avg_comments=("comments", "mean"),
        avg_engagement=("total_engagement", "mean"),
        avg_engagement_rate=("engagement_rate", "mean")
    )
    .reset_index()
)

content_analysis = content_analysis.sort_values(
    "avg_engagement",
    ascending=False
)

content_analysis.to_csv(
    os.path.join(CSV_DIR, "content_type_analysis.csv"),
    index=False
)

print("\n" + "=" * 60)
print("CONTENT TYPE ANALYSIS")
print("=" * 60)

print(content_analysis.to_string(index=False))


# ------------------------------------------------------------
# 13. BEST CONTENT TYPE
# ------------------------------------------------------------

best_content = content_analysis.iloc[0]["photo type"]
best_content_score = content_analysis.iloc[0]["avg_engagement"]

print(
    f"\n🏆 Best content type: {best_content}"
)

print(
    f"Average engagement: {best_content_score:.2f}"
)


# ------------------------------------------------------------
# 14. POSTING DAY ANALYSIS
# ------------------------------------------------------------

day_analysis = (
    photos.groupby("posting_day")
    .agg(
        posts=("id", "count"),
        avg_likes=("likes", "mean"),
        avg_comments=("comments", "mean"),
        avg_engagement=("total_engagement", "mean")
    )
    .reset_index()
)

day_analysis = day_analysis.sort_values(
    "avg_engagement",
    ascending=False
)

day_analysis.to_csv(
    os.path.join(CSV_DIR, "posting_day_analysis.csv"),
    index=False
)

print("\n" + "=" * 60)
print("POSTING DAY ANALYSIS")
print("=" * 60)

print(day_analysis.to_string(index=False))


# ------------------------------------------------------------
# 15. POSTING HOUR ANALYSIS
# ------------------------------------------------------------

hour_analysis = (
    photos.groupby("posting_hour")
    .agg(
        posts=("id", "count"),
        avg_likes=("likes", "mean"),
        avg_comments=("comments", "mean"),
        avg_engagement=("total_engagement", "mean")
    )
    .reset_index()
)

hour_analysis = hour_analysis.sort_values(
    "avg_engagement",
    ascending=False
)

hour_analysis.to_csv(
    os.path.join(CSV_DIR, "posting_hour_analysis.csv"),
    index=False
)

print("\n" + "=" * 60)
print("POSTING HOUR ANALYSIS")
print("=" * 60)

print(hour_analysis.to_string(index=False))


# ------------------------------------------------------------
# 16. HASHTAG ANALYSIS
# ------------------------------------------------------------

photo_tag_data = photo_tags.merge(
    tags[["id", "tag text"]],
    left_on="tag ID",
    right_on="id",
    how="left"
)

photo_tag_data = photo_tag_data.merge(
    photos[
        [
            "id",
            "likes",
            "comments",
            "total_engagement"
        ]
    ],
    left_on="photo",
    right_on="id",
    how="left"
)

hashtag_analysis = (
    photo_tag_data.groupby("tag text")
    .agg(
        usage_count=("photo", "count"),
        total_likes=("likes", "sum"),
        avg_likes=("likes", "mean"),
        total_comments=("comments", "sum"),
        avg_comments=("comments", "mean"),
        avg_engagement=("total_engagement", "mean")
    )
    .reset_index()
)

hashtag_analysis = hashtag_analysis.sort_values(
    "avg_engagement",
    ascending=False
)

hashtag_analysis.to_csv(
    os.path.join(CSV_DIR, "hashtag_analysis.csv"),
    index=False
)

print("\n" + "=" * 60)
print("TOP HASHTAGS")
print("=" * 60)

print(
    hashtag_analysis.head(10).to_string(index=False)
)


# ------------------------------------------------------------
# 17. FOLLOWER ANALYSIS
# ------------------------------------------------------------

follower_analysis = (
    follows.groupby("followee ")
    .agg(
        follower_count=("follower", "count"),
        active_followers=("is follower active", "sum")
    )
    .reset_index()
)

follower_analysis["active_follower_rate"] = np.where(
    follower_analysis["follower_count"] > 0,
    (
        follower_analysis["active_followers"] /
        follower_analysis["follower_count"]
    ) * 100,
    0
)

follower_analysis.to_csv(
    os.path.join(CSV_DIR, "follower_analysis.csv"),
    index=False
)


# ------------------------------------------------------------
# 18. LIKE TYPE ANALYSIS
# ------------------------------------------------------------

like_type_analysis = (
    likes.groupby("like type")
    .size()
    .reset_index(name="count")
)

like_type_analysis = like_type_analysis.sort_values(
    "count",
    ascending=False
)

like_type_analysis.to_csv(
    os.path.join(CSV_DIR, "like_type_analysis.csv"),
    index=False
)


# ------------------------------------------------------------
# 19. VERIFIED USER ANALYSIS
# ------------------------------------------------------------

verified_analysis = (
    users.groupby("Verified status")
    .agg(
        users=("id", "count"),
        avg_post_count=("post count", "mean")
    )
    .reset_index()
)

verified_analysis.to_csv(
    os.path.join(CSV_DIR, "verified_user_analysis.csv"),
    index=False
)


# ------------------------------------------------------------
# 20. FILTER ANALYSIS
# ------------------------------------------------------------

filter_analysis = (
    photos.groupby("Insta filter used")
    .agg(
        posts=("id", "count"),
        avg_likes=("likes", "mean"),
        avg_comments=("comments", "mean"),
        avg_engagement=("total_engagement", "mean")
    )
    .reset_index()
)

filter_analysis.to_csv(
    os.path.join(CSV_DIR, "filter_analysis.csv"),
    index=False
)


# ============================================================
# VISUALIZATIONS
# ============================================================

print("\nCreating charts...")


# ------------------------------------------------------------
# CHART 1 - CONTENT TYPE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    content_analysis["photo type"],
    content_analysis["avg_engagement"]
)

plt.title("Average Engagement by Content Type")
plt.xlabel("Content Type")
plt.ylabel("Average Engagement")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "01_content_type_engagement.png"),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# CHART 2 - CONTENT TYPE LIKES
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    content_analysis["photo type"],
    content_analysis["avg_likes"]
)

plt.title("Average Likes by Content Type")
plt.xlabel("Content Type")
plt.ylabel("Average Likes")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "02_content_type_likes.png"),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# CHART 3 - CONTENT TYPE COMMENTS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    content_analysis["photo type"],
    content_analysis["avg_comments"]
)

plt.title("Average Comments by Content Type")
plt.xlabel("Content Type")
plt.ylabel("Average Comments")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "03_content_type_comments.png"),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# CHART 4 - POSTING DAYS
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    day_analysis["posting_day"],
    day_analysis["avg_engagement"]
)

plt.title("Average Engagement by Posting Day")
plt.xlabel("Day")
plt.ylabel("Average Engagement")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "04_posting_day.png"),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# CHART 5 - POSTING HOURS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    hour_analysis["posting_hour"].astype(str),
    hour_analysis["avg_engagement"]
)

plt.title("Average Engagement by Posting Hour")
plt.xlabel("Posting Hour")
plt.ylabel("Average Engagement")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "05_posting_hour.png"),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# CHART 6 - TOP HASHTAGS
# ------------------------------------------------------------

top_hashtags = hashtag_analysis.head(10).sort_values(
    "avg_engagement"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_hashtags["tag text"],
    top_hashtags["avg_engagement"]
)

plt.title("Top Hashtags by Average Engagement")
plt.xlabel("Average Engagement")
plt.ylabel("Hashtag")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "06_top_hashtags.png"),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# CHART 7 - LIKE TYPES
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    like_type_analysis["like type"],
    like_type_analysis["count"]
)

plt.title("Like Type Distribution")
plt.xlabel("Like Type")
plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "07_like_types.png"),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# CHART 8 - FILTER ANALYSIS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    filter_analysis["Insta filter used"],
    filter_analysis["avg_engagement"]
)

plt.title("Average Engagement: Filter vs No Filter")
plt.xlabel("Instagram Filter Used")
plt.ylabel("Average Engagement")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "08_filter_engagement.png"),
    dpi=300
)

plt.close()


# ============================================================
# 21. CONTENT CALENDAR
# ============================================================

print("\nCreating content calendar...")

best_days = day_analysis.head(7)["posting_day"].tolist()

calendar_rows = []

for i, day in enumerate(best_days):

    calendar_rows.append({
        "Day": day,
        "Recommended Content": best_content,
        "Reason": "Based on observed engagement performance"
    })

content_calendar = pd.DataFrame(calendar_rows)

content_calendar.to_csv(
    os.path.join(CSV_DIR, "recommended_content_calendar.csv"),
    index=False
)


# ============================================================
# 22. FIVE STRATEGIES
# ============================================================

print("\nGenerating strategies...")

top_hashtag = (
    hashtag_analysis.iloc[0]["tag text"]
    if len(hashtag_analysis) > 0
    else "high-performing hashtags"
)

strategies = [
    f"1. Increase the use of {best_content} content because it shows the highest observed average engagement.",

    f"2. Prioritize the strongest-performing hashtags such as #{top_hashtag} while keeping hashtags relevant to the content.",

    "3. Encourage comments and conversations through questions, calls-to-action and interactive captions.",

    "4. Monitor active follower behavior and focus content on audiences that show stronger interaction.",

    "5. Use the observed engagement patterns to maintain a consistent content calendar and continuously compare performance."
]

with open(
    os.path.join(OUTPUT_DIR, "five_strategies.txt"),
    "w",
    encoding="utf-8"
) as f:

    f.write("ALFIDO TECH - 5 INSTAGRAM ENGAGEMENT STRATEGIES\n")
    f.write("=" * 60 + "\n\n")

    for strategy in strategies:
        f.write(strategy + "\n\n")


# ============================================================
# 23. ONE-PAGE STRATEGY DOCUMENT
# ============================================================

strategy_document = f"""
============================================================
INSTAGRAM DATA ANALYSIS - ALFIDO TECH
============================================================

OBJECTIVE
Analyze Instagram posts, engagement, hashtags, content types
and follower signals to create a practical engagement strategy.

KEY FINDINGS
------------

1. BEST CONTENT TYPE
{best_content}

Average engagement:
{best_content_score:.2f}

2. BEST POSTING DAY
{day_analysis.iloc[0]["posting_day"]}

3. TOP HASHTAG
#{top_hashtag}

4. TOTAL POSTS
{len(photos)}

5. TOTAL LIKES
{int(photos["likes"].sum()):,}

6. TOTAL COMMENTS
{int(photos["comments"].sum()):,}


RECOMMENDED STRATEGIES
----------------------

{chr(10).join(strategies)}


CONTENT CALENDAR
----------------

{content_calendar.to_string(index=False)}


IMPORTANT DATA LIMITATION
-------------------------

The supplied dataset contains limited variation in timestamp
information. Therefore, posting-time recommendations should be
treated as observations from the available data rather than as
a statistically conclusive optimal posting time.

============================================================
"""

with open(
    os.path.join(OUTPUT_DIR, "one_page_strategy.txt"),
    "w",
    encoding="utf-8"
) as f:

    f.write(strategy_document)


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nMain Results:")

print(f"Total users       : {len(users):,}")
print(f"Total posts       : {len(photos):,}")
print(f"Total likes       : {len(likes):,}")
print(f"Total comments    : {len(comments):,}")
print(f"Total follows     : {len(follows):,}")
print(f"Total hashtags    : {len(tags):,}")

print(f"\nBest content type : {best_content}")
print(f"Average engagement: {best_content_score:.2f}")

print("\nGenerated folders:")
print(f"✓ {OUTPUT_DIR}/charts/")
print(f"✓ {OUTPUT_DIR}/csv/")
print(f"✓ {OUTPUT_DIR}/five_strategies.txt")
print(f"✓ {OUTPUT_DIR}/one_page_strategy.txt")

print("\n🎉 Instagram analysis project completed!")