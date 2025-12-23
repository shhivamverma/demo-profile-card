import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("netflix_titles.csv")

df.drop_duplicates(subset="show_id", inplace=True)

df["country"] = df["country"].fillna("Unknown")
df["director"] = df["director"].fillna("Not Available")


df["Is_Recent"] = df["release_year"].apply(lambda x: 1 if x >= 2015 else 0)

df["duration_minutes"] = pd.NA
df["seasons"] = pd.NA

df.loc[df["type"] == "Movie", "duration_minutes"] = (
    df.loc[df["type"] == "Movie", "duration"]
    .str.replace(" min", "", regex=False)
)

df.loc[df["type"] == "TV Show", "seasons"] = (
    df.loc[df["type"] == "TV Show", "duration"]
    .str.replace(" Seasons", "", regex=False)
    .str.replace(" Season", "", regex=False)
)

df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")
df["seasons"] = pd.to_numeric(df["seasons"], errors="coerce")

plt.figure()
sns.countplot(x="type", data=df, color="#4C72B0")
plt.title("Count of Movies vs TV Shows on Netflix")
plt.xlabel("Content Type")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.hist(df["release_year"], bins=25, color="#DD8452")
plt.title("Distribution of Release Years")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.show()


top_countries = df["country"].value_counts().head(10)

plt.figure()
top_countries.plot(kind="bar", color="#55A868")
plt.title("Top 10 Countries by Number of Netflix Titles")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.show()

movies_df = df[df["type"] == "Movie"].dropna(subset=["duration_minutes"])

plt.figure()
sns.boxplot(
    x="Is_Recent",
    y="duration_minutes",
    data=movies_df,
    color="#C44E52"
)
plt.title("Movie Duration: Recent vs Older Movies")
plt.xlabel("Is Recent (1 = Recent, 0 = Older)")
plt.ylabel("Duration (Minutes)")
plt.show()


numeric_df = df[
    ["release_year", "duration_minutes", "seasons", "Is_Recent"]
].dropna()

plt.figure()
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="Blues"
)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()