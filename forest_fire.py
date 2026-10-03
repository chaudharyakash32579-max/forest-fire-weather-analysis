# Forest Fire & Weather Analysis
# BCA 1st Year Project

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------
# 1. Load Dataset
# ---------------------------------

df = pd.read_csv("forestfires.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ---------------------------------
# 2. Dataset Information
# ---------------------------------

print("\nDataset Information:")
print(df.info())

print("\nColumn Names:")
print(df.columns.tolist())


# ---------------------------------
# 3. Check Missing Values
# ---------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# ---------------------------------
# 4. Check Duplicate Values
# ---------------------------------

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())


# ---------------------------------
# 5. Descriptive Statistics
# ---------------------------------

print("\nDescriptive Statistics:")
print(df.describe())


# ---------------------------------
# 6. Univariate Analysis
# ---------------------------------

# Distribution of burned area
plt.figure(figsize=(8, 5))
plt.hist(df["area"], bins=30)
plt.xlabel("Burned Area")
plt.ylabel("Number of Observations")
plt.title("Distribution of Forest Fire Area")
plt.show()


# ---------------------------------
# 7. Monthly Fire Analysis
# ---------------------------------

monthly_fires = df.groupby("month")["area"].sum().sort_values(ascending=False)

print("\nTotal Burned Area by Month:")
print(monthly_fires)

plt.figure(figsize=(8, 5))
monthly_fires.plot(kind="bar")
plt.xlabel("Month")
plt.ylabel("Total Burned Area")
plt.title("Total Forest Fire Area by Month")
plt.xticks(rotation=45)
plt.show()


# ---------------------------------
# 8. Bivariate Analysis
# ---------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(df["temp"], df["area"])
plt.xlabel("Temperature")
plt.ylabel("Burned Area")
plt.title("Temperature vs Burned Area")
plt.show()


# ---------------------------------
# 9. Correlation Analysis
# ---------------------------------

numeric_columns = df.select_dtypes(include="number")

correlation = numeric_columns.corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(10, 7))
plt.imshow(correlation, cmap="coolwarm")
plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# ---------------------------------
# 10. Weather Analysis
# ---------------------------------

print("\nAverage Weather Conditions:")

print("Average Temperature:",
      df["temp"].mean())

print("Average Relative Humidity:",
      df["RH"].mean())

print("Average Wind:",
      df["wind"].mean())

print("Average Rain:",
      df["rain"].mean())


# ---------------------------------
# 11. Grouping and Aggregation
# ---------------------------------

monthly_weather = df.groupby("month").agg({
    "temp": "mean",
    "RH": "mean",
    "wind": "mean",
    "rain": "mean",
    "area": "sum"
})

print("\nMonthly Weather and Fire Analysis:")
print(monthly_weather)


# ---------------------------------
# 12. Key Findings
# ---------------------------------

print("\nKey Findings:")
print("1. The dataset contains forest fire and weather information.")
print("2. Temperature, humidity, wind and rain are analyzed.")
print("3. Fire area varies considerably between observations.")
print("4. Monthly grouping helps identify fire patterns.")
print("5. Correlation analysis helps study relationships between weather and burned area.")


# ---------------------------------
# End of Project
# ---------------------------------

print("\nForest Fire & Weather Analysis Completed!")
plt.show()


