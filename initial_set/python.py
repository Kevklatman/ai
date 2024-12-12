import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./ASA All PGA Raw Data - PGA Tour Golf Data - (2015-2022).csv')

# Display the shape of the dataset
print(f"Dataset shape: {df.shape}")

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Get summary statistics
print("Summary statistics:")
print(df.describe())

# Analyze Recent Tournament Winners
recent_winners = df.loc[df['pos'] == 1][['Player_initial_last', 'tournament name', 'date']]
print("Recent Tournament Winners:")
print(recent_winners.head(25))

# Visualize the number of wins by player
wins_by_player = df[df['pos'] == 1]['Player_initial_last'].value_counts()

plt.figure(figsize=(12, 6))
wins_by_player.head(10).plot(kind='bar')
plt.title('Top 10 Players with Most Tournament Wins (2015-2022)')
plt.xlabel('Player')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)
plt.show()

# Convert date to datetime format for trend analysis
df['date'] = pd.to_datetime(df['date'])

# Group by year and count tournaments
tournaments_per_year = df.groupby(df['date'].dt.year)['tournament name'].nunique()

plt.figure(figsize=(10, 6))
tournaments_per_year.plot(kind='line', marker='o')
plt.title('Number of Unique Tournaments per Year (2015-2022)')
plt.xlabel('Year')
plt.ylabel('Number of Tournaments')
plt.grid()
plt.show()