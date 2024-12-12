# PGA Tour Data Analysis Script 🏌️‍♂️

## Overview
This script analyzes PGA Tour golf tournament data from 2015-2022, providing insights into tournament winners and performance trends.

## Code Structure

### 1. Data Loading and Setup
```python
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../input/pga-tour-golf-data-20152022/ASA All PGA Raw Data - Tourn Level.csv')
```
Loads the PGA Tour dataset using pandas and sets up matplotlib for visualization.

### 2. Dataset Information
The dataset contains:
- Tournament results from 2015-2022
- Player information and performance
- Tournament details
- 36,864 rows × 37 columns of data

### 3. Key Features
- **Tournament Winners Analysis**: Filters and displays recent tournament winners
- **Performance Tracking**: Analyzes player positions and tournament outcomes
- **Data Visualization**: Creates plots to visualize trends and statistics

### 4. Main Functions

#### Tournament Winners
```python
recent_winners = df.loc[df['pos'] == 1][['Player_initial_last', 'tournament name', 'date']]
print("Recent Tournament Winners:")
print(recent_winners.head(25))
```
Extracts and displays information about tournament winners.

#### Data Analysis
```python
print(f"Dataset shape: {df.shape}")
print("Missing values in each column:")
print(df.isnull().sum())
print("Summary statistics:")
print(df.describe())
```
Performs shape analysis, checks for missing values, and provides statistical summaries.

#### Data Visualization
```python
wins_by_player = df[df['pos'] == 1]['Player_initial_last'].value_counts()
plt.figure(figsize=(12, 6))
wins_by_player.head(10).plot(kind='bar')
plt.title('Top 10 Players with Most Tournament Wins (2015-2022)')
plt.xlabel('Player')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)
plt.show()

df['date'] = pd.to_datetime(df['date'])
tournaments_per_year = df.groupby(df['date'].dt.year)['tournament name'].nunique()
plt.figure(figsize=(10, 6))
tournaments_per_year.plot(kind='line', marker='o')
plt.title('Number of Unique Tournaments per Year (2015-2022)')
plt.xlabel('Year')
plt.ylabel('Number of Tournaments')
plt.grid()
plt.show()
```
Creates plots to visualize tournament data and trends.

## Output
The script generates:
- List of recent tournament winners
- Statistical summaries
- Visualizations of tournament data

## Dependencies
- pandas: Data manipulation and analysis
- matplotlib: Data visualization

## Usage
Run the script to analyze PGA Tour tournament data and generate insights about player performance and tournament trends.