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

## PGA Tour Statistics Explained

### Overview
This document explains the various statistics and metrics used in PGA Tour data analysis.

### Fantasy Points
- **FDP (FanDuel Points)**: Fantasy points awarded on FanDuel platform
- **DKP (DraftKings Points)**: Fantasy points awarded on DraftKings platform
- **SDP (SuperDraft Points)**: Fantasy points awarded on SuperDraft platform

Each platform calculates points differently based on:
- Tournament finish position
- Birdies, eagles, and bogeys
- Streak bonuses
- Tournament importance

### Strokes Gained Statistics
Strokes Gained measures how many shots a player gains or loses compared to the field average in different aspects of the game.

- **sg_total**: Overall strokes gained per round vs. field average
- **sg_t2g (Tee-to-Green)**: Combined strokes gained in all aspects except putting
- **sg_ott (Off-the-Tee)**: Strokes gained on tee shots
- **sg_app (Approach)**: Strokes gained on approach shots
- **sg_arg (Around-the-Green)**: Strokes gained on shots around the green
- **sg_putt (Putting)**: Strokes gained on putts

Example: A sg_total of +2.0 means a player performed 2 strokes better than the field average per round

### Tournament Performance
- **pos**: Final position in tournament
- **n_rounds**: Number of rounds played in tournament
- **made_cut**: Boolean indicating if player made the cut (1=yes, 0=no)
- **strokes**: Total number of strokes taken
- **hole_par**: Par score for the hole
- **no_cut**: Tournament has no cut (1=true, 0=false)

### Fantasy Scoring Categories
Each statistic has three variants (FDP, DKP, SDP) representing different fantasy platforms:

- **finish_[FDP/DKP/SDP]**: Points awarded for tournament finish position
- **total_[FDP/DKP/SDP]**: Total fantasy points earned
- **streak_[FDP/DKP/SDP]**: Bonus points for consecutive birdies/eagles
- **hole_[FDP/DKP/SDP]**: Points earned per hole

### Tournament Information
- **purse**: Tournament prize money
- **season**: PGA Tour season
- **player_id**: Unique identifier for each player
- **tournament_id**: Unique identifier for each tournament

### Target Variable
- **is_top_10**: Binary variable indicating if player finished in top 10 (1=yes, 0=no)

### Statistical Significance
Based on correlation analysis:

Strong Predictors (|correlation| > 0.5):
- Fantasy points finish metrics (0.83-0.89)
- Total fantasy points (0.50-0.54)

Moderate Predictors (0.3 < |correlation| < 0.5):
- Streak and hole metrics (0.37-0.45)
- sg_total (0.41)
- sg_t2g (0.33)
- pos (-0.31)

Weak Predictors (|correlation| < 0.3):
- Individual strokes gained metrics
- Tournament characteristics
- Player characteristics
- Temporal features

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