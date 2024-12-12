import sqlite3
import pandas as pd
import os
from datetime import datetime

# Set paths
BASE_DIR = '/Users/kevinklatman/Development/Code/ai'
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'pga_predictions.db')

def create_tables(conn):
    """Create all necessary tables in the database."""
    
    # Create tournaments table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS tournaments (
        tournament_id INTEGER PRIMARY KEY,
        tournament_name TEXT,
        date DATE,
        course TEXT,
        purse REAL,
        season INTEGER
    )
    ''')
    
    # Create players table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS players (
        player_id INTEGER PRIMARY KEY,
        player_name TEXT
    )
    ''')
    
    # Create historical_stats table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS historical_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tournament_id INTEGER,
        player_id INTEGER,
        sg_total_last_3 REAL,
        sg_total_last_5 REAL,
        sg_total_last_10 REAL,
        sg_t2g_last_3 REAL,
        sg_t2g_last_5 REAL,
        sg_t2g_last_10 REAL,
        sg_putt_last_3 REAL,
        sg_putt_last_5 REAL,
        sg_putt_last_10 REAL,
        sg_arg_last_3 REAL,
        sg_arg_last_5 REAL,
        sg_arg_last_10 REAL,
        sg_app_last_3 REAL,
        sg_app_last_5 REAL,
        sg_app_last_10 REAL,
        sg_ott_last_3 REAL,
        sg_ott_last_5 REAL,
        sg_ott_last_10 REAL,
        made_cut_last_3 REAL,
        made_cut_last_5 REAL,
        made_cut_last_10 REAL,
        FOREIGN KEY (tournament_id) REFERENCES tournaments(tournament_id),
        FOREIGN KEY (player_id) REFERENCES players(player_id)
    )
    ''')
    
    # Create predictions table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tournament_id INTEGER,
        player_id INTEGER,
        player TEXT,
        date DATE,
        predicted_probability REAL,
        threshold_0_3 INTEGER,
        threshold_0_4 INTEGER,
        threshold_0_5 INTEGER,
        threshold_0_6 INTEGER,
        threshold_0_7 INTEGER,
        actual_top_10 INTEGER,
        FOREIGN KEY (tournament_id) REFERENCES tournaments(tournament_id),
        FOREIGN KEY (player_id) REFERENCES players(player_id)
    )
    ''')
    
    conn.commit()

def load_data_to_db(conn, historical_df, predictions_df):
    """Load data from DataFrames into the database."""
    
    # Load tournaments
    tournaments_df = historical_df[['tournament_id', 'tournament_name', 'date', 'course', 'purse', 'season']].drop_duplicates()
    tournaments_df.to_sql('tournaments', conn, if_exists='append', index=False)
    
    # Load players
    players_df = historical_df[['player_id', 'player']].drop_duplicates()
    players_df.columns = ['player_id', 'player_name']
    players_df.to_sql('players', conn, if_exists='append', index=False)
    
    # Load historical stats
    stats_cols = [col for col in historical_df.columns if '_last_' in col]
    historical_stats_df = historical_df[['tournament_id', 'player_id'] + stats_cols]
    historical_stats_df.to_sql('historical_stats', conn, if_exists='append', index=False)
    
    # Load predictions
    threshold_cols = [f'threshold_{str(t).replace(".", "_")}' for t in [0.3, 0.4, 0.5, 0.6, 0.7]]
    
    # Merge predictions with historical data to get player_id
    predictions_df = pd.merge(
        predictions_df,
        historical_df[['tournament_id', 'player', 'player_id']].drop_duplicates(),
        on=['tournament_id', 'player'],
        how='left'
    )
    
    # Convert predictions to thresholds
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        col_name = f'threshold_{str(threshold).replace(".", "_")}'
        predictions_df[col_name] = (predictions_df['predicted_probability'] >= threshold).astype(int)
    
    # Prepare predictions DataFrame for database
    predictions_for_db = predictions_df[['tournament_id', 'player_id', 'player', 'date', 'predicted_probability'] + 
                                      threshold_cols + ['actual_top_10']]
    
    predictions_for_db.to_sql('predictions', conn, if_exists='append', index=False)
    
    conn.commit()

def create_views(conn):
    """Create useful views in the database."""
    
    # View for prediction accuracy by threshold
    conn.execute('''
    CREATE VIEW IF NOT EXISTS prediction_accuracy AS
    SELECT 
        COUNT(*) as total_predictions,
        SUM(CASE WHEN threshold_0_7 = 1 AND actual_top_10 = 1 THEN 1 ELSE 0 END) as correct_0_7,
        SUM(CASE WHEN threshold_0_6 = 1 AND actual_top_10 = 1 THEN 1 ELSE 0 END) as correct_0_6,
        SUM(CASE WHEN threshold_0_5 = 1 AND actual_top_10 = 1 THEN 1 ELSE 0 END) as correct_0_5,
        SUM(CASE WHEN threshold_0_4 = 1 AND actual_top_10 = 1 THEN 1 ELSE 0 END) as correct_0_4,
        SUM(CASE WHEN threshold_0_3 = 1 AND actual_top_10 = 1 THEN 1 ELSE 0 END) as correct_0_3,
        SUM(threshold_0_7) as total_0_7,
        SUM(threshold_0_6) as total_0_6,
        SUM(threshold_0_5) as total_0_5,
        SUM(threshold_0_4) as total_0_4,
        SUM(threshold_0_3) as total_0_3
    FROM predictions
    ''')
    
    # View for player performance
    conn.execute('''
    CREATE VIEW IF NOT EXISTS player_performance AS
    SELECT 
        p.player_name,
        COUNT(*) as total_tournaments,
        SUM(pred.actual_top_10) as actual_top_10s,
        AVG(pred.predicted_probability) as avg_prediction,
        SUM(pred.threshold_0_7) as predictions_0_7,
        SUM(CASE WHEN pred.threshold_0_7 = 1 AND pred.actual_top_10 = 1 THEN 1 ELSE 0 END) as correct_0_7
    FROM predictions pred
    JOIN players p ON pred.player_id = p.player_id
    GROUP BY p.player_id, p.player_name
    ''')
    
    conn.commit()

def main():
    # Delete existing database if it exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Deleted existing database: {DB_PATH}")
    
    # Create database connection
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Create tables
        print("Creating database tables...")
        create_tables(conn)
        
        # Load historical data
        print("Loading historical data...")
        historical_df = pd.read_csv(os.path.join(DATA_DIR, 'pga_tour_historical.csv'))
        
        # Run predictions
        print("Running model predictions...")
        import logistic_regression_model as lrm
        X_train, X_test, y_train, y_test, feature_cols, tournament_info_test = lrm.load_and_prepare_data()
        model, y_pred, y_pred_proba, _ = lrm.train_and_evaluate_model(X_train, X_test, y_train, y_test)
        predictions_df = lrm.analyze_predictions_by_tournament(tournament_info_test, y_pred, y_pred_proba, model, feature_cols)
        
        # Load data into database
        print("Loading data into database...")
        load_data_to_db(conn, historical_df, predictions_df)
        
        # Create views
        print("Creating database views...")
        create_views(conn)
        
        print(f"\nDatabase created successfully at: {DB_PATH}")
        
        # Show some sample queries
        print("\nSample Queries:")
        
        print("\n1. Overall prediction accuracy:")
        accuracy_df = pd.read_sql('''
            SELECT 
                ROUND(CAST(correct_0_7 AS FLOAT) / total_0_7 * 100, 1) as accuracy_0_7,
                ROUND(CAST(correct_0_6 AS FLOAT) / total_0_6 * 100, 1) as accuracy_0_6,
                ROUND(CAST(correct_0_5 AS FLOAT) / total_0_5 * 100, 1) as accuracy_0_5
            FROM prediction_accuracy
        ''', conn)
        print(accuracy_df)
        
        print("\n2. Top 10 players by prediction accuracy (0.7 threshold):")
        top_players_df = pd.read_sql('''
            SELECT 
                player_name,
                total_tournaments,
                predictions_0_7,
                correct_0_7,
                ROUND(CAST(correct_0_7 AS FLOAT) / predictions_0_7 * 100, 1) as accuracy
            FROM player_performance
            WHERE predictions_0_7 >= 5
            ORDER BY accuracy DESC
            LIMIT 10
        ''', conn)
        print(top_players_df)
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()
