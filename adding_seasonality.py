
import pandas as pd
def add_seasonality(df):
    # Extracting time features
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month

    # Mapping months to seasons
    season_mapping = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                      6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
    df['season'] = df['month'].map(season_mapping)

    # Encoding seasons into numerical values
    season_to_numerical = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}
    df['season_enc'] = df['season'].map(season_to_numerical)

    return df



