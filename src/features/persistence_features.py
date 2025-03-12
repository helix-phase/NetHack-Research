import numpy as np
import pandas as pd
from tabulate import tabulate


def categorize_deaths(data):
    """
    Categorizes deaths into predefined categories.
    """
    death_map = {
        'quit': 'voluntary', 'escaped': 'success', 'ascended': 'success',
        'killed by': 'monster', 'petrified by': 'monster', 'poisoned by': 'monster',
        'turned to': 'environment', 'drowned': 'environment', 'burned by': 'environment',
        'fell': 'environment', 'slipped': 'environment', 'zapped': 'self',
        'choked': 'self'
    }

    data['death_category'] = 'other'
    for keyword, category in death_map.items():
        data.loc[data['death'].str.contains(keyword, case=False, na=False), 'death_category'] = category

    return data


def track_failure_streaks(data):
    """
    Tracks failure streaks and assigns unique streak IDs.
    Also, captures the last failure streak before ascension properly.
    """
    # Identify failure events
    data['failure'] = (data['death_category'] != 'success').astype(int)

    # Identify new failure streaks
    data['new_failure_streak'] = data.groupby('name')['failure'].diff().fillna(0) > 0
    data['failure_streak_id'] = data.groupby('name')['new_failure_streak'].cumsum()

    # Compute failure streak lengths
    failure_streak_lengths = data[data['failure'] == 1].groupby(['name', 'failure_streak_id']).size().reset_index(
        name='failure_streak_length')

    # Merge failure streak lengths
    data = data.merge(failure_streak_lengths, on=['name', 'failure_streak_id'], how='left')
    data['failure_streak_length'] = data['failure_streak_length'].fillna(0)

    # Identify ascension events
    data['ascended'] = (data['death_category'] == 'success') & (data['death'] == 'ascended')

    # Track the last failure streak before ascension per player
    data['failure_streak_before_ascension'] = 0
    for player in data['name'].unique():
        player_data = data[data['name'] == player]
        last_failure_streak = 0
        for idx, row in player_data.iterrows():
            if row['ascended']:  # If the player ascended, assign the last failure streak
                data.at[idx, 'failure_streak_before_ascension'] = last_failure_streak
            if row['failure']:  # Update the last failure streak if it’s a failure
                last_failure_streak = row['failure_streak_length']

    return data


def track_success_after_failure(data):
    data['rolling_failures'] = data.groupby('name')['failure'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum())
    data['success_after_failure'] = (data['death_category'] == 'success') & (data['rolling_failures'] > 3)
    data['rolling_success_after_failure'] = data.groupby('name')['success_after_failure'].transform(
        lambda x: x.rolling(5, min_periods=1).sum())
    data['rolling_success_after_failure'] /= data.groupby('name')['gameid'].transform('count')

    return data


def compute_persistence_score(data):
    data['persistence_score'] = np.sqrt(data['failure_streak_before_ascension'])
    data.loc[data['success_after_failure'], 'persistence_score'] += 0.5

    max_persistence = data['persistence_score'].max()
    if max_persistence > 0:
        data['persistence_score'] = data['persistence_score'] / max_persistence

    return data


def process_persistence_features(data):
    """
    Applies all persistence-related feature transformations to the dataset.
    """
    data = categorize_deaths(data)
    data = track_failure_streaks(data)
    data = track_success_after_failure(data)
    data = compute_persistence_score(data)

    return data


if __name__ == "__main__":
    csv_path = "/code/NetHack-Research/data/processed/new_full_data.csv"

    df = pd.read_csv(csv_path)

    data = process_persistence_features(df)

    print(tabulate(data[['name', 'failure_streak_before_ascension', 'persistence_score']].head(20),
                   headers='keys', tablefmt='pretty'))

