import pandas as pd
import numpy as np
from datetime import timedelta


def engineer_persistence_features(df):
    """
    Engineer features for persistence analysis in NetHack using a Latent Growth Model approach.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing NetHack gameplay data, sorted by player name and starttime

    Returns:
    --------
    pandas.DataFrame
        DataFrame with engineered persistence features
    """
    print("Engineering persistence features...")

    # Make a copy to avoid modifying the original dataframe
    data = df.copy()

    # === TEMPORAL SEQUENCE FEATURES ===

    # Ensure data is sorted properly
    data = data.sort_values(['name', 'starttime'])

    # Add game sequence number for each player
    data['game_number'] = data.groupby('name').cumcount() + 1

    # Calculate game duration in minutes
    data['game_duration_minutes'] = (pd.to_datetime(data['endtime']) -
                                     pd.to_datetime(data['starttime'])).dt.total_seconds() / 60

    # Calculate time between games (for each player)
    data['next_game_time'] = data.groupby('name')['starttime'].shift(-1)
    data['time_to_next_game_hours'] = (pd.to_datetime(data['next_game_time']) -
                                       pd.to_datetime(data['starttime'])).dt.total_seconds() / 3600

    # Handle NaN values for the last game of each player
    data['time_to_next_game_hours'] = data['time_to_next_game_hours'].fillna(np.inf)

    # Create quick return indicator (within 24 hours)
    data['quick_return'] = (data['time_to_next_game_hours'] < 24).astype(int)

    # Calculate days since first game for each player
    data['first_game_date'] = data.groupby('name')['starttime'].transform('min')
    data['days_since_first_game'] = (pd.to_datetime(data['starttime']) -
                                     pd.to_datetime(data['first_game_date'])).dt.total_seconds() / (24 * 3600)

    # Calculate days active (number of unique days played)
    data['game_date'] = pd.to_datetime(data['starttime']).dt.date
    temp_days = data.groupby('name')['game_date'].nunique().reset_index()
    temp_days.columns = ['name', 'unique_days_active']
    data = pd.merge(data, temp_days, on='name', how='left')

    # Calculate play density (games per day active)
    temp_density = data.groupby('name').size().reset_index()
    temp_density.columns = ['name', 'total_games']
    temp_density = pd.merge(temp_density, temp_days, on='name')
    temp_density['play_density'] = temp_density['total_games'] / temp_density['unique_days_active']
    data = pd.merge(data, temp_density[['name', 'play_density']], on='name', how='left')

    # === PROGRESSION METRICS ===
    # Calculate cumulative max level reached
    data['cumulative_max_level'] = data.groupby('name')['maxlvl'].cummax()

    # Calculate level progression rate
    data['level_progression_rate'] = data['maxlvl'] / data['game_number']

    # Calculate normalized score (points per turn)
    data['points_per_turn'] = data['points'] / data['turns'].replace(0, 1)

    # Calculate relative performance (compared to player's average)
    data['avg_points'] = data.groupby('name')['points'].transform('mean')
    data['relative_performance'] = data['points'] / data['avg_points'].replace(0, 1)

    # Calculate progression velocity (change in max level)
    data['prev_max_level'] = data.groupby('name')['cumulative_max_level'].shift(1).fillna(0)
    data['level_improvement'] = data['cumulative_max_level'] - data['prev_max_level']

    # === BUILD CONSISTENCY FEATURES ===

    # Create combined build column
    data['build'] = data['role'] + '-' + data['race'] + '-' + data['align']

    # Calculate build consistency
    # 1. Previous build
    data['prev_build'] = data.groupby('name')['build'].shift(1)
    data['same_build_as_prev'] = (data['build'] == data['prev_build']).astype(int)

    # 2. Build streak (consecutive same builds)
    data['build_change'] = (data['build'] != data['prev_build']).astype(int)
    data['build_streak_id'] = data.groupby('name')['build_change'].cumsum()

    # Count occurrences of each build_streak_id
    build_streak_counts = data.groupby(['name', 'build_streak_id']).size().reset_index(name='streak_length')
    data = pd.merge(
        data,
        build_streak_counts,
        on=['name', 'build_streak_id'],
        how='left'
    )

    # 3. Calculate top build percentage
    build_counts = data.groupby(['name', 'build']).size().reset_index(name='build_count')
    player_counts = data.groupby('name').size().reset_index(name='player_total')
    build_stats = pd.merge(build_counts, player_counts, on='name')
    build_stats['build_percentage'] = (build_stats['build_count'] / build_stats['player_total']) * 100

    # Get top build for each player
    top_builds = build_stats.loc[build_stats.groupby('name')['build_count'].idxmax()]
    top_builds = top_builds[['name', 'build', 'build_percentage']]
    top_builds.columns = ['name', 'top_build', 'top_build_percentage']

    data = pd.merge(data, top_builds, on='name', how='left')
    data['is_top_build'] = (data['build'] == data['top_build']).astype(int)

    # === DEATH & RECOVERY FEATURES ===

    # Create death type categories
    death_categories = {
        'quit': 'voluntary',
        'escaped': 'success',
        'ascended': 'success',
        'killed by': 'monster',
        'petrified by': 'monster',
        'poisoned by': 'monster',
        'drowned': 'environment',
        'burned by': 'environment',
        'zapped': 'self',
        'fell': 'environment',
        'slipped': 'environment'
    }

    # Categorize deaths
    for category, keywords in death_categories.items():
        if category == 'quit' or category == 'escaped' or category == 'ascended':
            data.loc[data['death'] == category, 'death_category'] = keywords
        else:
            data.loc[data['death'].str.contains(category, case=False, na=False), 'death_category'] = keywords

    # Fill any uncategorized deaths
    data['death_category'] = data['death_category'].fillna('other')

    # Check if player returned after a negative outcome (not quit/success)
    data['negative_outcome'] = (~data['death_category'].isin(['voluntary', 'success'])).astype(int)
    data['prev_negative'] = data.groupby('name')['negative_outcome'].shift(1).fillna(0)
    data['returned_after_negative'] = (data['prev_negative'] == 1).astype(int)

    # === CONDUCT DIFFICULTY FEATURES ===

    # Parse conducts into separate columns
    common_conducts = ['foodless', 'vegan', 'vegetarian', 'atheist', 'weaponless',
                       'pacifist', 'illiterate', 'polypileless', 'polyselfless', 'wishless']

    for conduct in common_conducts:
        data[f'conduct_{conduct}'] = data['conduct'].str.contains(conduct, case=False, na=False).astype(int)

    # Count total conducts per game
    data['total_conducts'] = data[['conduct_' + c for c in common_conducts]].sum(axis=1)

    # Calculate conduct difficulty score (weighted sum)
    conduct_weights = {
        'foodless': 3,
        'pacifist': 3,
        'illiterate': 2.5,
        'weaponless': 2,
        'atheist': 2,
        'wishless': 1.5,
        'polypileless': 1,
        'polyselfless': 1,
        'vegan': 1,
        'vegetarian': 0.5
    }

    data['conduct_difficulty'] = sum(data[f'conduct_{c}'] * w for c, w in conduct_weights.items())

    # Calculate average conduct difficulty for each player
    data['avg_conduct_difficulty'] = data.groupby('name')['conduct_difficulty'].transform('mean')
    data['relative_conduct_difficulty'] = data['conduct_difficulty'] / data['avg_conduct_difficulty'].replace(0, 1)

    # === LGM-SPECIFIC FEATURES ===

    # Create centered and squared terms for growth modeling
    data['game_number_centered'] = data.groupby('name')['game_number'].transform(
        lambda x: x - x.mean()
    )
    data['game_number_squared'] = data['game_number_centered'] ** 2

    # Calculate rolling statistics (e.g., 3-game windows)
    window_size = 3
    data['rolling_max_level'] = data.groupby('name')['maxlvl'].rolling(
        window=window_size, min_periods=1
    ).mean().reset_index(level=0, drop=True)

    data['rolling_points'] = data.groupby('name')['points'].rolling(
        window=window_size, min_periods=1
    ).mean().reset_index(level=0, drop=True)

    # === PERSISTENCE COMPOSITE SCORE ===

    # Create standardized variables for composite score
    persistence_components = [
        'quick_return',
        'level_progression_rate',
        'streak_length',
        'returned_after_negative',
        'relative_conduct_difficulty'
    ]

    # Ensure all components exist
    for component in persistence_components:
        if component not in data.columns:
            print(f"Warning: Component {component} not found in dataframe")
            persistence_components.remove(component)

    # Standardize each component
    for component in persistence_components:
        mean_val = data[component].mean()
        std_val = data[component].std()
        if std_val > 0:
            data[f'{component}_std'] = (data[component] - mean_val) / std_val
        else:
            data[f'{component}_std'] = 0

    # Calculate composite persistence score
    std_components = [f'{component}_std' for component in persistence_components]
    data['persistence_score'] = data[std_components].mean(axis=1)

    # Calculate cumulative persistence score
    data['cumulative_persistence'] = data.groupby('name')['persistence_score'].cumsum()
    data['avg_persistence'] = data.groupby('name')['persistence_score'].transform('mean')

    print(f"Engineered {len(data.columns) - len(df.columns)} new features for persistence analysis")

    return data