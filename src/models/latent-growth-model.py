import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

csv_file = "/code/NetHack-Research/data/processed/features.csv"
data = pd.read_csv(csv_file)

data['starttime'] = pd.to_datetime(data['starttime'])
# Convert starttime into elapsed days since first game
data['time_since_first_game'] = (data['starttime'] -
                                 data.groupby('name')['starttime'].transform('min')).dt.total_seconds() / (24 * 3600)

# Standardize elapsed time
scaler = StandardScaler()
data['time_since_first_game_scaled'] = scaler.fit_transform(data[['time_since_first_game']])
data['time_since_first_game_squared'] = data['time_since_first_game_scaled'] ** 2

def calculate_build_consistency(df):
    # Create combined build column
    df['build'] = df['role'] + '-' + df['race'] + '-' + df['align']

    build_stats = []
    for name, group in df.groupby('name'):
        # Skip players with too few games
        if len(group) < 5:
            continue

        # Get build counts
        build_counts = group['build'].value_counts()
        top_build = build_counts.index[0]
        top_build_count = build_counts.iloc[0]
        top_build_pct = top_build_count / len(group) * 100

        # Calculate build entropy (lower means more consistent)
        build_probs = build_counts / len(group)
        entropy = -sum(p * np.log2(p) for p in build_probs)

        build_stats.append({
            'name': name,
            'total_games': len(group),
            'unique_builds': len(build_counts),
            'top_build': top_build,
            'top_build_count': top_build_count,
            'top_build_percentage': top_build_pct,
            'build_entropy': entropy
        })

    return pd.DataFrame(build_stats)


# Calculate build consistency
build_consistency = calculate_build_consistency(data)

# Identify most consistent players
min_games = 100
players = 100

# Get top most consistent players
top_consistent_players = \
    (build_consistency.query(f'total_games >= {min_games}').sort_values('build_entropy').head(players)['name'])

# Filter to games from the most consistent players
df = data[data['name'].isin(top_consistent_players)]

lgm_data = df[['name', 'starttime', 'persistence_score', 'cumulative_persistence',
               'time_since_first_game_scaled', 'time_since_first_game_squared']]

# Ensure proper data types
lgm_data = lgm_data.dropna()  # Drop missing values if needed
lgm_data = lgm_data.sort_values(['name', 'starttime'])  # Ensure time order

# Define LGM formula
# Update model formula
lgm_formula = "persistence_score ~ time_since_first_game_scaled + time_since_first_game_squared"

# Fit the model with random intercepts and slopes
lgm_model = smf.mixedlm(lgm_formula, lgm_data, groups=lgm_data["name"], re_formula="1 + time_since_first_game_scaled")

lgm_results = lgm_model.fit(method="lbfgs")

# Print summary of model results
print(lgm_results.summary())

# Extract player-specific parameters
player_params = []
for player, effects in lgm_results.random_effects.items():
    player_params.append({
        'name': player,
        'random_intercept': effects[0],
        'random_slope': effects[1],
        'total_intercept': lgm_results.fe_params[0] + effects[0],
        'total_slope': lgm_results.fe_params[1] + effects[1]
    })

player_params_df = pd.DataFrame(player_params)

# Visualize results
plt.figure(figsize=(12, 8))
plt.scatter(
    player_params_df['total_intercept'],
    player_params_df['total_slope'],
    alpha=0.6
)

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Initial Persistence (Intercept)')
plt.ylabel('Persistence Growth (Slope)')
plt.title(f'Player Persistence Trajectories')
plt.grid(True, alpha=0.3)

# Highlight most persistent players (high intercept, positive slope)
top_persistent = player_params_df[
    (player_params_df['total_intercept'] > player_params_df['total_intercept'].median()) &
    (player_params_df['total_slope'] > 0)
].nlargest(10, 'total_slope')

for _, player in top_persistent.iterrows():
    plt.annotate(
        player['name'],
        (player['total_intercept'], player['total_slope']),
        xytext=(5, 5),
        textcoords='offset points',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

plt.savefig('persistence_lgm_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the top persistent players
print("Top persistent players identified by LGM:")
print(top_persistent[['name', 'total_intercept', 'total_slope']])
top_persistent.to_csv('top_persistent_players.csv', index=False)


# Cluster based on persistence-related metrics
cluster_features = ['avg_persistence', 'cum_progression_velocity', 'smoothed_play_density']
kmeans = KMeans(n_clusters=3, random_state=42)
data['persistence_cluster'] = kmeans.fit_predict(data[cluster_features])

# Run separate LGM per cluster
for cluster in data['persistence_cluster'].unique():
    cluster_data = data[data['persistence_cluster'] == cluster]
    lgm_model = smf.mixedlm(lgm_formula, cluster_data, groups=cluster_data["name"], re_formula="1 + time_since_first_game_scaled")
    lgm_results = lgm_model.fit(method="lbfgs")
    print(f"Cluster {cluster} LGM Summary:")
    print(lgm_results.summary())