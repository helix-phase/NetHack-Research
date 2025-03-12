import pandas as pd
import pickle
import numpy as np


class ExpertExtraction:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

    def filter_expert_players(self):
        # Isolate Ascended Games
        ascended_games = self.df[self.df['death'] == "ascended"]

        # Criterion 1: At least 20 total ascensions
        total_ascensions = ascended_games.groupby('name').size()
        criterion_1 = set(total_ascensions[total_ascensions >= 20].index)

        # Criterion 2: At least 8 consecutive ascensions
        self.df = self.df.sort_values(by=['name', 'realtime'])
        streaks = self.df.groupby('name').apply(self._longest_consecutive_ascensions)
        criterion_2 = set(streaks[streaks >= 8].index)

        # Criterion 3: At least one ascension in the top 2% fastest real-time
        criterion_3 = set(ascended_games[ascended_games['realtime'] <= 18000]['name'].unique())

        # Criterion 4: Ascension under 10,000 turns
        criterion_4 = set(ascended_games[ascended_games['turns'] < 10000]['name'].unique())

        # Criterion 5: Achieved a difficult conduct (pacifist, foodless, illiterate, or 7+ conducts)
        criterion_5 = set(
            ascended_games[ascended_games['conduct'].apply(self._meets_conduct_criteria)]['name'].unique())

        # Combine all criteria according to logical-OR and logical-AND

        # Combine conditions for expert data
        expert_players = criterion_1.union(criterion_2, criterion_3, criterion_4, criterion_5)

        # Strict Expert generation
        strict_experts = set(criterion_1) & set(criterion_2) & (set(criterion_3) | set(criterion_4) | set(criterion_5))

        print(f"Number of expert players: {len(expert_players)}")
        print(f"Number of strict expert players: {len(strict_experts)}")

        return expert_players, strict_experts

    def extract_expert_games(self):
        min_games = 10
        expert_players, strict_experts = self.filter_expert_players()

        expert_df = self.df[self.df['name'].isin(expert_players)]

        strict_expert_df = self.df[self.df['name'].isin(strict_experts)]

        build_consistency = self._calc_build_consistency(strict_expert_df)

        # Get top most consistent players
        top_consistent_players = \
            (build_consistency.query(f'total_games >= {min_games}').sort_values('build_entropy').head(5)['name'])

        # Filter to games from the most consistent players
        consistent_df = strict_expert_df[strict_expert_df['name'].isin(top_consistent_players)]

        # Obtain a list of the gameids for all expert games
        expert_gameids = expert_df["gameid"].unique().tolist()
        consistent_expert_gameids = consistent_df["gameid"].unique().tolist()

        print(f"Unique expert games: {len(expert_gameids)}")
        print(f"Unique strict expert games: {len(consistent_expert_gameids)}")

        return expert_gameids, consistent_expert_gameids

    def save_expert_games(self, output_path="/code/NetHack-Research/data/processed/experts.pkl"):
        expert_games, strict_expert_games = self.extract_expert_games()
        with open(output_path, 'wb') as fp:
            pickle.dump(strict_expert_games, fp)

    def save_processed_data(self, output_path="/code/NetHack-Research/data/processed/new_full_data.csv"):
        # Convert to a set for lookup
        expert_games, strict_games = self.extract_expert_games()
        temp_set = set(expert_games)

        # Exclude expert games
        non_expert_df = self.df[~self.df["gameid"].isin(temp_set)]
        non_expert_df.to_csv(output_path, index=False)

        print(f"Remaining games dataset saved to {output_path}")

    @staticmethod
    def _longest_consecutive_ascensions(player_games):
        count, max_streak = 0, 0
        for _, row in player_games.iterrows():
            if row['death'] == "ascended":
                count += 1
                max_streak = max(max_streak, count)
            else:
                count = 0

        return max_streak

    @staticmethod
    def _meets_conduct_criteria(conduct_str):
        if pd.isna(conduct_str):
            return False
        conducts = conduct_str.split(',')
        difficult_conducts = {"pacifist", "foodless", "illiterate"}

        return any(c in conducts for c in difficult_conducts) or len(conducts) >= 7

    @staticmethod
    def _calc_build_consistency(df):
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

    @staticmethod
    def _calc_build_frequency(df, expert_names):
        expert_games = df[df['name'].isin(expert_names)]
        expert_games['build'] = expert_games['role'] + '-' + expert_games['race'] + '-' + expert_games['align']

        # Count builds
        build_counts = expert_games['build'].value_counts().reset_index()
        build_counts.columns = ['build', 'count']

        # Split the build back into components
        build_counts[['role', 'race', 'align']] = build_counts['build'].str.split('-', expand=True)

        return build_counts, expert_games


if __name__ == "__main__":
    csv_path = "/code/NetHack-Research/data/processed/processed_data.csv"
    extractor = ExpertExtraction(csv_path)

    # Extract experts and save gameids for behavioral cloning experiments
    extractor.save_expert_games("/code/NetHack-Research/data/processed/experts.pkl")

    # Save remaining games for persistence processing
    extractor.save_processed_data("/code/NetHack-Research/data/processed/new_full_data.csv")
