import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizations:
    def __init__(self, csv_path):
        # Initialize with Dataframe
        self.full_data = pd.read_csv(csv_path)

    # Generates Average percentage of Levels Reached by NLD-NAO measured against corresponding score.
    def maxlvl_vs_score(self):
        # Load the CSV file a dataframe:
        stats = self.full_data

        # Check required columns for plot:
        columns = ["gameid", "points", "turns", "maxlvl"]

        if not all(col in stats.columns for col in columns):
            raise ValueError(f"CSV file must contain the following columns: {columns}")

        # Calculate Percentage of maxlvl reached by players:
        levels_reached = stats["maxlvl"]
        depths = sorted(set(range(1, max(levels_reached) + 1)))
        percentage = [(levels_reached >= depth).sum() / len(levels_reached) * 100 for depth in depths]

        # percentage = [sum(1 for level in levels_reached if level >= depth) / len(levels_reached) * 100
        #               for depth in depths]

        # Create the plot:
        fig, ax1 = plt.subplots()
        ax1.bar(depths, percentage, label="Percentage of Levels Reached")
        ax1.set_xlabel("Depth of Dungeon")
        ax1.set_ylabel("Percentage of Levels Reached")

        # Add the second plot with seaborn:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Average Score if Died on Level")
        sns.lineplot(
            data=stats,
            x="maxlvl",
            y="points",
            color="red",
            errorbar=('ci', 90),
            ax=ax2,
            label="Average Score"
        )

        # Add title and legend:
        plt.title(f"Average Percentage of Levels Reached by NLD-NAO")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        # Visualize the plot:
        plt.show()

    # Retrieves inter-session times and max levels reached by players and plots
    def maxlvl_vs_intersession_times(self):
        df = self.full_data

        # Ensure required columns are present
        columns = ["gameid", "name", "starttime", "maxlvl"]
        if not all(col in df.columns for col in columns):
            raise ValueError(f"CSV file must contain the following columns: {columns}")

        # Ensure 'starttime' is in datetime format
        df['starttime'] = pd.to_datetime(df['starttime'])

        # Sort by player name and start time
        df = df.sort_values(['name', 'starttime'])

        # Compute inter-session times per player in hours
        df['inter_session_time'] = df.groupby('name')['starttime'].diff().dt.total_seconds() / 3600

        # Remove NaN values
        df = df.dropna(subset=['inter_session_time'])
        # df = df[df['inter_session_time'] > 0]  # Remove 0 or negative inter-session times

        # Plot inter-session times against max level
        plt.figure(figsize=(12, 6))
        plt.scatter(df['inter_session_time'], df['maxlvl'], alpha=0.5, edgecolor='k')
        plt.xlabel('Inter-session Time (Hours)')
        plt.ylabel('Max Level')
        plt.title('Inter-session Times vs. Max Level Reached by Player')
        plt.grid(True)

        # # Logarithmic x-scale and trendline
        # plt.xscale('log')  # Log scale for inter-session time
        # sns.regplot(x='inter_session_time', y='maxlvl',
        #             data=df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, logx=True)

        # Show the plot
        plt.show()

    def maxlvl_vs_insertsomething(self):
        """Plots the distribution of dungeon depths reached."""
        # Visualization code here
        pass

    def maxlvl_vs_insertsomething(self):
        """Plots scores against turns with a regression line."""
        # Visualization code here
        pass


# Test visualizations:
csv_path = "/code/NetHack-Research/data/raw/full_data.csv"
viz = Visualizations(csv_path)

# viz.maxlvl_vs_score()
viz.maxlvl_vs_intersession_times()
