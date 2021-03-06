import pandas as pd

df = pd.read_csv("data/Salaries.csv")
fan_graph_df = pd.read_csv("data/FanGraphs Leaderboard.csv")
fan_graph_df.drop(columns=["Team", "playerid"], inplace=True)
df = df.join(fan_graph_df.set_index('Name'), on='Name', lsuffix="_salries", rsuffix="_fan_graphs")
df.dropna(inplace=True)
df["BB%"].replace({r'%': ''}, inplace=True, regex=True)
df["K%"].replace({r'%': ''}, inplace=True, regex=True)
df["BB%"] = df["BB%"].astype(float)
df["K%"] = df["K%"].astype(float)
df.to_csv("data/salaries-and-fan-graph-stats.csv",index=False)
