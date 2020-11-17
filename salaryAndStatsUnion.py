import pandas as pd
df = pd.read_csv("data/baseballreference-stats.csv")
df.drop(columns=['Rk'],inplace=True)
fan_graph_df = pd.read_csv("data/FanGraphs Leaderboard.csv")
df = df.join(fan_graph_df.set_index('Name'), on='Name',lsuffix="br",rsuffix="salaries")
# df.dropna(inplace=True)
df.to_csv("data/salaries-and-fan-graph-stats.csv")

