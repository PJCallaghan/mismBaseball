import pandas as pd
df = pd.read_csv("data/baseballreference-stats.csv")
df.drop(columns=['Rk'],inplace=True)
salary_df = pd.read_csv("data/Salaries.csv")
df = df.join(salary_df.set_index('Name'), on='Name',lsuffix="br",rsuffix="salaries")
df.dropna(inplace=True)
df.to_csv("data/salaries-and-stats.csv")
