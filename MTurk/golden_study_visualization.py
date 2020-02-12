import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('golden_results.csv')
df1 = df[['Answer.slider_values']]
df1 = df1['Answer.slider_values'].str.split(',', expand=True)
df1 = df1.drop(labels=65, axis=1).transpose()
people = ['person%i' % x for x in range(df1.shape[1])]
images = ['image%i' % x for x in range(df1.shape[0])]
df1.columns = people
df1 = df1.astype(int)
df_stats = pd.DataFrame()
df_stats['image'] = images
df_stats['mean'] = df1.mean(axis=1).values
df_stats['std'] = df1.std(axis=1).values
df_stats = df_stats.sort_values(by='std', ascending=True)
df_stats['low_var_order'] = range(df1.shape[0])
df_stats

plt.figure(figsize=(20, 10))
plt.scatter(df_stats['low_var_order'], df_stats['mean'])
plt.errorbar(df_stats['low_var_order'], df_stats['mean'], yerr=df_stats['std'], linestyle='none')
