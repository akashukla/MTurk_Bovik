import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('golden_results.csv')
df1 = df[['Answer.slider_values']]
df1 = df1['Answer.slider_values'].str.split(',', expand=True)
df1 = df1.drop(labels=65, axis=1)

people = ['person%i' % x for x in range(df1.shape[0])]
images = ['image%i' % x for x in range(df1.shape[1])]
df1.columns = images
df1.index = people
df1 = df1.astype(int)
df2 = df1.transpose()

df_stats = pd.DataFrame()
df_stats['image'] = images
df_stats['mean'] = df1.mean().values
df_stats['std'] = df1.std().values
df_stats = df_stats.sort_values(by='std', ascending=True)
df_stats['low_var_order'] = range(df1.shape[1])
df_stats

plt.figure(figsize=(20, 10))
ax = plt.gca()
plt.scatter(df_stats['low_var_order'], df_stats['mean'])
plt.errorbar(df_stats['low_var_order'], df_stats['mean'], yerr=(df_stats['std']*1), linestyle='none', color='green')
plt.ylim([0,50])
ax.axvline(x=30.5, ymin=0, ymax=50, color='red', linestyle='--')
