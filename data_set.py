import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://media.githubusercontent.com/media/Fa11ingDeep/Data-Set/refs/heads/main/Spotify_Youtube.csv"
df_songs=pd.read_csv(url, encoding="UTF-8")
df_songs=df_songs.dropna()
df_songs.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
df_songs.describe()
print(df_songs.head(5))
print(df_songs.columns)
print(df_songs['ID'].nunique())

sns.kdeplot(df_songs['Stream'],fill=True)
plt.title('Densidad de stream', fontsize=14)
plt.xlabel('stream', fontsize=12)
plt.ylabel('densidad', fontsize=12)
plt.show()

sns.kdeplot(df_songs['Views'],fill=True)
plt.title('Densidad de views', fontsize=14)
plt.xlabel('views', fontsize=12)
plt.ylabel('densidad', fontsize=12)
plt.show()