import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

n_bins = 10

list_betting_firms = ['B365', 'BS', 'BW', 'GB', 'IW', 'LB', 'PS', 'SB', 'SJ', 'SO', 'SY', 'VC', 'WH']

df = pd.read_csv('all_data.csv',index_col=0)
df['H_longest'] = df[[i+'H' for i in list_betting_firms]].max(axis=1,skipna=True).values
df['D_longest'] = df[[i+'D' for i in list_betting_firms]].max(axis=1,skipna=True).values
df['A_longest'] = df[[i+'A' for i in list_betting_firms]].max(axis=1,skipna=True).values
df['H_mean'] = df[[i+'H' for i in list_betting_firms]].mean(axis=1,skipna=True).values
df['D_mean'] = df[[i+'D' for i in list_betting_firms]].mean(axis=1,skipna=True).values
df['A_mean'] = df[[i+'A' for i in list_betting_firms]].mean(axis=1,skipna=True).values
df['H_return_longest'] = np.where(df['FTHG']>df['FTAG'],df['H_longest']-1,-1)
df['D_return_longest'] = np.where(df['FTHG']==df['FTAG'],df['D_longest']-1,-1)
df['A_return_longest'] = np.where(df['FTHG']<df['FTAG'],df['A_longest']-1,-1)
df['H_return_mean'] = np.where(df['FTHG']>df['FTAG'],df['H_mean']-1,-1)
df['D_return_mean'] = np.where(df['FTHG']==df['FTAG'],df['D_mean']-1,-1)
df['A_return_mean'] = np.where(df['FTHG']<df['FTAG'],df['A_mean']-1,-1)

df = df[['H_longest','D_longest','A_longest','H_mean','D_mean','A_mean','H_return_longest','D_return_longest','A_return_longest','H_return_mean','D_return_mean','A_return_mean']]

bin_size = len(df) // n_bins

venues = ['H','D','A']
v_dict = {'H':'Home','D':'Draw','A':'Away'}
fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True,sharey=True,figsize=(7.5,5))
for i, category in enumerate(['longest','mean']):
    for j, venue in enumerate(venues):
        dfi = df.sort_values(by=f'{venue}_{category}',ascending=True)
        x = []
        y = []
        for k in range(n_bins):
            start_index = k * bin_size
            end_index = (k + 1) * bin_size if k < n_bins - 1 else len(dfi)
            dfii = dfi.iloc[start_index:end_index]
            x.append(np.mean(dfii[f'{venue}_{category}']))
            y.append(np.mean(dfii[f'{venue}_return_{category}']))

        if i == 1:
            xlabel = r'$\bar{o}_{gj}$'
        else:
            xlabel = ''
            title = str(v_dict[venue])
            axs[i][j].xaxis.set_ticks_position('none') 
        if j == 0:
            ylabel = r'$\bar{r}_{gj}$'
        elif j == 2:
            ylabel = f'{category.capitalize()}'
            axs[i][j].yaxis.set_ticks_position('none') 
        else:
            ylabel = None
            axs[i][j].yaxis.set_ticks_position('none') 
        axs[i][j].plot(np.array(x),np.array(y),color='black',linewidth=0.75)
        axs[i][j].set_xlabel(xlabel,fontsize=13)
        axs[i][j].set_ylabel(ylabel,fontsize=13)
        if i == 0:
            axs[i][j].set_title(title,fontsize=15)
        if j == 2:
            axs[i][j].yaxis.set_label_position("right")
            axs[i][j].set_ylabel(ylabel, rotation=0, fontsize=15,ha='left')
        axs[i][j].spines['top'].set_visible(False)
        axs[i][j].spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

