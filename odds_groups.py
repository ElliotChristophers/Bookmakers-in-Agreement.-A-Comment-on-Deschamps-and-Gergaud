import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, ttest_1samp
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

n_bins = 5
n_games = 100

list_betting_firms = ['B365', 'BS', 'BW', 'GB', 'IW', 'LB', 'PS', 'SB', 'SJ', 'SO', 'SY', 'VC', 'WH']

df = pd.read_csv('all_data.csv',index_col=0)

def mad(row):
    valid_values = row.dropna()
    mean = valid_values.mean()
    mad = (valid_values - mean).abs().mean()
    return mad

def process_group(group):
    group['bin'] = pd.qcut(group[f'{v}_mad'], n_bins, labels=False)
    bin_means = group.groupby('bin').mean()
    return linregress(bin_means[f'{v}_mad'], bin_means[f'{v}_return_longest'])[0]

venues = ['H','D','A']
v_dict = {'H':'Home','D':'Draw','A':'Away'}

max_counts = []
list_lists_slopes = []
table = []

for v in venues:
    df[f'{v}_longest'] = df[[i+v for i in list_betting_firms]].max(axis=1,skipna=True).values
df['H_return_longest'] = np.where(df['FTHG']>df['FTAG'],df['H_longest']-1,-1)
df['D_return_longest'] = np.where(df['FTHG']==df['FTAG'],df['D_longest']-1,-1)
df['A_return_longest'] = np.where(df['FTHG']<df['FTAG'],df['A_longest']-1,-1)
for v in venues:
    df[f'{v}_mad'] = df[[i+v for i in list_betting_firms]].apply(mad, axis=1)
    group = df[[f'{v}_longest',f'{v}_return_longest',f'{v}_mad']].groupby(by=f'{v}_longest').filter(lambda x: len(x) >= n_games).groupby(by=f'{v}_longest')
    max_c = group.count().sort_values(by=f'{v}_return_longest').iloc[-1]
    max_counts.append([max_c.name,max_c[f'{v}_mad']])
    result = group.apply(process_group).reset_index(drop=True).values
    table.append([round(i,3) for i in ttest_1samp(np.array(result),0,alternative='two-sided')] + [round(np.mean(result),3),round(np.std(result),3),len(result)])
    list_lists_slopes.append(result)

print(max_counts)

df_print = pd.DataFrame(table,columns=[r'$t$',r'$Pr(|T|\geq|t| \, \mid H_0)$',r'$\overline{\beta_j}',r'$s_{\beta_j}$',r'$n$'],index=['Home','Draw','Away'])
print(df_print.transpose().to_latex())

binsi = [15,15,15]
rangei = [(-9,9),(-9,9),(-10*9/8,10*9/8)]
xticksi = [[-8,-4,0,4,8],[-8,-4,0,4,8],[-10,-5,0,5,10]]
venues_full = ['Home', 'Draw', 'Away']
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=False,sharey=False)
for i, l in enumerate(list_lists_slopes):
    ax = [ax1,ax2,ax3][i]
    tt, tp = ttest_1samp(np.array(l),0,alternative='two-sided')
    ax.hist(np.array(l),bins=binsi[i],range=rangei[i],color='0.8', edgecolor='black',density=True)
    ax.set(ylabel='Density',title=f'{venues_full[i]}')
    ax.set_xlabel(xlabel=r'$\beta_{jk}$',fontsize=13)
    low, high = ax.set_xlim()
    bound = max(abs(low), abs(high))
    ax.set_xlim(-bound, bound)
    low, high = ax.set_ylim()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(xticksi[i])
plt.tight_layout()
plt.show()
