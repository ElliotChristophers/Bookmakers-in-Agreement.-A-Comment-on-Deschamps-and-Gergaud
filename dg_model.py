import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import chi2

n_bins = 5

list_betting_firms = ['B365', 'BS', 'BW', 'GB', 'IW', 'LB', 'PS', 'SB', 'SJ', 'SO', 'SY', 'VC', 'WH']

df = pd.read_csv('all_data.csv',index_col=0)
df['n_bookmakers'] = 13-df.isna().sum(axis=1)/3

#comment out this line for Table 1, below
df = df[df['n_bookmakers']==6]

def mad(row):
    valid_values = row.dropna()
    mean = valid_values.mean()
    mad = (valid_values - mean).abs().mean()
    return mad

venues = ['H','D','A']
v_dict = {'H':'Home','D':'Draw','A':'Away'}

bp_test = []
df_bin = pd.DataFrame()

for v in venues:
    df[f'{v}_longest'] = df[[i+v for i in list_betting_firms]].max(axis=1,skipna=True).values
df['H_return_longest'] = np.where(df['FTHG']>df['FTAG'],df['H_longest']-1,-1)
df['D_return_longest'] = np.where(df['FTHG']==df['FTAG'],df['D_longest']-1,-1)
df['A_return_longest'] = np.where(df['FTHG']<df['FTAG'],df['A_longest']-1,-1)

for v in venues:
    df[f'{v}_mad'] = df[[i+v for i in list_betting_firms]].apply(mad, axis=1)
    df[f'{v}_mean'] = df[[i+v for i in list_betting_firms]].mean(axis=1,skipna=True).values

    df_reg = df[[f'{v}_mean']]
    df_reg = sm.add_constant(df_reg)
    df_reg[f'{v}_mean_squared'] = df_reg[f'{v}_mean'] ** 2
    model = sm.OLS(df[f'{v}_mad'], df_reg).fit()

    bp_test.append([v] + list(het_breuschpagan(model.resid, model.model.exog)[:2]))
    
    df[f'{v}_res'] = model.resid
    df[f'{v}_bins'] = pd.qcut(df[f'{v}_res'], q=n_bins, labels=False)
    df_bin[[f'{v}_return_longest',f'{v}_res',f'{v}_longest']] = df[[f'{v}_return_longest',f'{v}_res',f'{v}_longest',f'{v}_bins']].groupby(by=f'{v}_bins').mean().sort_values(by=f'{v}_res',ascending=True).reset_index(drop=True)

df_dg = pd.DataFrame({
    'Disagreement':['Very Low','Low','Average','High','Very High'],
    'Home':[-0.087,-0.078,-0.099,-0.076,-0.031],
    'Draw':[-0.107,-0.021,-0.027,-0.012,-0.036],
    'Away':[-0.143,-0.07,-0.108,-0.086,-0.043]
})



colors = ['0', '0.5', '0.85']
fig = plt.figure()
ax = fig.add_subplot(211)
for idx,v in enumerate([v_dict[v] for v in venues]):
    ax.plot(df_dg['Disagreement'],df_dg[v],color=colors[idx])
ax.set(xlabel=r'Disagreement',ylabel=r'$\bar{r}_{gj}$',title='Deschamps and Gergaud (2007) Results')
ax.tick_params(axis='x', labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax = fig.add_subplot(212)
for idx, v in enumerate(venues):
    ax.plot(df_bin[f'{v}_res'],df_bin[f'{v}_return_longest'],label=v_dict[v],color=colors[idx])
ax.set_title('Replicated Results')
ax.set_xlabel(r'$\bar{\varepsilon}_{gj}$',fontsize=13)
ax.set_ylabel(r'$\bar{r}_{gj}$',fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.legend(loc=(0.025, 0.45))
plt.tight_layout()
plt.subplots_adjust(left=0.175)
plt.show()


fig = plt.figure()
for idx, v in enumerate(venues):
    ax = fig.add_subplot(311+idx)
    ax.scatter(df[f'{v}_mean'],df[f'{v}_res'],color='0',s=2)
    bp_value = f"{bp_test[idx][1]:.2e}".replace("e+0", r"\times 10^{") + "}"
    ax.text(df[f'{v}_mean'].min(),df[f'{v}_res'].max(),fr'Breusch Pagan: $LM= {bp_value}$')
    ax.set_title(v_dict[v])
    ax.set_xlabel(r'$\bar{o}_{ij}$',fontsize=13)
    ax.set_ylabel(r'$\varepsilon_{ij}$',fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

#BP Critical Values
print(chi2.ppf(0.99, 2))
print(chi2.ppf(0.999, 2))

#For Table 1, comment out the n_bookmaker restriction above
print(bp_test)

print(df_bin[['H_longest','D_longest','A_longest']].apply(lambda x: round(x,2)))

print(df_bin[['H_longest','D_longest','A_longest']].std(axis=0).apply(lambda x: round(x,3)))
