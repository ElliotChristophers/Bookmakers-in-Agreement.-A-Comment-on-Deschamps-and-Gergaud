import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.stats.diagnostic import het_breuschpagan

n_bins = 5

list_betting_firms = ['B365', 'BS', 'BW', 'GB', 'IW', 'LB', 'PS', 'SB', 'SJ', 'SO', 'SY', 'VC', 'WH']

df = pd.read_csv('all_data.csv',index_col=0)
df['n_bookmakers'] = 13-df.isna().sum(axis=1)/3

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
    df[f'{v}_longest_inverse'] = 1/df[f'{v}_longest']
    df[f'{v}_mad'] = (1/df[[i+v for i in list_betting_firms]]).apply(mad, axis=1)
    df[f'{v}_mean_inverse'] = 1/df[[i+v for i in list_betting_firms]].mean(axis=1,skipna=True).values
    df[f'{v}_mean_inverse_squared'] = df[f'{v}_mean_inverse'] ** 2
    df[f'{v}_longest_inverse_squared'] = df[f'{v}_longest_inverse'] ** 2

    scaler = StandardScaler()
    pca = PCA(n_components=4)
    pipeline = make_pipeline(scaler, pca)

    X_pca = pipeline.fit_transform(df[[f'{v}_mean_inverse', f'{v}_longest_inverse',f'{v}_mean_inverse_squared',f'{v}_longest_inverse_squared']])
    X_pca = pd.DataFrame(X_pca,columns=[f'{v}_mean_inverse', f'{v}_longest_inverse',f'{v}_mean_inverse_squared',f'{v}_longest_inverse_squared'])
    X_pca['n_bookmakers'] = df['n_bookmakers']
    X_pca['n_bookmakers_squared'] = df['n_bookmakers']**2
    X_pca = sm.add_constant(X_pca)
    model = sm.OLS(df[f'{v}_mad'], X_pca).fit()

    bp_test.append([v] + list(het_breuschpagan(model.resid, model.model.exog)[:2]))

    df[f'{v}_res'] = model.resid
    df[f'{v}_bins'] = pd.qcut(df[f'{v}_res'], q=n_bins, labels=False)
    df_bin[[f'{v}_return_longest',f'{v}_res',f'{v}_longest']] = df[[f'{v}_return_longest',f'{v}_res',f'{v}_longest',f'{v}_bins']].groupby(by=f'{v}_bins').mean().sort_values(by=f'{v}_res',ascending=True).reset_index(drop=True)

print(df_bin[['H_longest','D_longest','A_longest']].apply(lambda x: round(x,2)))

print(df_bin[['H_longest','D_longest','A_longest']].std(axis=0).apply(lambda x: round(x,3)))
