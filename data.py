import pandas as pd
import xlrd
from tqdm import tqdm

list_seasons = ['2001-2002', '2002-2003', '2003-2004', '2004-2005', '2005-2006', '2006-2007', '2007-2008', '2008-2009', '2009-2010', '2010-2011', '2011-2012', '2012-2013', '2013-2014', '2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023','2023-2024']
base_cols = ['Season','Div','Date','HomeTeam','AwayTeam','FTHG','FTAG']
list_betting_firms = ['B365', 'BS', 'BW', 'GB', 'IW', 'LB', 'PS', 'SB', 'SJ', 'SO', 'SY', 'VC', 'WH']
cols = base_cols + [i+j for i in list_betting_firms for j in ['H','D','A']]

dfs = []
for season in tqdm(list_seasons):
    sheets = xlrd.open_workbook(rf'excel_files_football\all-euro-data-{season}.xls').sheets()
    for sheet in sheets:
        sheet = str(sheet)[:-1].split('<')[-1]
        df_season_sheet = pd.read_excel(rf'excel_files_football\all-euro-data-{season}.xls',sheet_name=sheet)
        dfj = pd.DataFrame()
        for c in cols:
            dfj[c] = df_season_sheet[c] if c in df_season_sheet.columns else pd.NA
        dfj['Season'] = season
        dfs.append(dfj)

df = pd.concat(dfs, axis=0, ignore_index=True)
df = df.dropna(subset=base_cols)
df['nan_count'] = df.isna().sum(axis=1)
df = df[(df['nan_count'] % 3 == 0) & (df['nan_count'] != 3 * len(list_betting_firms))]
df = df.drop('nan_count',axis=1)
df[[col for col in df.columns if col not in base_cols]] = df[[col for col in df.columns if col not in base_cols]].replace(0, 1)
df = df.sample(frac=1)
df = df.sort_values(by='Date',ascending=True)
df = df.reset_index(drop=True)

df.to_csv('all_data.csv')
