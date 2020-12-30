# Read from cleansed data
df = pd.read_csv('data/dfdata/df.csv')
df = df.drop(df.columns[0], axis=1)
df.date= pd.to_datetime(df.date)

# Function to reshape df into country-level or state-level etc
def subset_geo(source, indices):
    data = pd.pivot_table(source, values=source.columns[3:], index=indices, columns=None, aggfunc='sum')
    data = data.reset_index()
    data = data.sort_values('date')
    data = data[indices + allcols[-7:]]
    return data
    
 def calc_growth(data, geo='country', rolling_window=3):
    dftemp = pd.DataFrame()
    for g in data[geo].unique():
        dfc = data[data[geo]==g].copy().reset_index(drop=True)
        dfc['new_growth'] = 0
        dfc['growth'] = 0
        for i in range(1,len(dfc.index)):
            for newcol in ['growth', 'new_growth']:
                oldcol = 'confirmed' if newcol == 'growth' else 'new_confirmed'
                dfc.iloc[i, dfc.columns.get_loc(newcol)] = 0 if dfc.iloc[i-1][oldcol] == 0 else round(dfc.iloc[i][oldcol]/dfc.iloc[i-1][oldcol],3)
        dftemp =  pd.concat([dftemp, dfc])
    data = dftemp.copy()
    data['growth_avg'] = round(data.growth.rolling(window=rolling_window).mean(),3).clip(1)
    data['growth_avg_7'] = round(data.growth.rolling(window=7).mean(),3).clip(1)
    data = data.sort_values(['date', 'confirmed'], ascending=False)
    data = data.set_index('date')
    return data
    
    # Country-level data (df1)
df1 = subset_geo(source=df, indices=['date','country'])
df1 = calc_growth(df1, geo='country', rolling_window=3)
df1 = pd.pivot_table(df, values=df.columns[3:], index=['date','country'], columns=None, aggfunc='sum')
df1 = df1.reset_index()

df1 = pd.read_csv('data/dfdata/df1.csv', index_col=0)
df1.index = pd.to_datetime(df1.index)

# Combine df1 with Oxford table (df1a)
oxford = pd.read_csv('data/csv2/govt_tracker_v2.csv', index_col=0)
oxford.index = pd.to_datetime(oxford.index)

df1a = pd.merge(df1.reset_index(), oxford.reset_index(),  how='left', left_on=['date','country'], right_on = ['index','country'])
df1a = df1a.set_index('date')
df1a.to_csv('data/dfdata/df1a.csv'

ccodes = pd.read_csv('data/dfdata/countrycode.csv', encoding='latin-1', index_col=0)
pop = pd.read_csv('data/dfdata/population.csv', index_col=0)

# Testing database
testraw = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv')
testraw.to_csv('data/csv2/testing_data_raw.csv')

testraw = pd.read_csv('data/csv2/testing_data_raw.csv', index_col=0)
testcols = {'Entity':'country', 'Date':'date', 'Source URL':'source', 'Notes': 'notes', 'Cumulative total':'tested'}
testraw = testraw.rename(columns=testcols).iloc[:,:7]
testraw['type'] = testraw.country.apply(lambda row: row[row.find(' -')+3:].replace(' ','_'))
testraw.country = testraw.country.apply(lambda row: row[:row.find(' -')])
testraw.date = pd.to_datetime(testraw.date)

datelist = []
date0 = pd.to_datetime('20200118', format='%Y%m%d')
while date0 <= df1.index.max():
    datelist.append(date0)
    date0 = date0 + oneday
countries = testraw.country.unique()
# If there are more than 1 type of test, pick the highest one (aggfunc='max')
testraw1 = pd.pivot_table(testraw, values='tested', index=['date','country'], columns=None, aggfunc='max')
testraw1 = testraw1.reset_index()

testnum = pd.DataFrame()
for country in countries:
    temp = pd.DataFrame(index=datelist)
    temp = temp.join(testraw1[testraw1.country==country].set_index('date'))
    temp.country = country
    temp['tested_raw'] = temp.tested
    
    # Extrapolate testing to date as a smooth fill between 2 available dates of data
    temp['tested_raw_ffill'] = temp.tested.fillna(method='ffill')
    temp.tested = temp.tested.interpolate()
    
    # but don't fill beyond latest available data
    lastdata = temp[temp.tested_raw>0].index[-1]
    temp.tested = temp.apply(lambda row: row.tested if row.name <= lastdata else np.nan, axis=1)
    
    # Daily tests
    temp['tested_daily'] = temp.tested.diff().replace(0, np.nan)
    temp['tested_daily_ffill'] = temp.tested_daily.fillna(method='ffill')
    
    for i in range(1, len(temp)):
        before = temp.iloc[i-1, temp.columns.get_loc('tested')]
        latest = temp.iloc[i,   temp.columns.get_loc('tested')]
        temp.iloc[i, temp.columns.get_loc('tested')] = before if latest<before else latest
    
    latest_data_date = temp[temp.tested_raw>0].index[-1]
    testnum = pd.concat([testnum, temp])
    
tests = testnum.join(pop, on='country')
tests['testper1Mdaily'] = tests.apply(lambda row: np.nan if pd.isna(row.tested_daily) else round(row.tested_daily/row['pop']*1000000,1), axis=1)
tests['testper1Mdaily_ff'] = tests.testper1Mdaily.fillna(method='ffill')
tests.to_csv('data/csv2/testing_data.csv')
