# Latest available Actuals for Forecasting
day = 9
month = 6

# Import libraries
import pandas as pd
pd.options.display.max_rows = 200
import numpy as np
import math
import json
import requests
import os
import datetime as dt
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.dates as mdates
from matplotlib import cm, gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.optimize import curve_fit
import warnings
import time
from IPython.display import Image
warnings.filterwarnings("ignore")
oneday = pd.Timedelta(1, unit='days')
date  = pd.to_datetime(f"20200{month}{0 if day<10 else ''}{day}", format='%Y%m%d')
day_pub, month_pub, date_pub = day, month, date


allcols = ['date','country', 'state', 'confirmed_raw', 'confirmed', 'deaths', 'recovered', 'active', 'new_confirmed', 'new_deaths', 'new_recovered']

# Multiple downloads from John Hopkins Hospital database
start  = pd.to_datetime(f'20200603', format='%Y%m%d')
end    = pd.to_datetime(f'20200609', format='%Y%m%d')
for d in range((end - start).days+1):
    date = start + oneday * d
    day0, month0, year0 = str(date)[8:10], str(date)[5:7], str(date)[:4]
    url = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{month0}-{day0}-{year0}.csv'
    zero = 0 if day < 10 else ''
    df = pd.read_csv(url)
    df.to_csv(f'data/csv2/{month0}-{day0}-{year0}.csv')
    
   
# Download Oxford global govt response tracker (new format)
# https://www.bsg.ox.ac.uk/research/research-projects/oxford-covid-19-government-response-tracker
oxford = pd.read_csv('https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv')
oxford.to_csv('data/csv2/govt_tracker_raw_v2.csv')

# Clean up Oxford table (new format)
oxford = pd.read_csv('data/csv2/govt_tracker_raw_v2.csv')
oxcols = {'CountryName':'country', 'Date':'date', 'C1_School closing':'school',
          'C2_Workplace closing': 'workplace', 'C3_Cancel public events':'events',
          'C5_Close public transport':'transport', 'H1_Public information campaigns':'campaigns',
          'C7_Restrictions on internal movement':'movement', 'C8_International travel controls':'immigration',
          'C4_Restrictions on gatherings':'gatherings', 'C6_Stay at home requirements':'at_home',
          'H2_Testing policy':'testing', 'H3_Contact tracing':'tracing',
          'E1_Income support':'income', 'E2_Debt/contract relief':'relief', 'E3_Fiscal measures':'fiscal',
          'E4_International support':'intl_aid', 'H4_Emergency investment in healthcare':'healthcare',
          'H5_Investment in vaccines':'vaccines', 'StringencyIndex':'stringency_raw','StringencyIndexForDisplay':'stringency'}

oxford = oxford.rename(columns=oxcols)
oxford = oxford[oxcols.values()]
skipcols = ['country','fiscal','healthcare','vaccines','intl_aid','income','relief']
cols = [col for col in list(oxford.columns) if col not in skipcols]
oxford.at_home = oxford.apply(lambda row: row.movement if pd.isna(row.at_home) else row.at_home, axis=1)
oxford.at_home = oxford.apply(lambda row: row.movement if ((row.at_home==0)&(row.country=='Malaysia')) else row.at_home, axis=1)

for col in skipcols: oxford[col] = oxford[col].replace(0, np.nan)
oxford.date = pd.to_datetime(oxford.date, format='%Y%m%d')
oxford = oxford.set_index('date',drop=True)

datelist = []
date0 = oxford.index.min()
while date0 <= date_pub: # choose to publish either based on latest forecast or latest actual
    datelist.append(date0)
    date0 = date0 + oneday
countries = oxford.country.unique()

oxford_raw = oxford.copy()
oxford = pd.DataFrame()
for country in countries:
    temp = pd.DataFrame(index=datelist)
    temp = temp.join(oxford_raw[oxford_raw.country==country])
    temp.country = country
    oxford = pd.concat([oxford, temp])
    
oxford.stringency = oxford.groupby('country').stringency.ffill()
oxford.to_csv('data/csv2/govt_tracker_v2.csv')


 Process files with formatting prior to 23/03/2020
path = 'data/csv/'
fileList = os.listdir(path)
df = pd.DataFrame()
for file in fileList:
    if file.endswith('.csv'):
        df0 = pd.read_csv(f'{path}{file}')
        df0.columns = map(str.lower, df0.columns)
        df0 = df0.rename(columns={'country/region':'country'})
        df0 = df0.rename(columns={'province/state':'state'})
        df0 = df0.rename(columns={'last update':'updated'})
        df0 = df0[['updated','country','state','confirmed','deaths','recovered']]
        df0.updated = pd.to_datetime(df0.updated)
        df0.state = df0.state.fillna('')
        df0['date'] = pd.to_datetime(f'{file[6:10]}{file[0:2]}{file[3:5]}', format='%Y%m%d')
        df0['fips'] = 0
        for col in ['confirmed','deaths','recovered']:
            df0[col] = df0[col].fillna(0)
            df0[col] = df0[col].astype(int)
        df = pd.concat([df, df0])
    df01 = df.copy()
        
# Process files with formatting from 23/03/2020
path = 'data/csv2/'
fileList = os.listdir(path)
df = pd.DataFrame()
for file in fileList:
    if file.endswith('2020.csv'):
        df0 = pd.read_csv(f'{path}{file}')
        df0.columns = map(str.lower, df0.columns)
        df0 = df0.rename(columns={'country_region':'country'})
        df0 = df0.rename(columns={'province_state':'state'})
        df0 = df0.rename(columns={'last_update':'updated'})
        df0 = df0[['updated','country','state','confirmed','deaths','recovered','active','fips','admin2']]
        df0.updated = pd.to_datetime(df0.updated)
        df0.state = df0.state.fillna('')
        df0['date'] = pd.to_datetime(f'{file[6:10]}{file[0:2]}{file[3:5]}', format='%Y%m%d')
        for col in ['confirmed','deaths','recovered', 'active','fips']:
            df0[col] = df0[col].fillna(0)
            df0[col] = df0[col].astype(int)
        df = pd.concat([df, df0])
    df02 = df.copy()
    
# Append data from Hubei prior to 22/01/2020
raw = pd.read_csv('data/csv/older/pre_01-22-2020.csv')
prior = pd.DataFrame(columns=df0.columns)
for col in ['confirmed', 'country', 'deaths', 'recovered', 'state', 'updated']:
    prior[col] = raw[col]
for col in ['confirmed', 'deaths', 'recovered']:
    prior[col] = prior[col].astype(int)
prior.state = 'Hubei'
prior.country = 'Mainland China'
prior.fips = 0
prior.updated = pd.to_datetime(prior.updated, format='%d/%m/%Y')
prior.date = prior.updated

df = pd.concat([df02, df01, prior]).reset_index()
df.active = df.active.fillna(0)
df0 = df.copy()
df0.to_csv('data/dfdata/df0.csv') # raw before manual adjustments

# Overwrite raw data with manual fix e.g. when cumulative data had to be restated to be lower
manual_fix = pd.read_csv('data/dfdata/manual_fix.csv')
manual_fix.date = pd.to_datetime(manual_fix.date, format='%Y%m%d')
for n in range(len(manual_fix)):
    row = manual_fix.iloc[n]
    if pd.isna(row.state): find_state = ''
    else: find_state = row.state    
    ix = df0[(df0.country==row.country) & (df0.date==row.date) & (df0.state==find_state)].index[0]
    c = df0.columns.get_loc('confirmed')
    d = df0.columns.get_loc('deaths')
    r = df0.columns.get_loc('recovered')
    if(pd.notnull(row.confirmed)): df0.iloc[ix,c] = row.confirmed
    if(pd.notnull(row.deaths))   : df0.iloc[ix,d] = row.deaths
    if(pd.notnull(row.recovered)): df0.iloc[ix,r] = row.recovered
    #print(ix,row.country,row.date,find_state,row.deaths,df0.iloc[ix,d])
for col in ['confirmed','deaths','recovered']:
    df0[col] = df0[col].astype('int')
    
 
time_start = time.time()
print(f'Start: {(dt.datetime.utcfromtimestamp(time_start).strftime("%H:%M"))}')

# Geo clean-up mapping dictionary
geo = json.load(open('data/geo.json'))
df.country = df.country.apply(lambda row: geo['countries'][row] if row in list(geo['countries'].keys()) else row)
df['temp'] = ''
for i in range(len(df)):
    state = df.iloc[i].state
    country = df.iloc[i].country
    
    if state=='Mexico':
        df.iloc[i, df.columns.get_loc('state')] = 'Mexico (state)'
    elif state=='Amazonas':
        df.iloc[i, df.columns.get_loc('state')] = f'Amazonas ({country})'  
    elif (state == 'None') | (state == 'UK'):
        df.iloc[i, df.columns.get_loc('state')] = country
    elif state[:8]=='Falkland':
        df.iloc[i, df.columns.get_loc('state')] = 'Falkland Islands'
    elif state.find('Princess') > 0:
        if country != 'Cruise Ship':
            df.iloc[i, df.columns.get_loc('state')] = f'Cruise {country}'
        else:
            df.iloc[i, df.columns.get_loc('state')] = 'Cruise Ship'
    elif state == 'Unknown':    
         df.iloc[i, df.columns.get_loc('state')] = f'Unknown {country}'
    elif state == 'Recount':    
         df.iloc[i, df.columns.get_loc('state')] = f'Recount {country}'
    elif country == 'United States':
        if state == 'Washington, D.C.':
            s = 'District of Columbia'
        elif state == 'US':
            s = 'Unassigned'
        elif state == 'Chicago':
            s = 'Illinois'
        elif state.find('Virgin Islands')!=-1:
            s = 'US Virgin Islands'
        elif state.find(', ') > 0:
            s0 = state[state.find(', ')+2:state.find(', ')+4]
            s = geo['us_states'][s0]
        elif state in list(geo['us_states'].keys()):
            s = geo['us_states'][state]
        else:
            s = state
        df.iloc[i, df.columns.get_loc('temp')] = s
            
    elif country == 'Canada':
        if state.find('Princess') > 0:
            s = 'Cruise Ship'               
        elif state.find(', ') > 0:
            ca = {'Edmonton, Alberta':'Alberta', 'Calgary, Alberta':'Alberta',
                  'London, ON':'Ontario', 'Toronto, ON':'Ontario', ' Montreal, QC':'Quebec'}
            s = ca[state]
        else:
            s = state
        df.iloc[i, df.columns.get_loc('temp')] = s
        
time_states1 = time.time()
print(f'Time to process states 1 fix: {int((time_states1 - time_start)/60)} mins')
        
df['state'] = df.apply(lambda row: row['temp'] if row['temp']!='' else (row['state'] if row['state']!='' else row['country']), axis=1)
territories = ['Hong Kong', 'Macau',
               'Aruba', 'Curacao', 'Sint Maarten',
               'French Polynesia', 'New Caledonia', 'Saint Barthelemy', 'Saint Pierre and Miquelon', 'St Martin',
               'Northern Mariana Islands', 'Guam', 'Puerto Rico', 'American Samoa', 'US Virgin Islands']
df.country = df.apply(lambda row: row.state if (row.state in territories) else row.country, axis=1)
df.country = df.apply(lambda row: row.state if (row.country in ['United Kingdom', 'Denmark']) else row.country, axis=1)
df.drop('temp', axis=1, inplace=True)

time_states2 = time.time()
print(f'Time to process states 2 fix: {int((time_states2 - time_states1)/60)} mins')

df = pd.pivot_table(df, values=['confirmed','deaths','recovered','active'], index=['date','country','state'], columns=None, aggfunc='sum')
df = df.rename(columns={'confirmed':'confirmed_raw'})
df = df.reset_index()
df.to_csv('data/dfdata/df.csv')

dftemp = pd.DataFrame()
for state in df.state.unique():
    dfstate = df[df.state==state].copy()
    dfstate = dfstate.sort_index(ascending=False)
    dfstate['confirmed'] = dfstate.confirmed_raw
    # Replace raw with average between 2 dates due to incomplete data due to cut-off reporting time
    #e.g. if new cases suddenly drop by 90% in one day, most likely it's due to incomplete data
    for i in range(1,len(dfstate.index)-1):
        ratio = 1/2 # smoothing average ratio
        if abs((dfstate.iloc[i+1].confirmed_raw - dfstate.iloc[i].confirmed_raw) / (dfstate.iloc[i].confirmed_raw - dfstate.iloc[i-1].confirmed_raw)) < 0.1:
            dfstate.iloc[i, dfstate.columns.get_loc('confirmed')] = int(dfstate.iloc[i+1].confirmed_raw * ratio
                                                                        + dfstate.iloc[i-1].confirmed_raw * (1-ratio))
    for col in ['confirmed','deaths','recovered']:
        colname = f'new_{col}'
        dfstate[colname] = dfstate[col].diff(-1)
        dfstate[colname] = dfstate[colname].fillna(dfstate[col])
        dfstate[colname] = dfstate[colname].astype('int')
    dftemp =  pd.concat([dftemp, dfstate])
df = dftemp.sort_values(['date','new_confirmed'], ascending=False)

newcols = ['date','country', 'state', 'confirmed_raw', 'confirmed', 'deaths', 'recovered', 'active', 'new_confirmed', 'new_deaths', 'new_recovered']
df = df[allcols]
df = df.reset_index(drop=True)
df.to_csv('data/dfdata/df.csv')

time_end = time.time()
print(f'Time to process smoothing   : {int((time_end - time_states2)/60)} mins')
print(f'Total time                  : {int((time_end - time_start)/60)} mins')


# Read from cleansed data
df = pd.read_csv('data/dfdata/df.csv')
df = df.drop(df.columns[0], axis=1)
df.date= pd.to_datetime(df.date)
