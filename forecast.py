# Use predict_plot function to run model by country:
#
# predict_plot('United Kingdom',                         # compulsory
#              rolling=1,                                # optional: number of period of moving avg if > 1
#              fit_start='20200311', fit_end='20200311', # optional: but essential if converge error
#              days=110,                                 # optional: number of days to plot (default 110)
#              A=.., K=.., C=.., Q=.., B=.., v=..        # optional: manual tweak of Richard's curve parameters
#              showzero=True, showpeak=True)             # optional: show or hide date of plateau or peak
#
# 1. Run the function with predict_plot(country)
#
# 2. If error message "Optimal parameters not found" is generated, this means that
#    data points can't be fit into a curve due to failure to converge after 3000 iterations
#
# 3. In case of such error, change fit_start or/and fit_end (dates) and/or rolling period (days)
#    manually until the error disappears and a resonable result is generated.
#
# 4. If the curve overfits and/or it doesn't follow the expected result, tweak A/K/C/Q/B/v
#    parameters manually. Use the auto-generated parameter values shown underneath the plots
#    as a starting point and refer to be guide below on the effect of changing these parameters.


# country='United Kingdom'
# rolling=3
# days=300
# fit_start=None
# fit_end=None


# showzero=True
# showpeak=True
# showforecast=True
# A=None
# K=None
# C=None
# Q=None
# B=None
# v=None

# Generalised logistic function (Richard's curve)
def richards(x, A, K, C, Q, B, v):
    y = A + (K-A) / ((C + (Q * np.exp(-1 * B * x))) ** (1/v))
    return y

# Fit to Richard's curve model
def predict_plot(country, rolling=1, fit_start=None, fit_end=None, days=130,
                 showzero=True, showpeak=True, showforecast=True,
                 A=None, K=None, C=None, Q=None, B=None, v=None, df2=df2):

    data = df2.loc[df2.country==country].reset_index(drop=True)
    dates = data.date
    latestdate = dates.iloc[-1]

    # # All dates in raw Actuals
    # dates = df2.loc[df2.country==country].reset_index(drop=True).date
    # # All raw Actuals
    # actual0    = df2.loc[df2.country==country, 'confirmed'].reset_index(drop=True)
    # actual0new = df2.loc[df2.country==country, 'new_confirmed'].reset_index(drop=True)

    # Limit Actual data period for model fit
    oneday = pd.Timedelta(1, unit='days')
    fit_start   = pd.to_datetime(fit_start, format='%Y%m%d')
    fit_end     = pd.to_datetime(fit_end, format='%Y%m%d')
    index_start = list(dates[dates==fit_start].index)[0] if fit_start != None else None
    index_end   = list(dates[dates==fit_end+oneday].index)[0] if fit_end != None else None
    actual1 = data.confirmed.iloc[index_start:index_end]

    # Data for model fit input after rolling average
    dropna = range(actual1.index.min(),actual1.index.min()+(rolling-1))
    actual2 = actual1.rolling(window=rolling).mean().drop(dropna) if rolling > 1 else actual1
    xdata = np.array(range(rolling,len(actual2)+rolling))
    ydata = np.array(actual2.values)

    # Auto fit model
    popt, pcov = curve_fit(richards, xdata, ydata, maxfev=3000)
    # Hyperparams of auto fit
    A0, K0, C0, Q0, B0, v0 = popt

    # Overwrite auto fit hyperparams if manual value exists
    A = A0 if A==None else A
    K = K0 if K==None else K
    C = C0 if C==None else C
    Q = Q0 if Q==None else Q
    B = B0 if B==None else B
    v = v0 if v==None else v

    # Calculate prediction over required num of days
    xdata2 = np.array(range(days))
    predcalc = richards(xdata2, A, K, C, Q, B, v)
    pred_index = range(actual2.index.min(), days+actual2.index.min())

    # Forecast dates that overlap with Actuals
    predict_dates1 = list(filter(lambda row: row <= dates.index.max(), pred_index))
    dates_part_1 = []
    for dy in predict_dates1:
        dates_part_1.append(dates.loc[dy]) 
    predict1 = pd.DataFrame(predcalc[:len(dates_part_1)], columns=['predict'], index=pred_index[:len(dates_part_1)]).clip(0)
    data = data.join(predict1)

    # Forecast dates beyond Actuals period
    predict_dates2 = list(filter(lambda row: row > dates.index.max(), pred_index))
    dates_part_2 = []
    for ix in range(len(predict_dates2)):
        dates_part_2.append(dates.max() + oneday * (ix+1))
    predict2 = pd.DataFrame(predcalc[len(dates_part_1):], columns=['predict'], index=pred_index[len(dates_part_1):])
    predict2['date'] = dates_part_2
    predict2['country'] = country

    # Combine both tables
    data = pd.concat([data, predict2])
    # Shift prediction time for rolling avg and remove final nan rows
    data.predict = data.predict.shift(-1*(rolling+1))
    data = data[:-1*(rolling+1)]
    # Calc new cases & growth
    data.predict = data.predict.fillna(-1)
    data.predict = data.predict.astype(int)
    data.predict = data.predict.replace(-1, np.nan)
    data['new_predict'] = data.predict.diff()
    data['new_conf_avg_7'] = round(data.new_confirmed.rolling(window=7).mean())
    data['temp_conf'] = data.apply(lambda row: row.predict if pd.isnull(row.confirmed) else row.confirmed, axis=1)
    data['temp_new'] = data.temp_conf.diff()
    data['new_pred_avg_7'] = round(data.temp_new.rolling(window=7).mean())
    data['p_growth'] = np.nan
    for i in data.index:
        data.iloc[i, data.columns.get_loc('p_growth')] = 0 if data.iloc[i-1].predict == 0 else round(data.iloc[i].predict/data.iloc[i-1].predict,3).clip(1)    
    data['p_gr_avg_7'] = round(data.p_growth.rolling(window=7).mean(),3).clip(1)
    data = data[['date', 'country',
                 'confirmed', 'new_confirmed', 'new_conf_avg_7', 'growth', 'growth_avg_7',
                 'predict', 'new_predict', 'new_pred_avg_7', 'p_growth', 'p_gr_avg_7']]

    # Save df in csv
    data.to_csv(f"data/dfdata/forecast_{country.replace(' ','_')}.csv")
    data.to_csv(f"data/archive/forecast_{country.replace(' ','_')}_{str(latestdate)[:10]}.csv")

    # Find dates for plot labels
    # Latest available date in raw Actual
    latestindex = data[data.confirmed.isnull()].index.min()-1
    latestdate  = data.date.iloc[latestindex]

    # Date of plateau
    oneday = pd.Timedelta(1, unit='days')
    zeroindex = data.iloc[latestindex:][data.iloc[latestindex:].new_predict < 0.5].index.min()
    to_zero_days  = zeroindex - latestindex
    to_zero_weeks = round(to_zero_days/7)
    zerodate = latestdate + oneday*to_zero_days
    if (to_zero_days < 5) | (np.isnan(zeroindex)): showpeak = False

    # Date of peak
    peakindex = data[data.new_predict==data.new_predict.max()].index[0]
    to_peak_days  = peakindex - latestindex
    to_peak_weeks = round(to_peak_days/7)
    peakdate = latestdate + oneday*to_peak_days
    if to_peak_days < 5: showpeak = False

    # Plot prediction
    fig, ax = plt.subplots(figsize=(12,5), ncols=2, nrows=1)
    mdict = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    if showforecast==True:
        ax[0].plot(data.date, data.predict,  ls='dashed', c='C1', label='Forecast')
        ax[1].plot(data.date, data.new_predict, ls='dashed', c='C1', label='Forecast')

    # Plot actual
    value_latest = data.confirmed.iloc[latestindex]
    value_latest_new = data.new_confirmed.iloc[latestindex]
    ax[0].plot(data.date, data.confirmed, c='C0', label='Actual to date')
    ax[0].scatter(latestdate, int(value_latest), c='C0')
    ax[0].text(latestdate+oneday*2, value_latest, f'{latestdate.day} {mdict[latestdate.month]}\n{format(int(value_latest), ",")} cases', va='top')
    ax[1].plot(data.date, data.new_confirmed, c='C0', label='Actual daily')
    ax[1].scatter(latestdate, int(value_latest_new), color='C0')
    ax[1].text(latestdate+oneday*2, value_latest_new, f'{latestdate.day} {mdict[latestdate.month]}\n{format(int(value_latest_new), ",")} cases', va='top')

    # Plot prediction end points
    if showforecast==True:
        if (showzero==True) & (pd.isna(to_zero_days)==False) :
            ax[0].scatter(zerodate, data.predict.max(), c='C1')      
            predictmax = int(math.ceil(data.predict.max()/100.0))*100 if data.predict.max() > value_latest else int(math.ceil(value_latest/100.0))*100
            zerolabel = f"{int(round(to_zero_weeks))} week{'s' if to_zero_weeks>1 else ''}"
            ax[0].text(zerodate+oneday*2, predictmax, f"{zerodate.day} {mdict[zerodate.month]}\n{format(predictmax, ',')} cases\nin {zerolabel}", va='top')
        if (showpeak==True) & (pd.isna(to_peak_days)==False):
            ax[1].scatter(peakdate, data.new_predict.max(), c='C1')
            peaklabel = f"{int(round(to_peak_weeks))} week{'s' if to_peak_weeks>1 else ''}"
            ax[1].text(peakdate+oneday*2, data.new_predict.max(), f"{peakdate.day} {mdict[peakdate.month]}\nin {peaklabel}", va='top')

    if showforecast==False:
        ax[0].text(data.date.iloc[0] + oneday*days/2, (data.confirmed.min()+data.confirmed.max())/2, 'Latest forecast\nto be reviewed', bbox=dict(facecolor='none', edgecolor='C1', linewidth=2, linestyle='dashed'), ha='center')
        ax[1].text(data.date.iloc[0] + oneday*days/2, (data.new_confirmed.min()+data.new_confirmed.max())/2, 'Latest forecast\nto be reviewed', bbox=dict(facecolor='none', edgecolor='C1', linewidth=2, linestyle='dashed'), ha='center')

    # Plot titles and axis
    ax[0].set_title(f'{country.upper()}\nTotal cumulated cases')
    ax[0].set_ylabel('Num of cumulative cases')    
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax[0].legend(loc='lower right' if showforecast==True else 'center right')

    ax[1].set_title(f'{country.upper()}\nNew daily cases')
    ax[1].set_ylabel('Num of new daily cases')
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax[1].legend(loc='lower right' if showforecast==True else 'center right')

    if showforecast==False:
        ax[0].set_xlim(dates.values[0], dates.values[0] + oneday * days)
        ax[1].set_xlim(dates.values[0], dates.values[0] + oneday * days)

    fig.autofmt_xdate(rotation=90)
    plt.tight_layout()
    plt.savefig(f"img/forecast_{country.replace(' ','_')}_latest", pad_inches=0)
    #plt.savefig(f"img_archive/forecast_{country.replace(' ','_')}_{str(latestdate)[:10]}", pad_inches=0)    
    plt.show()

    print(country)
    print(f'Auto fit  : A={int(A0)}, K={int(K0)}, C={round(C0,4)}, Q={round(Q0,4)}, B={round(B0,4)}, v={round(v0,4)}')
    if (A!=A0) | (K!=K0) | (C!=C0) | (Q!=Q0) | (B!=B0) | (v!=v0):
        print(f'Manual fit: A={int(A)}, K={int(K)}, C={round(C,4)}, Q={round(Q,4)}, B={round(B,4)}, v={round(v,4)}')
        
     
     
    def plot_forecasts(country, fdates, days=200):
    fig, ax = plt.subplots(figsize=(12,5), nrows=1, ncols=2)
    titles = ['Total cumulated cases', 'New daily cases']
    ylabels = ['Num of cumulated cases', 'Num of new daily cases']
    for n in [0,1]:

        # Plot actual
        col = 'confirmed' if n==0 else 'new_confirmed'
        data0 = pd.read_csv(f"data/dfdata/forecast_{country.replace(' ','_')}.csv", index_col=1).iloc[:,1:]
        data0.index = pd.to_datetime(data0.index)
        actualmax = data0[data0.confirmed>0].index.max()
        ax[n].plot(data0[col], lw=2, label=f'Actual as of {str(actualmax)[8:10]} {mdict[int(str(actualmax)[5:7])]}')
        ax[n].scatter(actualmax, data0[col].loc[actualmax])
        xmin = data0.index.min()
        xmax = data0.index.min()+oneday*days
        shift = (xmax-xmin)*0.02
        ax[n].text(actualmax+shift, data0[col].loc[actualmax], format(int(data0[col].loc[actualmax]),","), va='center')

        # Plot forecast
        col = 'predict' if n==0 else 'new_predict'
        for i, fdate in enumerate(fdates):
            color = f'C{i}'
            txtcolor = 'black' if i==0 else color
            alpha = 1 if i==0 else 0.5
            data1 = pd.read_csv(f"data/archive/forecast_{country.replace(' ','_')}_{fdate}.csv", index_col=1).iloc[:,1:]
            data1.index = pd.to_datetime(data1.index)
            label = f"Forecast as of {str(fdate)[-2:]} {mdict[int(str(fdate)[5:7])]}"
            ax[n].plot(data1[col], ls='dashed', alpha=alpha, label=label, color=color)
            if n==0:
                ax[n].text(xmax-shift, data1[col].iloc[-1], format(int(round(data1[col].iloc[-1],-2)), ","), va='center', ha='right', color=txtcolor)

        ax[n].set_title(f'{country.upper()}\n{titles[n]}')
        ax[n].set_xlim(data0.index.min(),xmax)    
        ax[n].set_ylabel(ylabels[n])
        ax[n].legend()
        ax[n].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.autofmt_xdate(rotation=90)
    plt.tight_layout()
    plt.savefig(f"img/forecast_{country.replace(' ','_')}", pad_inches=0) 
    plt.show()
    
    
predict_plot('China', days=190, showzero=False)
    
predict_plot('Malaysia', days=150, rolling=3, fit_start='20200313', fit_end='20200525', A=7820)

plot_forecasts('Malaysia', ['2020-05-27','2020-05-19','2020-05-08','2020-05-02','2020-04-03'], days=180)
