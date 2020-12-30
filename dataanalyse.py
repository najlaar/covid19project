# Display one day data
day = day
month = month
date = pd.to_datetime(f"20200{month}{0 if day<10 else ''}{day}", format='%Y%m%d')

top0 = 50
latest = df1.loc[date].iloc[:,:8]
print(f'\nTop {top0} countries by total cases as of {date.day}/{date.month}/2020')
display(latest[:top0])

top = 20
# Select diplay data
until = date
data = df1[until:]

# Plot data
print(f'\nNew confirmed cases from 1/2/2020 to {date.day}/{date.month}/2020')
countries = list(latest[:top-2].country) + ['Australia', 'Malaysia']
# countries.remove('Cruise Ship')
# fig, axes = plt.subplots(ncols=4, nrows=round(len(countries)/4), figsize=(11,6), constrained_layout=True)
# for i, ax in zip(range(len(countries)), axes.flat):
#     cdata = data[data.country==countries[i]]
#     ax.plot(cdata.new_confirmed)
#     ax.text(until + oneday*2, cdata.loc[until].new_confirmed, format(cdata.loc[until].new_confirmed, ","), color='red')
#     ax2 = ax.twinx()
#     ax2.bar(cdata.index, cdata.confirmed, color='red', alpha=0.05, width=1)
#     ax.set_title(f'{countries[i]} {format(cdata.loc[until].confirmed, ",")}', fontsize=10)
#     ax.axis('off')
#     ax2.axis('off')
# plt.tight_layout()
# plt.show()


mdict = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
cols = ['confirmed','new_confirmed','deaths','new_deaths']
titles = ['Total cumulated Confirmed Cases as of  ','New daily Confirmed Cases on ','Total cumulated Deaths as of  ','New daily Deaths on ']
colors = ['C3','C0','C3','C0']
fig, axes = plt.subplots(figsize=(12,10), nrows=2, ncols=2)
topnum = 20
for i, ax in zip(range(len(cols)), axes.flat):
    data = latest[['country',cols[i]]].reset_index(drop=True).set_index('country').sort_values(by=cols[i])[-topnum:]
    ax.barh(data.index, data[cols[i]], color=colors[i], alpha=0.7)
    ax.set_title(f'Top {topnum} {titles[i]}{date.day} {mdict[date.month]} 2020', fontsize='11')
    for n, v in enumerate(data[cols[i]]):
        ax.text(v, n, format(v, ","), va='center', fontsize=10, color=colors[i])
plt.tight_layout()
plt.savefig('img/total_cases_bar', pad_inches=0)
plt.show()

threshold = 50 # set min threshold

cols = ['country','confirmed','new_confirmed','growth','growth_avg','growth_avg_7']
df2 = df1a[cols].reset_index()
df2 = df2.sort_values('date')

print(f'\nTop 20 countries with the highest day-on-day growth as of {date.day}/{date.month}/2020')
growthdata = df2.loc[(df2.date==date) & (df2.confirmed > 100)].sort_values(by='growth', ascending=False).set_index('country', drop=True)
growthdata[['confirmed','growth','growth_avg']][:20]

countries = list(latest[:17].country) + ['Sweden','Australia', 'Malaysia']

print(f'\nDay-to-day % growth of new confirmed cases for 8 weeks prior to {date.day} {mdict[date.month]} 2020')
fig, axes = plt.subplots(ncols=5, nrows=round(len(countries)/5), figsize=(12,9), constrained_layout=True)
col = ['growth_avg']
for i, ax in zip(range(len(countries)), axes.flat):
    data = df2[df2.country==countries[i]].iloc[-7*8:].reset_index(drop=True)
    ax.plot(data[col].clip(1)[:], alpha=0.7)
    latest_growth = data[col].iloc[-1]
    if latest_growth[0] < 2**(1/28):
        color = 'C2'
    elif latest_growth[0] < 2**(1/21):
        color = 'C7'
    elif latest_growth[0] < 2**(1/14):
        color = 'C1'
    else: color = 'C3'
    double_days = f'doubles in\n{round(math.log(2)/math.log(latest_growth),1)} days' if latest_growth[0] > 2**(1/(7*6)) else ''
    ax.scatter(data.index.max(), latest_growth, marker='o', color=color)
    #ax.hlines(0,data.index.min(),data.index.max(), alpha=0.1)
    ax.text(data.index.max()+7, latest_growth, str(round(latest_growth[0],2)),
            color='white', ha='left', va='center', size=12, bbox=dict(facecolor=color, edgecolor='none'))
    ax.text(data.index.max()+4, latest_growth-0.2, double_days, ha='left', va='center', size=9)
    ax.set_ylim(0.9, 1.8)
    ax.axis('off')
    ax.set_title(countries[i], fontsize=12)
plt.tight_layout()
plt.savefig('img/growth', pad_inches=0)
plt.savefig(f'img_archive/growth_{str(date)[:10]}', pad_inches=0)
plt.show()
