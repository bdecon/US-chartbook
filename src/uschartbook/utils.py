import numpy as np
import pandas as pd

qtrs = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}


def bea_api_nipa(table_list, bea_key):
    ''' Return tables in table list for years in range'''
    import requests
    from datetime import datetime

    years = ','.join(map(str, range(1989, 2020)))

    api_results = []

    for table in table_list:
        url = f'https://www.bea.gov/api/data/?&UserID={bea_key}'\
              f'&method=GetData&datasetname=NIPA&TableName={table}'\
              f'&Frequency=Q&Year={years}&ResultFormat=json'

        r = requests.get(url)

        name, date = (r.json()['BEAAPI']['Results']['Notes'][0]['NoteText']
                       .split(' - LastRevised: '))

        date = datetime.strptime(date, '%B %d, %Y').strftime('%Y-%m-%d')

        api_results.append((table, name, r.text, date))

    return api_results
    

def bea_api_gdpstate(bea_key):
    ''' Return tables in table list for years in range'''
    import requests
    from datetime import datetime

    years = ','.join(map(str, range(2008, 2020)))

    api_results = []

    table = 'RGDP_SQN'
    url = f'https://www.bea.gov/api/data/?&UserID={bea_key}'\
          f'&method=GetData&datasetname=RegionalProduct'\
          f'&IndustryId=1&Component={table}&GeoFIPS=STATE'\
          f'&Year={years}&ResultFormat=json'

    r = requests.get(url)
    
    name = 'GDP by State'
    
    date = (r.json()['BEAAPI']['Results']['Notes'][0]['NoteText']
             .split('--')[0].split(': ')[1])

    date = datetime.strptime(date, '%B %d, %Y').strftime('%Y-%m-%d')

    api_results.append((table, name, r.text, date))

    return api_results

    
def bea_to_db(api_results):
	'''Connect to SQL database and add API results'''
	import sqlite3
	conn = sqlite3.connect('../data/chartbook.db')

	c = conn.cursor()

	c.execute('''CREATE TABLE IF NOT EXISTS bea_nipa_raw(id, name, data, date,
	             UNIQUE(id, date))''')

	c.executemany('INSERT OR IGNORE INTO bea_nipa_raw VALUES (?,?,?,?)', api_results)

	conn.commit()
	conn.close()
	

def retrieve_table(table_id):
    '''Returns table from local database'''
    import json
    import sqlite3
    table_id = (table_id, table_id)
    conn = sqlite3.connect('../data/chartbook.db')
    c = conn.cursor()
    c.execute('''SELECT data FROM bea_nipa_raw WHERE id=? AND 
                 date=(SELECT MAX(date) FROM bea_nipa_raw
                 WHERE id=?)''', table_id)
    data = json.loads(c.fetchone()[0])['BEAAPI']['Results']
    conn.close()
    return data
    
    
def nipa_df(nipa_table, series_list):
    '''Returns dataframe from table and series list'''
    import pandas as pd
    data = {}
    for code in series_list:
    	lineno = [i['LineNumber'] for i in nipa_table if (i['SeriesCode'] == code) & (i['TimePeriod'] == '2016Q4')]
    	obs = [i['DataValue'] for i in nipa_table if (i['SeriesCode'] == code) & (i['LineNumber'] == lineno[0])]
    	index = [pd.to_datetime(i['TimePeriod']) for i in nipa_table if (i['SeriesCode'] == code) & (i['LineNumber'] == lineno[0])]
    	data[code] = (pd.Series(data=obs, index=index).sort_index().str.replace(',', '').astype(float))
        
    return pd.DataFrame(data)
    
def gdpstate_df(table):
    '''Returns dataframe from table and series list'''
    import pandas as pd
    data = {}
    
    series_list = list(set([i['GeoName'] 
                            for i in retrieve_table('RGDP_SQN')['Data']]))
    for code in series_list:
        obs = [i['DataValue'] for i in table
               if i['GeoName'] == code]
        index = [pd.to_datetime(i['TimePeriod']) for i in table 
                 if i['GeoName'] == code]
        data[code] = (pd.Series(data=obs, index=index)
                        .str.replace(',', '').astype(float))
        
    return pd.DataFrame(data)
        
    
def nipa_series_codes(nipa_table):
    '''Return series codes and names from table code, e.g. T20100'''
    r = nipa_table['Data']

    series_dict = {item['SeriesCode']: item['LineDescription'] for item in r}
    
    return series_dict
    
    
def growth_rate(series):
	''' Return the annualized quarterly growth rate in percent'''
	return ((((series.pct_change() + 1) ** 4) - 1) * 100)
	
    
def growth_contrib(df, srs):
    '''Calculate df column contribution to srs growth'''
    dft = df.diff()
    dft = dft.div(dft[srs], axis=0)
    c = dft.multiply((((df[srs].pct_change() + 1) ** 4) - 1) * 100, axis=0)
    return c.round(2)
    
    
def growth_contrib_ann(df, srs):
    '''Calculate df column contribution to srs growth'''
    dft = df.diff(4)
    dft = dft.div(dft[srs], axis=0)
    c = dft.multiply(df[srs].pct_change(4) * 100, axis=0)
    return c.round(2)


def weighted_average(df, variable):
    return np.average(df[variable], weights=df['BASICWGT'])

    
def write_txt(filename, filetext):
    ''' Write label to txt file '''
    with open(filename, 'w') as text_file:
        text_file.write(filetext)
        
      
def dtxt(date):
	'''
	Return strings for given datetime date
	'''
	date = pd.to_datetime(date)
	d = {'qtr1': f'{date.year} Q{date.quarter}', 
	     'qtr2': f'the {qtrs[date.quarter]} quarter of {date.year}',
	     'qtr3': f'Q{date.quarter}',
	     'qtr4': f'`{date.strftime("%y")} Q{date.quarter}',
	     'year': f'{date.year}',
	     'mon1': date.strftime('%B %Y'),
	     'mon2': date.strftime('%b %Y'),
	     'mon3': date.strftime('%B'),
	     'mon4': date.strftime(f'`{date.strftime("%y")} {date.strftime("%b")}'),
	     'day1': date.strftime('%B %-d, %Y'),
	     'day2': date.strftime('%b %-d, %Y'),
	     'day3': date.strftime('%d'),
	     'datetime': date.strftime('%Y-%d-%m')}	
	return d
	

def cont_subt(value, style='main'):
    '''
    Return text for value
    
    -------
    
    Styles:
    
    main: "contributed x.xx percentage points to"
    
    of: "contribution of x.xx percentage points"
    
    end: "contributed x.xx percentage points"
    
    '''
    text = 'ERROR'
    abs_val = abs(value)
    if value > 0:
        if style == 'main':
            text = f'contributed {abs_val:.2f} percentage points to'
        elif style == 'of':
            text = f'contribution of {abs_val:.2f} percentage points'
        elif style == 'end':
            text = f'contributed {abs_val:.2f} percentage points'
    if value < 0:
        if style == 'main':
            text = f'subtracted {abs_val:.2f} percentage points from'
        elif style == 'of':    
            text = f'subtraction of {abs_val:.2f} percentage points'
        elif style == 'end':
            text = f'subtracted {abs_val:.2f} percentage points'
    if value == 0:
        if style == 'main':
            text = 'did not contribute to'
        elif style == 'of':
            text = 'no contribition to'
        elif style == 'end':
            text = 'did not contribute'

    return text 


def series_info(s):
    '''Return info about a pandas series'''
    
    import pandas as pd
    
    obs_per_year = len(s.loc['2017'])
    d = {}
    d['obs'] = len(s)
    d['start'] = s.index[0]
    d['val_latest'] = s.iloc[-1]
    d['date_latest'] = s.index[-1]
    d['mean'] = s.mean()
    d['std'] = s.std()
    d['val_latest_z'] = (s.iloc[-1] - s.mean()) / s.std()
    d['val_max'] = s.max()
    d['date_max'] = s.idxmax()
    d['val_min'] = s.min()
    d['date_min'] = s.idxmin()
    d['val_prev'] = s.iloc[-2]
    d['val_prev_z'] = (s.iloc[-2] - s.mean()) / s.std()
    d['date_prev'] = s.index[-2]
    d['val_year_ago'] = s.iloc[-obs_per_year -1]
    d['date_year_ago'] = s.index[-obs_per_year -1]
    if d['val_latest'] > d['val_prev']:
        dlm = s[s >= d['val_latest']].sort_index()
        if len(dlm) > 1:
        	dlm = dlm.index[-2]
        	dl_txt = 'the highest level since'
        elif len(dlm) == 1:
        	dlm = dlm.index[-1]
        	dl_txt = 'the highest level in the data'
    elif d['val_latest'] < d['val_prev']:
        dlm = s[s <= d['val_latest']].sort_index()
        if len(dlm) > 1:
        	dlm = dlm.index[-2]
        	dl_txt = 'the lowest level since'
        elif len(dlm) == 1:
        	dlm = dlm.index[-1]
        	dl_txt = 'the lowest level in the data'
    else:
        dlm = d['date_prev']
        dl_txt = 'the same level as'
    if obs_per_year == 4:
        dlm_txt = dlm.to_period("Q").strftime("%Y Q%q")
        for key in list(d.keys()):
            if type(d[key]) == pd._libs.tslibs.timestamps.Timestamp:
                d[key + '_ft'] = d[key].to_period("Q").strftime("%Y Q%q")
    elif obs_per_year == 12:
        dlm_txt = dlm.strftime("%B %Y")
        for key in list(d.keys()):
            if type(d[key]) == pd._libs.tslibs.timestamps.Timestamp:
                d[key + '_ft'] = d[key].strftime("%B %Y")
        d['last_3m'] = s.iloc[-3:].mean()
        d['prev_3m'] = s.iloc[-6:-3].mean()
    elif obs_per_year == 1:
        dlm_txt = dlm.strftime("%Y")
    else:
        print("Observations per year error")
        
    d['last_matched'] = f'{dl_txt} {dlm_txt}'
    d['days_since_match'] = (d['date_latest'] - dlm).days
    d['late90s'] = s.loc['1998': '1999'].mean()
    d['one_year_mean'] = s.iloc[-obs_per_year:].mean()
    d['five_year_mean'] = s.iloc[-obs_per_year*5:].mean()
    d['three_year_mean'] = s.iloc[-obs_per_year*3:].mean()
    d['prev_year_mean'] = s.iloc[-obs_per_year*2:-obs_per_year].mean()
    d['change_prev'] = d['val_latest'] - d['val_prev']
    d['change_year_ago'] = d['val_latest'] - d['val_year_ago']
    
    return d


def three_year_growth(data, series):
    '''Annualized growth rate over past three years'''
    return ((data[series].pct_change(36).iloc[-1] + 1)**(1/3)-1) * 100


def bls_api(series, date_range, bls_key):
    """Collect list of series from BLS API for given dates"""
        
    # Import preliminaries
    import requests
    import pandas as pd
    import json
    
    # The url for BLS API v2
    url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

    # API key in config.py which contains: bls_key = 'key'
    key = f'?registrationkey={bls_key}'

    # Handle dates
    dates = [(str(date_range[0]), str(date_range[1]))]
    while int(dates[-1][1]) - int(dates[-1][0]) >= 10:
        dates = [(str(date_range[0]), str(date_range[0] + 9))]
        d1 = int(dates[-1][0])
        while int(dates[-1][1]) < date_range[1]:
            d1 = d1 + 10
            d2 = min([date_range[1], d1 + 9])
            dates.append((str(d1), (d2)))

    df = pd.DataFrame()

    for start, end in dates:
        # Submit the list of series as data
        data = json.dumps({
            "seriesid": list(series.keys()),
            "startyear": start, "endyear": end})

        # Post request for the data
        p = requests.post(
            f'{url}{key}',
            headers={'Content-type': 'application/json'},
            data=data).json()

        for s in p['Results']['series']:
            col = series[s['seriesID']]
            for r in s['data']:
                date = pd.to_datetime(
                    (f"{r['year']}Q{r['period'][-1]}"
                     if r['period'][0] == 'Q'
                     else f"{r['periodName']} {r['year']}"))
                df.at[date, col] = float(r['value'])
    df = df.sort_index()
    # Output results
    print('Post Request Status: {}'.format(p['status']))

    return df
    
    
def binned_wage(group, wage_var='WKWAGE', percentile=0.1, bins=list(np.arange(25, 3000, 50.0)), bin_size=50.0):
    """Return BLS-styled binned decile/percentile wage"""
    
    import pandas as pd
    import numpy as np
    
    # Use ORG weight since wage defined only for ORG sample
    weight = 'PWORWGT'
    
    # Cut wage series according to bins of bin_size
    bin_cut = lambda x: pd.cut(x[wage_var], bins, include_lowest=True)
    
    # Calculate cumulative sum for weight variable
    cum_sum = lambda x: x[weight].cumsum()
    
    # Sort wages then apply bin_cut and cum_sum
    df = (group.sort_values(wage_var)
               .assign(WAGE_BIN = bin_cut, CS = cum_sum))
    
    # Find the weight at the percentile of interest
    pct_wgt = df[weight].sum() * percentile

    # Find wage bin for person nearest to weighted percentile
    pct_bin = df.iloc[df['CS'].searchsorted(pct_wgt)].WAGE_BIN
    
    # Weight at bottom and top of bin
    wgt_btm, wgt_top = (df.loc[df['WAGE_BIN'] == pct_bin, 'CS']
                          .iloc[[0, -1]].values)
    
    # Find where in the bin the percentile is and return that value
    pct_value = ((((pct_wgt - wgt_btm) / 
                   (wgt_top - wgt_btm)) * bin_size) + pct_bin.left)
    
    return pct_value
    
    
def cps_date():
    '''Returns latest month of available data'''
    import os

    cps_loc = '/home/brian/Documents/CPS/data/'
    
    raw_files = [(file[0:3], [f'19{file[3:5]}' 
                              if int(file[3:5]) > 25 
                              else f'20{file[3:5]}'][0]) 
                 for file in os.listdir(cps_loc)
                 if file.endswith('pub.dat')]
    
    dates = (pd.to_datetime([f'{mm}, 1, {yy}' 
                            for mm, yy in raw_files])
              .sort_values())
    
    return dates[-1]
