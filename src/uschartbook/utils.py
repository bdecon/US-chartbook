import numpy as np
import pandas as pd
import requests
import json
import os
import re
from datetime import datetime
import sqlite3

qtrs = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}


def bea_api_nipa(table_list, bea_key):
    ''' Return tables in table list for years in range'''

    years = ','.join(map(str, range(1988, 2020)))

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

    years = ','.join(map(str, range(2008, 2020)))

    api_results = []

    table = 'SQGDP9'
    url = f'https://www.bea.gov/api/data/?&UserID={bea_key}'\
          f'&method=GetData&datasetname=Regional'\
          f'&LineCode=1&TableName={table}&GeoFIPS=STATE'\
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
	conn = sqlite3.connect('../data/chartbook.db')

	c = conn.cursor()

	c.execute('''CREATE TABLE IF NOT EXISTS bea_nipa_raw(id, name, data, date,
	             UNIQUE(id, date))''')

	c.executemany('INSERT OR IGNORE INTO bea_nipa_raw VALUES (?,?,?,?)', api_results)

	conn.commit()
	conn.close()
	

def retrieve_table(table_id):
    '''Returns table from local database'''
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
    data = {}
    for code in series_list:
    	lineno = [i['LineNumber'] for i in nipa_table if (i['SeriesCode'] == code) & (i['TimePeriod'] == '2016Q4')]
    	obs = [i['DataValue'] for i in nipa_table if (i['SeriesCode'] == code) & (i['LineNumber'] == lineno[0])]
    	index = [pd.to_datetime(i['TimePeriod']) for i in nipa_table if (i['SeriesCode'] == code) & (i['LineNumber'] == lineno[0])]
    	data[code] = (pd.Series(data=obs, index=index).sort_index().str.replace(',', '').astype(float))
        
    return pd.DataFrame(data)
    
def gdpstate_df(table):
    '''Returns dataframe from table and series list'''
    data = {}
    
    series_list = list(set([i['GeoName'] 
                            for i in retrieve_table('SQGDP9')['Data']]))
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
	     'mon5': date.strftime('%Y-%m'),
	     'mon6': f'{date.strftime("%b")} `{date.strftime("%y")}',
	     'day1': date.strftime('%B %-d, %Y'),
	     'day2': date.strftime('%b %-d, %Y'),
	     'day3': date.strftime('%d'),
	     'day4': date.strftime('%B %-d'),
	     'datetime': date.strftime('%Y-%m-%d')}	
	return d
	

def cont_subt(value, style='main', digits=2):
    '''
    Return text for value
    
    -------
    
    Styles:
    
    main: "contributed x.xx percentage points to"
    
    of: "contribution of x.xx percentage points"
    
    end: "contributed x.xx percentage points"
    
    '''
    text = 'ERROR'
    if abs(value) > 1:
    	abs_val = '{0:.{1}f} percentage points'.format(abs(value), digits)
    else:
    	abs_val = '{0:.{1}f} percentage point'.format(abs(value), digits)
    if value >= 0.01:
        if style == 'main':
            text = f'contributed {abs_val} to'
        elif style == 'of':
            text = f'contribution of {abs_val}'
        elif style == 'end':
            text = f'contributed {abs_val}'
    elif value <= -0.01:
        if style == 'main':
            text = f'subtracted {abs_val} from'
        elif style == 'of':    
            text = f'subtraction of {abs_val}'
        elif style == 'end':
            text = f'subtracted {abs_val}'
    else:
        if style == 'main':
            text = 'did not contribute to'
        elif style == 'of':
            text = 'no contribition to'
        elif style == 'end':
            text = 'did not contribute'

    return text 


def series_info(s):
    '''Return info about a pandas series'''
    
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
    
    
def fred_df(series, start='1989'):
    url = f'http://research.stlouisfed.org/fred2/series/{series}/downloaddata/{series}.csv'

    df = pd.read_csv(url, index_col='DATE', parse_dates=True, na_values=['.'])

    return df.loc[start:]    
    
    
def end_node(data, color):
    latest = data.iloc[-1]
    date = dtxt(data.index[-1])['datetime']
    text = (f'\\node[label={{0:{{\scriptsize {latest:.1f}\%}}}}, circle, '+
            f'{color}, fill, inner sep=1.0pt] at '+
            f'(axis cs:{date}, {latest:.3f}) {{}};')
    
    return text
    

def val_inc_pp(val):
    if val >= 0.1:
        txt = f'increased by a total of {val:.1f} percentage points'
    elif val <= -0.1:
        txt = f'decreased by a total of {abs(val):.1f} percentage points'
    else:
        txt = 'was virtually unchanged'
        
    return txt
    
    
def cps_12mo(cps_dir, cps_dt, cols):
    '''
    Return 12 months of bd_CPS variables cols ending cps_dt
    '''

    if 'MONTH' not in cols:
        cols = cols + ['MONTH']
    if 'YEAR' not in cols:
        cols = cols + ['YEAR']

    cps_year = cps_dt.year
    cps_month = cps_dt.month
    if cps_month != 12:
        cps_year1 = cps_year - 1
        cps_year2 = cps_year
        df = (pd.read_feather(cps_dir / f'cps{cps_year1}.ft', columns=cols)
              .query('MONTH > @cps_month')
              .append(pd.read_feather(cps_dir / f'cps{cps_year2}.ft', columns=cols)
                        .query('MONTH <= @cps_month')))
    else:
        df = pd.read_feather(cps_dir / f'cps{cps_year}.ft', columns=cols)
        
    return df
    

def cps_1mo(cps_dir, cps_dt, cols):
    '''
    Return 1 month of bd_CPS variables cols ending cps_dt
    '''

    if 'MONTH' not in cols:
        cols = cols + ['MONTH']
    if 'YEAR' not in cols:
        cols = cols + ['YEAR']

    cps_year = cps_dt.year
    cps_month = cps_dt.month
    df = (pd.read_feather(cps_dir / f'cps{cps_year}.ft', columns=cols)
                .query('MONTH == @cps_month'))
        
    return df
    
    
def inc_dec_percent(n, how='main'):
    '''Return short text based on value of n'''
    if how == 'main':
    	return (f'increased by {abs(n):.1f} percent' if n >= 0.1 
        	else f'decreased by {abs(n):.1f} percent' 
        	if n <= -0.1 else 'was virtually unchanged')
    elif how == 'of':
    	return (f'an increase of {abs(n):.1f} percent' if n >= 0.1 
        	else f'a decrease of {abs(n):.1f} percent' 
        	if n <= -0.1 else 'virtually no change')
        
        
def compare_text(latest, previous, cutoffs):
    '''Simple text based on difference between two numbers'''
    direction = 'above' if latest - previous > 0 else 'below'
    size = abs(latest - previous)
    if type(cutoffs) not in [list, tuple] or len(cutoffs) != 3:
        print('Cutoffs should be list of four numeric values')
        
    if size <= cutoffs[0]:
        text = 'in line with'
    elif size <= cutoffs[1]:
        text = f'slightly {direction}'
    elif size <= cutoffs[2]:
        text = f'substantially {direction}'
    else:
        text = f'far {direction}'
    
    return text
    
    
def clean_fed_data(url):
    raw_data = pd.read_csv(url)

    d = {v: re.sub("\s+[\(\[].*?[\)\]]", "", i.split(';')[0]) 
         for i, v in raw_data.iloc[4, 1:].iteritems()}

    date_column = raw_data.loc[5:, 'Series Description']
    date_index = pd.to_datetime(date_column).rename('Date')
    columns = raw_data.iloc[4, 1:].values
    clean_data = raw_data.iloc[5:, 1:].replace('ND', np.NaN).astype('float')
    clean_data.index = date_index
    clean_data.columns = columns
    
    return (d, clean_data)
