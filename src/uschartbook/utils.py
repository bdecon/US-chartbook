import numpy as np
import pandas as pd
import requests
import json
import io
import os
import re
from datetime import datetime
import sqlite3

qtrs = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}

numbers = {'1.0': 'one', '2.0': 'two', '3.0': 'three', 
           '4.0': 'four', '5.0': 'five', 
           '6.0': 'six', '7.0': 'seven', 
           '8.0': 'eight', '9.0': 'nine'}

numbers2 = {0: 'no', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 
           6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
           
def to_date(ym):
    return pd.to_datetime(f'{ym[0]}-{ym[1]}-01')

def bea_api_nipa(table_list, bea_key, freq='Q'):
    ''' Return tables in table list for years in range'''

    years = ','.join(map(str, range(1988, 2024)))

    api_results = []

    for table in table_list:
        url = f'https://apps.bea.gov/api/data/?&UserID={bea_key}'\
              f'&method=GetData&datasetname=NIPA&TableName={table}'\
              f'&Frequency={freq}&Year={years}&ResultFormat=json'
              
        r = requests.get(url)

        name, date = (r.json()['BEAAPI']['Results']['Notes'][0]['NoteText']
                       .split(' - LastRevised: '))

        date = datetime.strptime(date, '%B %d, %Y').strftime('%Y-%m-%d')

        api_results.append((table, name, r.text, date))

    return api_results
    

def bea_api_gdpstate(bea_key):
    ''' Return tables in table list for years in range'''

    years = ','.join(map(str, range(2008, 2024)))

    api_results = []

    table = 'SQGDP9'
    url = f'https://apps.bea.gov/api/data/?&UserID={bea_key}'\
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

    
def bea_api_ita(ind_list, bea_key):
    ''' Return tables in table list for years in range'''
    years = ','.join(map(str, range(1988, 2024)))

    api_results = []

    for ind in ind_list:
        url = f'https://apps.bea.gov/api/data/?&UserID={bea_key}'\
              f'&method=GetData&datasetname=ITA&Indicator={ind}'\
              f'&Frequency=QSA&Year={years}&ResultFormat=json'

        r = requests.get(url)

        api_results.append((ind, r.text))

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
    	lineno = [i['LineNumber'] for i in nipa_table if (i['SeriesCode'] == code) & (i['TimePeriod'] in ['2016Q4', '2016', '1998Q4'])]
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


def growth_rate_monthly(series):
	''' Return the annualized growth rate in percent for a monthly series'''
	return ((((series.pct_change() + 1) ** 12) - 1) * 100)
	

def cagr(s, freq='Q'):
    '''
    Return compound annual growth rate for series
    '''
    if freq == 'Q':
        p = 4
    elif freq == 'M':
        p = 12
    elif freq == 'A':
        p = 1
    n = len(s) - 1
    r = ((s.iloc[-1] / s.iloc[0])**(p/n) - 1) * 100
    return r
		
    
def growth_contrib(df, srs):
    '''Calculate df column contribution to srs growth'''
    dft = df.diff()
    dft = dft.div(dft[srs], axis=0)
    c = dft.multiply((((df[srs].pct_change() + 1) ** 4) - 1) * 100, axis=0)
    return c.round(2)
    
    
def growth_contrib_ann(df, srs, freq='Q'):
    '''Calculate df column contribution to srs growth'''
    freq_d = {'Q': 4, 'M': 12, 'A': 1}
    freq_n = freq_d[freq]
    dft = df.diff(freq_n)
    dft = dft.div(dft[srs], axis=0)
    c = dft.multiply(df[srs].pct_change(freq_n) * 100, axis=0)
    return c.round(2)


def weighted_average(df, variable):
    return np.average(df[variable], weights=df['BASICWGT'])


def m3rate(series):
    '''Three-month / three-month growth rate'''
    return ((((series.rolling(3).mean() / 
               series.rolling(3).mean().shift(3)) 
              ** 4) - 1) * 100).dropna(how='all')
              
                  
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
	     'qtr5': f'the {qtrs[date.quarter]} quarter',
	     'year': f'{date.year}',
	     'mon1': date.strftime('%B %Y'),
	     'mon2': date.strftime('%b %Y'),
	     'mon3': date.strftime('%B'),
	     'mon4': date.strftime(f'`{date.strftime("%y")} {date.strftime("%b")}'),
	     'mon5': date.strftime('%Y-%m'),
	     'mon6': f'{date.strftime("%b")} `{date.strftime("%y")}',
	     'mon7': f'{date.strftime("%b")} {date.strftime("%y")}',
	     'mon8': f'{date.strftime("%b")} \n {date.strftime("%Y")}',
	     'day1': date.strftime('%B %-d, %Y'),
	     'day2': date.strftime('%b %-d, %Y'),
	     'day3': date.strftime('%d'),
	     'day4': date.strftime('%B %-d'),
	     'datetime': date.strftime('%Y-%m-%d'),	
	     'datetime2': date.strftime('%Y-%m-%d').replace('-08-', '-8-').replace('-09-', '-9-')}	
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
    print('cont_subt will be removed from a future version')
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
        d['change_3m_3m'] = (((d['last_3m'] / d['prev_3m'])**4) - 1) * 100
    elif obs_per_year == 1:
        dlm_txt = dlm.strftime("%Y")
    elif obs_per_year > 100:
    	dlm_txt = ''
    	print("Observations per year error")
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
    print('three_year_growth() to be removed in future version')
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
    

def binned_wage(df, wage_var='WKEARN', percentile=0.1, wgt_var='PWORWGT',
                bins=np.arange(-25, 3000, 50)):
    '''
    Returns wage estimate based on linear interpolation through 
    the bin containing the wage.
    
    perc = percentile of interest (0.5 is median)
    bins = list of bin start locations
    '''
    cdf = (df.groupby(pd.cut(df[wage_var], bins))
             [wgt_var].sum().cumsum() / df[wgt_var].sum())
    
    return np.interp(percentile, cdf, bins[1:])
        

def median_age(df, wgt='PWSSWGT', percentile=0.5):
    '''
    Returns age associated with given percentile.
    
    Default is median (0.5).
    '''
    bins=np.arange(-1, 86, 1)
    cdf = (df.groupby(pd.cut(df.AGE, bins))
             [wgt].sum().cumsum() / df[wgt].sum())
    
    return np.interp(percentile, cdf, bins[1:])

    
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
    

def c_line(color, see=True, paren=True):
	'''Return (see ---) for a given color'''
	s = 'see ' if see == True else ''
	p = ['(', ')'] if paren == True else ['', '']
	cl = f'{p[0]}{s}{{\color{{{color}}}\\textbf{{---}}}}{p[1]}'
	return cl
	
	
def c_box(color, see=True):
	'''Return (see []) for a given color'''
	s = 'see' if see == True else '\hspace{-1mm}'
	return f'({s}\cbox{{{color}}})'
    
    
def end_node(series, color, percent=False, date=None, offset=0, xoffset=0,
             anchor=None, digits=1, full_year=False, dollar=False,
             colon=True, align='left', loc='end', size=1.0, italic=False):
    '''
    Generate small dot and value text for end of line plot.
    Input is pandas series and color. Output is tex code. 
    Options allow for percent sign, date, y offset, or 
    switching anchor south (text above node).
    '''  
    anchor_opt = ''
    if loc == 'end':
    	i = -1
    elif loc == 'start':
    	i = 0
    	align='right'
    	anchor_opt = 'anchor=east, '
    if anchor != None:
        if anchor.lower() not in ['south', 'north']:
            print('Anchor should be south or north')
        else:
            anchor_opt = f'anchor={anchor.lower()}, '
        
    col = ':' if colon == True else ''
    
    pct = '' if percent == False else '\%'
        
    dt = ''    
    if date != None:
        if date.lower() not in ['month', 'mon', 'm', 'year', 'yr', 'y', 
        						'fy', 'day', 'd', 'ds', 'dayshort', 'short', 's',
                                'quarter', 'qtr', 'q', 'qtrshort', 'qshort', 'qs']:
            print('Date should be month or quarter or year or short')
        yr = series.index[i].strftime('`%y')
        if full_year == True:
        	yr = series.index[i].strftime('%Y')
        if date.lower() in ['qtrshort', 'qshort', 'qs']:
            yr = series.index[i].strftime('%y')
        qtr = series.index[i].to_period('Q').strftime('Q%q')
        mo = series.index[i].strftime('%b')
        day = series.index[i].strftime('%-d')
        daysh = series.index[i].strftime('%-d')
        if date.lower() in ['month', 'mon', 'm']:
            dt = f'\scriptsize {mo}\\\\ \scriptsize {yr}{col} \\\\ '
        elif date.lower() in ['quarter', 'qtr', 'q']:
            dt = f'\scriptsize {yr}\\\\ \scriptsize {qtr}{col} \\\\ '
        elif date.lower() in ['qtrshort', 'qshort', 'qs']:
            dt = f'\scriptsize {yr} \scriptsize {qtr}{col} \\\\ '
        elif date.lower() in ['year', 'yr', 'y']:
            dt = f'\scriptsize {yr}{col} \\\\ '
        elif date.lower() == 'fy':
            dt = f'\scriptsize FY{yr} \\\\ '.replace('`', '')
        elif date.lower() in ['d', 'day']:
        	dt = f'\scriptsize {mo} {day}\\\\ \scriptsize {yr}{col} \\\\ '
        elif date.lower() in ['ds', 'dayshort', 'short', 's']:  # Day short format
        	dt = f'\scriptsize {mo} {daysh}{col}\\\\'
        
    lt = series.iloc[i]
    vtxt = f'{lt:.1f}'
    if digits == 2:
    	vtxt = f'{lt:.2f}'
    elif digits == 3:
    	vtxt = f'{lt:.3f}'
    elif digits == 0:
    	vtxt = f'{lt:.0f}'
    elif digits == 'comma':
    	vtxt = f'{lt:,.0f}'
    elif loc == 'start':
    	vtxt = f'{vtxt} '
    	
    dol = ''
    if dollar == True:
    	dol = '\$'
    elif dollar == 'thousands':
    	dol = '\$'
    	vtxt = f'{lt * 1000:,.0f}'
    	
    it = ''
    if italic == True:
    	it = '\\textit'
    
    if (offset == True) and (date != None):
        vmax = series.max()
        vmin = series.min()
        vrng = vmax - vmin
        offset = (-0.35 if lt > vmax - (vrng*0.1) 
                  else 0.35 if lt < vmin + (vrng*0.1) 
                  else 0)
    elif (offset == True) and (date == None):
        offset = 0
    offs = f'{offset}cm'
    offx = f'{xoffset}cm'
    datetime = dtxt(series.index[i])['datetime']
    text = (f'\\node[label={{[yshift={offs}, xshift={offx}, {anchor_opt}'+
            f'align={align}]0:{{{dt}\scriptsize {it}{dol}{vtxt}{pct}}}}}, circle, '+
            f'{color}, fill, inner sep={size}pt] at '+
            f'(axis cs:{datetime}, {lt}) {{}};')

    return text
    

def node_adjust(df, color_dict):
    '''Return offsets for node text'''
    df = df[color_dict.keys()]
    std = df.std().max()
    lt = df.iloc[-1]
    adj_list = lt[lt.sort_values().diff() < std]
    d = {name: abs(std/2) for name in adj_list.index}
    return d


def node_adj(df):
    '''
    Return dict with adjustment values for end node
    for each column in df.
    '''
    d = {name: 0 for name in df.columns}
    r = df.iloc[-1].sort_values()
    u = (df.max().max() - df.min().min()) / 16
    if len(df.columns) >= 4:
        if len(r.diff(3)[r.diff(3) < (u*3)]) > 0:
            print('Too many nodes conflicting, '+
                  'no results returned')
            return d
    if len(df.columns) >= 3:
        t3 = r.diff(2)[r.diff(2) < (u*2)]
        if len(t3) > 0:
            print('Three conflicting nodes')
            t2 = r.diff()[r.diff() < (u)]
            if len(t2) == 2:
                g1 = (((u) - t2[0]) / u) * 0.35
                i = r.index[r.index.get_loc(t2.index[0]) - 1]
                d[i] = - (g1)
                g2 = (((u) - t2[1]) / u) * 0.35
                d[t2.index[-1]] = (g2)
            if len(t2) == 1:
                g = ((u - t2[0]) / u) * 0.35
                if t3.index[0] == t2.index[0]:
                    d[t3.index[0]] = g
                if t3.index[0] != t2.index[0]:
                    bloc = r.index[r.index.get_loc(
                        t2.index[0]) - 1]
                    d[bloc] = - g
            return d
    if len(df.columns) >= 2:
        t2 = r.diff()[r.diff() < (u)]
        if len(t2) == 1:
            g = ((u - t2[0]) / u) * 0.35
            d[t2.index[0]] = (g/2)
            d[r.index[r.index.get_loc(t2.index[0]) - 1]] = - (g/2)
        if len(t2) == 2:
            g1 = ((u - t2[0]) / u) * 0.35
            d[t2.index[0]] = (g1/2)
            d[r.index[r.index.get_loc(t2.index[0]) - 1]] = - (g1/2) 
            g2 = ((u - t2[1]) / u) * 0.35
            d[t2.index[1]] = (g2/2)
            d[r.index[r.index.get_loc(t2.index[1]) - 1]] = - (g2/2)
            print('two sets of conflicting nodes')
        return d
    if len(df.columns) == 1:
        name = r.index[0]
        if df[name].iloc[-1] > (df[name].max() - u):
            d[name] = -0.35
        if df[name].iloc[-1] < (df[name].min() + u):
            d[name] = 0.35
        return d


def val_inc_pp(val, threshold=0.1):
    if threshold >= 0.1:
        format_string = '.1f'
    else:
        format_string = '.2f'
    if abs(val) > 1.05:
        pp = 'percentage points'
    else:
        pp =  'percentage point'
    if val >= threshold:
        txt = f'increased by a total of {val:{format_string}} {pp}'
    elif val <= -threshold:
        txt = f'decreased by a total of {abs(val):{format_string}} {pp}'
    else:
        txt = 'was virtually unchanged'
    
    print('val_inc_pp will be removed from a future version')    
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
    

def cps_3mo(cps_dir, cps_dt, cols):
    '''
    Return 3 months of bd_CPS variables cols ending cps_dt
    '''

    if 'MONTH' not in cols:
        cols = cols + ['MONTH']
    if 'YEAR' not in cols:
        cols = cols + ['YEAR']

    cps_year = cps_dt.year
    cps_month = cps_dt.month
    cps_month3 = (cps_dt - pd.DateOffset(months=2)).month
    if cps_month < 3:
        cps_year1 = cps_year - 1
        cps_year2 = cps_year
        df = (pd.read_feather(cps_dir / f'cps{cps_year1}.ft', columns=cols)
              .query('MONTH >= @cps_month3')
              .append(pd.read_feather(cps_dir / f'cps{cps_year2}.ft', columns=cols)
                        .query('MONTH <= @cps_month')))
    else:
        df = (pd.read_feather(cps_dir / f'cps{cps_year}.ft', columns=cols)
              .query('MONTH >= @cps_month3 and MONTH <= @cps_month'))
        
    return df
    

def cps_6mo(cps_dir, cps_dt, cols):
    '''
    Return 6 months of bd_CPS variables cols ending cps_dt
    '''

    if 'MONTH' not in cols:
        cols = cols + ['MONTH']
    if 'YEAR' not in cols:
        cols = cols + ['YEAR']

    cps_year = cps_dt.year
    cps_month = cps_dt.month
    cps_month6 = (cps_dt - pd.DateOffset(months=5)).month
    if cps_month < 6:
        cps_year1 = cps_year - 1
        cps_year2 = cps_year
        df = (pd.read_feather(cps_dir / f'cps{cps_year1}.ft', columns=cols)
              .query('MONTH >= @cps_month6')
              .append(pd.read_feather(cps_dir / f'cps{cps_year2}.ft', columns=cols)
                        .query('MONTH <= @cps_month')))
    else:
        df = (pd.read_feather(cps_dir / f'cps{cps_year}.ft', columns=cols)
              .query('MONTH >= @cps_month6 and MONTH <= @cps_month'))
        
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
    
    
def inc_dec_percent(n, how='main', annualized=False):
    '''Return short text based on value of n'''
    atxt1 = 'by'
    atxt2 = ''
    if annualized == True:
    	atxt1 = 'at an annual rate of'
    	atxt2 = ' (annualized)'
    if how not in ['of', 'main']:
    	print('Options: "of" or "main", annualized: False by default')
    elif how == 'main':
    	return (f'increased {atxt1} {abs(n):.1f} percent' if n >= 0.1 
        	else f'decreased {atxt1} {abs(n):.1f} percent' 
        	if n <= -0.1 else 'was virtually unchanged')
    elif how == 'of':
    	return (f'an increase of {abs(n):.1f} percent{atxt2}' if n >= 0.1 
        	else f'a decrease of {abs(n):.1f} percent{atxt2}' 
        	if n <= -0.1 else 'virtually no change')
    print('inc_dec_percent will be removed from a future version')
        
        
def compare_text(latest, previous, cutoffs, plain=False):
    '''
    Simple text based on difference between two numbers.
    Cutoffs should be list of three numbers that provide scale for 
    how significant the difference is. 
    '''
    direction = 'above' if latest - previous > 0 else 'below'
    size = abs(latest - previous)
    if type(cutoffs) not in [list, tuple] or len(cutoffs) != 3:
        print('Cutoffs should be list of three numeric values')
        
    if size <= cutoffs[0]:
        text = 'in line with'
    elif size <= cutoffs[1]:
        text = f'slightly {direction}'
    elif size <= cutoffs[2]:
        text = f'substantially {direction}'
    else:
        text = f'far {direction}'
    
    if plain == True:
        text = direction
        
    return text
    
    
def clean_fed_data(url, dtype='main'):
    s = requests.get(url).content
    raw_data = pd.read_csv(io.StringIO(s.decode('utf-8')))

    d = {v: re.sub("\s+[\(\[].*?[\)\]]", "", i.split(';')[0]) 
         for i, v in raw_data.iloc[4, 1:].items()}
    if dtype == 'full':
        d = {v: re.sub("\s+[\(\[].*?[\)\]]", "", ''.join(i.split(';')[0:])) 
             for i, v in raw_data.iloc[4, 1:].items()}

    date_column = raw_data.loc[5:, 'Series Description']
    date_index = pd.to_datetime(date_column).rename('Date')
    columns = raw_data.iloc[4, 1:].values
    clean_data = raw_data.iloc[5:, 1:].replace('ND', np.NaN).astype('float')
    clean_data.index = date_index
    clean_data.columns = columns
    
    return (d, clean_data)
    
    
def jolts_codes(d, code_text, ind, value='i'):
    '''Input dictionary, code_text (for example JOR) and industry dict'''
    for code, name in ind.items():
        i = 'JTS' + code + '000000000' + code_text
        d[i] = i
        if value == 'name':
            d[i] = name
    return d

    
def value_text(value, style='increase', ptype='percent', adj=None, 
               time_str='', digits=1, threshold=0, num_txt=True,
               casual=False, obj='singular', dollar=False, 
               round_adj=False):
    '''
    RETURN TEXT STRING FOR SPECIFIED FLOAT VALUE
    
    OPTIONS
    style: increase, increase_of, contribution, contribution_to,
           contribution_of, contribution_end
    ptype: percent, pp, None, million, etc
    adj: sa, annual, annualized, saa, saar, total, average
    time_str: blank unless specified directly, for example "one-year"
    num_txt: replace round numbers with text, for example: 9.0 -> nine
    casual: replaces certain words: decreased -> fell, for example
    obj: switch to "plural" if the object is plural, e.g. prices
    round_adj: adds "nearly" to values below the rounded value
    
    '''
    text = 'Error, options not available'
    dol = '' if dollar == False else '\$'
    abv = abs(value)
    val = f'{dol}{abv:,.{digits}f}'
    val2 = f'{dol}{value:,.{digits}f}'
    numbers = {'1.0': 'one', '2.0': 'two', '3.0': 'three', 
               '4.0': 'four', '5.0': 'five', 
               '6.0': 'six', '7.0': 'seven', 
               '8.0': 'eight', '9.0': 'nine',
               '10.0': 'ten'}
    if (num_txt == True) & (val in numbers.keys()):
        val = numbers[val] 
    indef = 'an' if ((val[0] == '8') | (val[0:3] in ['11.', '11,', '18.', '18,'])) else 'a'
    neg = True if value < 0 else False
    insig = True if abv < threshold else False
    plural = 's' if ((abv > 1.045) & (style[-3:] != 'end')) else ''
    ptxtd = {None: '', 'none': '', 'None': '', '': '', 'percent': ' percent', 
             'pp': f' percentage point{plural}', 'point': f' point{plural}',
             'trillion': ' trillion', 'billion': ' billion', 'million': ' million', 
             'thousand': ' thousand', 'units': ' units'}
    ptxt = ptxtd[ptype]
    rnd_adj = ('' if ((round_adj == False) | (abv >= round(abv, digits))) 
    		   else 'nearly ' if casual == False else 'almost ')
    
    if style in ['increase', 'increase_by', 'gain', 'return']:
        atxtd = {None: ' by ', 'sa': ' at a seasonally-adjusted rate of ', 
                 'annual': ' at an annual rate of ', 
                 'annualized': ' at an annualized rate of ', 
                 'average_annualized': ' at an average annualized rate of ',
                 'avg_ann': ' at an average annualized rate of ',
                 'saa': ' at a seasonally-adjusted and annualized rate of ', 
                 'saar': ' at a seasonally-adjusted annualized rate of ', 
                 'total': ' by a total of ', 
                 'inflation': ' the inflation rate by ',
                 'average': ' at an average rate of ',
                 'equivalent': ' by the equivalent of '}
        if style != 'increase_by':
            atxtd[None] = ' '
        if style in ['gain', 'return']:
        	atxtd['total'] = ' a total of '
        atxt = atxtd[adj]
        stxt = 'increased' if neg == False else 'decreased'
        if style == 'gain':
        	stxt = 'gained' if neg == False else 'lost'
        if style == 'return':
        	stxt = 'returned' if neg == False else 'lost'
        if adj == 'inflation':
        	stxt = 'increased' if neg == False else 'reduced'
        ttxt = f' over the {time_str} period' if time_str != '' else ''
        text = f'{stxt}{atxt}{val}{ptxt}{ttxt}'
        if insig == True:
            text = 'was virtually unchanged'
            if obj == 'plural':
                text = 'were unchanged'
            
    if style in ['contribution', 'contribution_to']:
        atxtd = {None: '', 'sa': ' on a seasonally-adjusted basis', 
                 'annual': ' on an annual basis', 
                 'annualized': ' on an annualized-basis', 
                 'average_annualized': ' on an average annualized basis ',
                 'avg_ann': ' on an average and annualized rate basis ',
                 'saa': ' on a seasonally-adjusted and annualized basis', 
                 'saar': ' on a seasonally-adjusted annualized basis', 
                 'total': ' in total',
                 'average': ' on an average basis'}
        atxt = atxtd[adj]
        atxt2 = 'the equivalent of ' if adj == 'equivalent' else ''
        stxt = ('contributed', 'to') if neg == False else ('subtracted', 'from')
        ttxt = f' over the {time_str} period' if time_str != '' else ''
        text = f'{stxt[0]} {atxt2}{val}{ptxt}{atxt}{ttxt}'
        if style == 'contribution_to':
            text = f'{stxt[0]} {atxt2}{val}{ptxt} {stxt[1]}'
        if insig == True:
            text = 'did not contribute'
            if style == 'contribution_to':
                text = 'did not contribute to'
            
    elif style in ['increase_of', 'contribution_of', 'return_of']:
        stxt1 = 'increase' if neg == False else 'decrease'
        stxt2 = 'an increase' if neg == False else 'a decrease'
        if style == 'contribution_of':
            stxt1 = 'contribution' if neg == False else 'subtraction'
            stxt2 = 'a contribution' if neg == False else 'a subtraction'    
        if style == 'return_of':
            stxt1 = 'return' if neg == False else 'loss'
            stxt2 = 'a return' if neg == False else 'a loss'   
        if style == 'gain_of':
            stxt1 = 'gain' if neg == False else 'loss'
            stxt2 = 'a gain' if neg == False else 'a loss'             
        if time_str != '':
            stxt2 = f'a {time_str}{stxt1}'
        atxtd = {None: f'{stxt2} of', 'sa': f'a seasonally-adjusted {time_str}{stxt1} of', 
                 'annual': f'an annual {time_str}{stxt1} of', 
                 'annualized': f'an annualized {time_str}{stxt1} of', 
                 'average_annualized': f' an average annualized {time_str}{stxt1} of',
                 'avg_ann': f' an average annualized {time_str}{stxt1} of',
                 'saa': f'a seasonally-adjusted and annualized {time_str}{stxt1} of', 
                 'saar': f'a seasonally-adjusted annualized {time_str}{stxt1} of', 
                 'total': f'a total {time_str}{stxt1} of',
                 'average': f'an average {time_str}{stxt1} of'}
        atxt = atxtd[adj]
        atxt2 = 'the equivalent of ' if adj == 'equivalent' else ''
        text = f'{atxt} {atxt2}{val}{ptxt}'
        if insig == True:
            text = 'virtually no change'
            if style[:3] == 'con':
                text = 'virtually no contribution'
            
    elif style in ['increase_end', 'contribution_end']:
        stxt = 'increase' if neg == False else 'decrease'
        if style == 'contribution_end':
            stxt = 'contribution' if neg == False else 'subtraction'
        atxtd = {None: f'{indef} ', 'sa': 'a seasonally-adjusted ', 
                 'annual': 'an annual ', 
                 'annualized': 'an annualized ', 
                 'average_annualized': ' an average annualized ',
                 'avg_ann': ' an average and annualized ',
                 'saa': 'a seasonally-adjusted and annualized ', 
                 'saar': 'a seasonally-adjusted annualized ', 
                 'total': 'a total ',
                 'average': 'an average '}
        atxt = atxtd[adj]
        ttxt = f'{time_str} ' if time_str != '' else ''
        text = f'{atxt}{ttxt}{val}{ptxt} {stxt}'
        if insig == True:
            text = 'virtually no change'
            if style[:3] == 'con':
                text = 'virtually no contribution'
                
    elif style == 'above_below':
        stxt = 'above' if neg == False else 'below'
        text = f'{val}{ptxt} {stxt}'
        if insig == True:
            text = 'in line with'
            
    elif style == 'plain':
    	val3 = val
    	pn = '' if neg == False else 'negative '
    	# Handle rounded values
    	num_abv = {k[0]: v for k, v in numbers.items()}
    	if (num_txt == True) & (val in num_abv.keys()):
        	val3 = num_abv[val] 
        	if float(val) > value:
        		rnd_adj = 'nearly ' if casual == False else 'almost '
    	text = f'{rnd_adj}{pn}{val3}{ptxt}'
    
    elif style in ['equivalent', 'eq']:
    	atxt = ' of GDP' if adj in ['gdp', 'GDP'] else ''
    	text = f'equivalent to {val}{ptxt}{atxt}'
    	
    elif style == 'added_lost':
    	stxt = 'added' if neg == False else 'lost'
    	atxtd = {None: '', 'average': 'an average of '}
    	atxt = atxtd[adj]
    	text = f'{stxt} {atxt}{val}{ptxt}'
    	
    elif style == 'added_lost_rev':
    	stxt = 'added' if neg == False else 'lost'
    	text = f'{val}{ptxt} {stxt}'
    
    if casual == True:
        text = (text.replace('added', 'gained')
        		    .replace('decreased', 'fell')
                    .replace('contributed', 'added')
                    .replace('increased', 'grew')
                    .replace('contribute ', 'add ')
                    .replace('subtract ', 'remove ')
                    .replace('a contribution', 'an addition')
                    .replace('contribution', 'addition')
                    .replace('decrease', 'fall')
                    .replace('subtraction', 'reduction')
                    .replace('an increase of', 'growth of')
                    .replace('increase of', 'growth of')
                    .replace('decrease of', 'fall of'))
                    
    if obj == 'plural':
        text = (text.replace('was', 'were'))
    
    return(text)
    
    
def gc_desc(lt, mu, sigma, also=False):
    '''Describe contribution to growth of 3-5 categories'''
    m, tot, sh = lt.mean(), lt.sum(), (lt / lt.sum()).sort_values()
    
    # Special case for offset adjective
    co_sh2 = abs(lt.loc[sh.index[0]]) / abs(lt.loc[lt.index != sh.index[0]].sum())
    
    # Add word also to reduce repetitiveness
    also_t = '' if also == False else 'also '
    
    # Describe overall growth (sum of contributions)
    desc = 'growth' if tot > 0 else 'decrease'
    adj = ('low ' if ((abs(tot) < (sigma / 2)) & (tot > 0)) 
           else 'small ' if ((abs(tot) < abs(mu)) & (tot < 0)) 
           else 'strong ' if (abs(tot) > (sigma*3)) else '')
    overall = 'low/none' if ((adj == 'low ') | (adj == 'small ')) else 'not low'
    desc = f'{adj}{desc}'
    
    # Identify supporting and conflicting categories and sort by size
    conf = lt[lt < 0].index.to_list() if tot >= 0 else lt[lt > 0].index.to_list()
    supp = lt[~lt.index.isin(conf)].index.to_list()
    incdec = 'an increase in ' if tot >= 0 else 'a decrease in '
    decinc = 'a decrease in ' if tot >= 0 else 'an increase in '
    xl = lt[abs(lt) > (sigma * 2)]
    xlincdec = 'a large increase in ' if tot >= 0 else 'a large decrease in '
    xldecinc = 'a large decrease in ' if tot >= 0 else 'a large increase in '
    lg = lt[abs(lt) > (sigma/2)]
    sg = lt[(abs(lt) > (sigma/6)) | # Somewhat large or a large share
            ((abs(lt) > (sigma/10)) & ((abs(lt) / lt.sum()) > 0.33))]
    csg = [(c, i) for c, i in sg.items() if c in conf] if len(conf) > 0 else []
    cxl = [(c, i) for c, i in xl.items() if c in conf] if len(xl) > 0 else []
    ssg = [(c, i) for c, i in sg.items() if c in supp] if len(supp) > 0 else []
    sxl = [(c, i) for c, i in xl.items() if c in supp] if len(xl) > 0 else []
    # Determine supporting/conflicting type
    if (len(csg) > 0) & (len(ssg) > 0):
        sct = 'sc'
    elif (len(csg) > 0) & (len(ssg) == 0):
        sct = 'co'
    elif (len(ssg) > 0) & (len(csg) == 0):
        sct = 'su'

    # Adjective for top category (conflicting and supporting)
    sha_co = abs(sh.sort_values()).iloc[0]
    sha_co2 = abs(sh.sort_values()).iloc[1]
    sha_su = abs(sh.sort_values()).iloc[-1]
    sha_su2 = abs(sh.sort_values()).iloc[-2]
    co_adj = ('' if sha_co > 1.5 else 'largely ' if (sha_co >= 1) & (co_sh2 > 0.85) else 'partially ' 
              if sha_co > 0.1 else 'slightly ')
    su_adj = 'largely ' if (sha_su > 0.75) & (sha_su2 < 0.5) else ''    
    co_l = (f'{decinc}{sh.index[0]}' if (co_adj == 'largely ' ) | 
            (len(conf) == 1) | (len(csg) == 1)
            else f'{decinc}{sh.index[0]} and {sh.index[1]}' 
            if (len(conf) > 1) & (len(csg) > 1) else
            f'{decinc}{sh.index[0]}, {sh.index[1]}, and {sh.index[2]}' 
            if (len(conf) > 2) & (len(csg) > 2) else '')
    su_l = (f'{incdec}{sh.index[-1]}' if (su_adj == 'largely ') | (len(lg) == 1)
            else f'{incdec}{sh.index[-1]} and {sh.index[-2]}')
    # Broad-based, category-driven, or conflicting 
    ssr = (((lt - m)**2).sum() / m**2)
    tsh = sh.iloc[-1]
    t2sh = sh.iloc[-2:].sum()
    t3sh = sh.iloc[-3:].sum()
    o = sh.index[-1] if (tsh > 0.66) else False
    same_sign = not min(lt) < 0 < max(lt)
    bbdb, bbdb_t, su_t, co_t = '', '', '', ''
    
    # By growth type, determine broad-based, driven-by, or conflicting
    if overall == 'not low':
        if (len(conf) == 0) & (ssr < (sigma*2)) & (t2sh < 0.8) & (tsh < 0.55):
            bbdb = 'bb1'
            bbdb_t = 'broad-based'
            su_t = ', with categories contributing relatively evenly.'
            if (ssr < sigma) & (o == False) & (len(sxl) == 1):
                su_t = ', with categories contributing relatively evenly, '
                co_t = f'and {xlincdec}{sh.index[-1]}. '
            elif (ssr < sigma) & (o == False) & (len(sxl) == 2):
                su_t = ', with categories contributing relatively evenly, '
                co_t = f'and {xlincdec}{sh.index[-1]} and {sh.index[-2]}.'
            elif (ssr < sigma) & (o == False) & (len(sxl) == 3):
                su_t = ', with categories contributing relatively evenly, '
                co_t = f'and {xlincdec}{sh.index[-1]}, {sh.index[-2]}, and {sh.index[-3]}.'
        elif (ssr < sigma * 2) & (t2sh < 0.95) & (tsh < 0.75):
            bbdb = 'bb2'
            bbdb_t = 'relatively broad-based'
        elif (((ssr < sigma * 10) & (t2sh < 1.1) & (tsh < 0.8)) | 
              ((same_sign == True) & (t2sh < 0.95))): 
            bbdb = 'bb3'
            bbdb_t = 'relatively broad-based'
        elif (tsh > 0.75) & (t2sh < tsh + 0.5):
            bbdb = 'db1'
            incdec_t = xlincdec if sxl == 1 else incdec
            bbdb_t = f'driven {su_adj}by {incdec_t}{sh.index[-1]}'
            if (t2sh > tsh + 0.3):
                su_t = f', and supported by {incdec}{sh.index[-2]}'
        elif (t2sh > 0.75):
            bbdb = 'db2'
            incdec_t = xlincdec if sxl == 2 else incdec
            bbdb_t = f'driven by {incdec_t}{sh.index[-1]} and {sh.index[-2]}'
        elif (t3sh > 0.66):
            bbdb = 'db3'
            incdec_t = xlincdec if sxl == 3 else incdec
            bbdb_t = f'driven by {incdec_t}{sh.index[-1]}, {sh.index[-2]}, and {sh.index[-3]}'
    elif overall == 'low/none':
        if len(sg) == 0: # All low-growth
            bbdb = 'co1'
            bbdb_t = 'the result of little change across several categories.'
        elif (len(csg) > 0) & (len(ssg) > 0):
            bbdb = 'co2'
            bbdb_t = 'the result of conflicting changes in subcategories.'
            su_t = f' The overall effect is {su_adj}the result of {su_l}'
            co_t = f' that is {co_adj}offset by {co_l}.'
        else: # One category is offset
            bbdb = 'co3'
            if (len(csg) == 0) & (len(ssg) > 0):
                bbdb_t = (f'largely the result of {incdec}{sh.index[-1]}, and partially offset '+
                          f'by {decinc}other categories.')
            elif (len(ssg) == 0) & (len(csg) > 0):
                bbdb_t = (f'the result of {incdec}several categories that is partially offset '+
                          f'by {co_l}.') 
    
    if bbdb in ['bb2', 'bb3']:
        if sct == 'sc':
            su_t = f'. The main contribution, {su_l}, '
            co_t = f'is {co_adj}offset by {co_l}.'
        elif sct == 'su':
            su_t = f'. The main contribution is {su_l}.'
            co_t = ''
    
    if bbdb in ['db1', 'db2', 'db3']:
        co_t = '.'
        if sct == 'sc':
            co_t = f', and {co_adj}offset by {co_l}.'
    text = f'{desc} is {also_t}{bbdb_t}{su_t}{co_t}'
    return(text, bbdb)
