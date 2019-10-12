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
        obs = [i['DataValue'] for i in nipa_table
               if i['SeriesCode'] == code]
        index = [pd.to_datetime(i['TimePeriod']) for i in nipa_table 
                 if i['SeriesCode'] == code]
        data[code] = (pd.Series(data=obs, index=index).sort_index()
                        .str.replace(',', '').astype(float))
        
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

    
def write_txt(filename, filetext):
    ''' Write label to txt file '''
    with open(filename, 'w') as text_file:
        text_file.write(filetext)


def cont_subt(value, style='main'):
    '''
    Return text for value
    
    -------
    
    Styles:
    
    main: "contributed x.xx percentage points to"
    
    of: "contribution of x.xx percentage points"
    
    end: "contributed x.xx percentage points"
    
    '''
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
    '''Returb info about a pandas series'''
    d = {}
    d['date_max'] = s.idxmax()
    d['date_min'] = s.idxmin()
    d['val_max'] = s.max()
    d['val_min'] = s.min()
    d['date_latest'] = s.index[-1]
    d['date_prev'] = s.index[-2]
    d['date_year_ago'] = s.index[-13]
    d['val_latest'] = s.iloc[-1]
    d['val_prev'] = s.iloc[-2]
    d['val_year_ago'] = s.iloc[-13]
    if d['date_latest'] > d['date_prev']:
        dlm = s[s >= d['val_latest']].sort_index().index[-2]
        d['last_matched'] = f'the highest level since {dlm.strftime("%B %Y")}'
        d['days_since_match'] = (d['date_latest'] - dlm).days
    elif d['date_latest'] < d['date_prev']:
        dlm = s[s <= d['val_latest']].sort_index().index[-2]
        d['last_matched'] = f'the lowest level since {dlm.strftime("%B %Y")}'
        d['days_since_match'] = (d['date_latest'] - dlm).days
    else:
        d['last matched'] = 'the same level as the previous month'
        d['days_since_match'] = 0
    d['late90s'] = s.loc['1998': '1999'].mean()
    d['last_3m'] = s.iloc[-3:].mean()
    d['prev_3m'] = s.iloc[-6:-3].mean()
    d['last_12m'] = s.iloc[-12:].mean()
    d['prev_12m'] = s.iloc[-24:-12].mean()
    d['change_1m'] = d['val_latest'] - d['val_prev']
    d['change_12m'] = d['val_latest'] - d['val_year_ago']
    
    return d


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
