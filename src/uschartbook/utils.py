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
    table_id = (table_id,)
    conn = sqlite3.connect('../data/chartbook.db')
    c = conn.cursor()
    c.execute('''SELECT data FROM bea_nipa_raw WHERE id=? AND 
                 date=(SELECT MAX(date) FROM bea_nipa_raw)''', table_id)
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
        data[code] = (pd.Series(data=obs, index=index)
                        .str.replace(',', '').astype(float))
        
    return pd.DataFrame(data)
    
    
def nipa_series_codes(nipa_table):
    '''
    
    Return series codes and names from table code, e.g. 'T20100'
    
    '''
    r = nipa_table['Data']

    series_dict = {item['SeriesCode']: item['LineDescription'] for item in r}
    
    return series_dict
    
    
def growth_rate(series):
	''' Return the annualized quarterly growth rate in percent'''
	return ((((series.pct_change() + 1) ** 4) - 1) * 100)
