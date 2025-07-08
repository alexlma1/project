import requests
import pandas as pd
import re
from tqdm import tqdm
from datetime import datetime, timedelta
import time 

class Tradier:
    def __init__(self, account_number, auth_token, live_trade=True):
        self.ACCOUNT_NUMBER = account_number
        self.AUTH_TOKEN = auth_token
        self.REQUESTS_HEADERS = {
            'Authorization': f'Bearer {self.AUTH_TOKEN}',
            'Accept': 'application/json'
        }
        self.LIVETRADE_URL = 'https://api.tradier.com'
        self.SANDBOX_URL = 'https://sandbox.tradier.com'
        self.BASE_URL = self.LIVETRADE_URL if live_trade else self.SANDBOX_URL

class Account(Tradier):
    def __init__(self, account_number, auth_token, live_trade=True):
        super().__init__(account_number, auth_token, live_trade)
        base = f"v1/accounts/{account_number}"
        self.PROFILE_ENDPOINT = "v1/user/profile"
        self.BALANCE_ENDPOINT = f"{base}/balances"
        self.GAINLOSS_ENDPOINT = f"{base}/gainloss"
        self.HISTORY_ENDPOINT = f"{base}/history"
        self.POSITIONS_ENDPOINT = f"{base}/positions"
        self.ORDERS_ENDPOINT = f"{base}/orders"

    def get_user_profile(self, sole=False):
        try:
            r = requests.get(f"{self.BASE_URL}/{self.PROFILE_ENDPOINT}", headers=self.REQUESTS_HEADERS)
            r.raise_for_status()
            data = r.json().get('profile', {})
            return pd.json_normalize(data) if not sole else pd.json_normalize(data).iloc[0]
        except Exception as e:
            print(f"Profile error: {e}")
            return pd.DataFrame() if not sole else pd.Series()

    def get_account_balance(self, as_series=False):
        try:
            r = requests.get(f"{self.BASE_URL}/{self.BALANCE_ENDPOINT}", headers=self.REQUESTS_HEADERS)
            r.raise_for_status()
            data = r.json().get('balances', {})
            return pd.Series(data) if as_series else pd.json_normalize(data)
        except Exception as e:
            print(f"Balance error: {e}")
            return pd.Series() if as_series else pd.DataFrame()

    def get_gainloss(self, **kwargs):
        params = {
            'page': kwargs.get('page', 1),
            'limit': kwargs.get('limit', 100),
            'sortBy': kwargs.get('sort_by', 'closeDate'),
            'sort': kwargs.get('sort_direction', 'desc')
        }
        if 'symbol_filter' in kwargs:
            params['symbol'] = kwargs['symbol_filter']
        if 'start_date' in kwargs:
            params['start'] = kwargs['start_date']
        if 'end_date' in kwargs:
            params['end'] = kwargs['end_date']

        try:
            r = requests.get(f"{self.BASE_URL}/{self.GAINLOSS_ENDPOINT}", params=params, headers=self.REQUESTS_HEADERS)
            r.raise_for_status()
            data = r.json().get('gainloss', {}).get('closed_position', [])
            if isinstance(data, dict):
                data = [data]
            return pd.json_normalize(data)
        except Exception as e:
            print(f"Gain/loss error: {e}")
            return pd.DataFrame()

    def get_orders(self):
        try:
            r = requests.get(f"{self.BASE_URL}/{self.ORDERS_ENDPOINT}", params={'includeTags': 'true'}, headers=self.REQUESTS_HEADERS)
            orders = r.json().get('orders', {}).get('order', [])
            return pd.json_normalize(orders)
        except Exception as e:
            print(f"Orders error: {e}")
            return pd.DataFrame()

    def get_positions(self, symbols=None, equities=False, options=False):
        try:
            r = requests.get(f"{self.BASE_URL}/{self.POSITIONS_ENDPOINT}", headers=self.REQUESTS_HEADERS)
            r.raise_for_status()
            data = r.json().get('positions', {}).get('position', [])
            df = pd.DataFrame(data)
            if symbols:
                df = df[df['symbol'].isin(symbols)]
            if equities:
                df = df[df['symbol'].str.len() < 5]
            if options:
                df = df[df['symbol'].str.len() > 5]
            return df
        except Exception as e:
            print(f"Positions error: {e}")
            return pd.DataFrame()

class OptionsData(Tradier):
    def __init__(self, account_number, auth_token, live_trade=True):
        super().__init__(account_number, auth_token, live_trade)
        self.STRIKE_ENDPOINT = "v1/markets/options/strikes"
        self.CHAIN_ENDPOINT = "v1/markets/options/chains"
        self.EXPIRY_ENDPOINT = "v1/markets/options/expirations"
        self.SYMBOL_LOOKUP_ENDPOINT = "v1/markets/options/lookup"

    def get_expiry_dates(self, symbol, strikes=False):
        try:
            r = requests.get(f"{self.BASE_URL}/{self.EXPIRY_ENDPOINT}", params={'symbol': symbol, 'includeAllRoots': True, 'strikes': str(strikes)}, headers=self.REQUESTS_HEADERS)
            r.raise_for_status()
            data = r.json().get('expirations', {})
            return data['expiration'] if strikes else data.get('date', [])
        except Exception as e:
            print(f"Expiry error for {symbol}: {e}")
            return []

    def get_closest_expiry(self, symbol, num_days):
        expiries = self.get_expiry_dates(symbol)
        target = datetime.now() + timedelta(days=num_days)
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in expiries]
        closest = min(dates, key=lambda d: abs(d - target))
        return closest.strftime("%Y-%m-%d")

    def get_chain_day(self, symbol, expiry=None, strike=None, strike_low=None, strike_high=None, option_type=None):
        expiry = expiry or self.get_expiry_dates(symbol)[0]
        try:
            r = requests.get(f"{self.BASE_URL}/{self.CHAIN_ENDPOINT}", params={'symbol': symbol, 'expiration': expiry, 'greeks': 'false'}, headers=self.REQUESTS_HEADERS)
            df = pd.DataFrame(r.json()['options']['option'])
            df = df.loc[:, df.apply(pd.Series.nunique) > 1]  # Drop constant columns
            df.dropna(axis=1, how='all', inplace=True)
            df.drop(columns=['description'], errors='ignore', inplace=True)
            if strike_low: df = df[df['strike'] >= strike_low]
            if strike_high: df = df[df['strike'] <= strike_high]
            if strike: df = df[df['strike'] == strike]
            if option_type in ['call', 'put']: df = df[df['option_type'] == option_type]
            return df
        except Exception as e:
            print(f"Chain error: {e}")
            return pd.DataFrame()

    def get_options_symbols(self, symbol, df=True):
        def parse(symbols):
            parsed = []
            for s in symbols:
                m = re.match(r'([A-Z]+)(\d{6})([CP])(\d+)', s)
                if m:
                    root, date, otype, strike = m.groups()
                    parsed.append({
                        'symbol': s,
                        'root_symbol': root,
                        'expiration_date': datetime.strptime(date, '%y%m%d').strftime('%Y-%m-%d'),
                        'option_type': otype,
                        'strike_price': int(strike) / 1000
                    })
            return pd.DataFrame(parsed)

        try:
            r = requests.get(f"{self.BASE_URL}/{self.SYMBOL_LOOKUP_ENDPOINT}", params={'underlying': symbol}, headers=self.REQUESTS_HEADERS)
            symbols = r.json()['symbols'][0]['options']
            return parse(symbols) if df else symbols
        except Exception as e:
            print(f"Option symbol error for {symbol}: {e}")
            print(r.content)

            return pd.DataFrame() if df else []

    def get_options_symbols_list(self, symbols):
        """
        Retrieve option symbols for a list of underlying symbols with a progress bar.

        Args:
            symbols (list): List of underlying symbols to retrieve options for.

        Returns:
            pd.DataFrame: A combined DataFrame of all option symbols for the given list of symbols.
        """
        options = []
        for symbol in tqdm(symbols, desc="Fetching options symbols"):
            try:
                df = self.get_options_symbols(symbol)
                if not df.empty:
                    options.append(df)
            except Exception as e:
                print(f"Error fetching options for {symbol}: {e}")
                continue
        if options:
            return pd.concat(options, axis=0, ignore_index=True)
        return pd.DataFrame()



class Quotes (Tradier):
    def __init__ (self, account_number, auth_token, live_trade=False):
        Tradier.__init__(self, account_number, auth_token, live_trade);

        #
        # Quotes endpoints for market data about equities
        #

        self.QUOTES_ENDPOINT 				= "v1/markets/quotes"; 											# GET (POST)
        self.QUOTES_HISTORICAL_ENDPOINT 	= "v1/markets/history"; 										# GET
        self.QUOTES_SEARCH_ENDPOINT 		= "v1/markets/search"; 											# GET
        self.QUOTES_TIMESALES_ENDPOINT 		= "v1/markets/timesales"; 										# GET

    def get_quotes(self, symbols):
        """
        Retrieve quotes for a list of symbols.

        Args:
            symbols (list): List of symbols to retrieve quotes for.

        Returns:
            pd.DataFrame: A DataFrame containing the quotes for the given symbols.
        """
        try:
            params = {'symbols': ','.join(symbols)}
            r = requests.get(f"{self.BASE_URL}/{self.QUOTES_ENDPOINT}", params=params, headers=self.REQUESTS_HEADERS)
            r.raise_for_status()
            data = r.json().get('quotes', {}).get('quote', [])
            return pd.json_normalize(data)
        except Exception as e:
            print(f"Quotes error: {e}")
            return pd.DataFrame()
    
    def get_historical_quotes(self, symbol, interval='daily', start_date=None, end_date=None, verbose=False):
        '''
		Fetch historical OHLCV bar data for a given security's symbol from the Tradier Account API.

		Args:
			• symbol (str): The trading symbol (ticker or OCC) of the security (e.g., 'AAPL', 'MSFT', 'NKE241018P00075000').
			• interval (str, optional): The time interval for historical data. Default is 'daily'. One of: daily, weekly, monthly.
			• start_date (str 'YYYY-MM-DD', optional): The start date for historical data. Default contingent on interval argument (see Notes below).
			• end_date (str 'YYYY-MM-DD', optional): The end date for historical data. Default is current date.
			• verbose (bool, optional): Print (possibly) helpful debugging info about request parameters, api response, etc.

		Returns:
			• pandas.DataFrame: A DataFrame containing historical stock data for the specified symbol.

		Notes:
			• Default start_date values conditional upon interval argument:
				• interval='daily'   -> start_date = most recent monday
				• interval='weekly'  -> start_date = 12 weeks before end_date
				• interval='monthly' -> start_date = 1 year before end_date
			• Tradier does not maintain historical data for expired options.

		Example 1: Minimal Arguments.
			# Create a Quotes instance.
			>>> quotes = Quotes(tradier_acct, tradier_token)

			# Retrieve daily OHLCV bar data for M&T Bank for current week.
			# (Run on a Tuesday -> returns row for Monday and Tuesday)
			>>> quotes.get_historical_quotes('MTB')
					date    open    high      low    close  volume
			0 2024-10-14  185.19  186.91  182.905  185.560  980439
			1 2024-10-15  187.08  190.23  185.280  189.905  756967

		Example 2: Weekly OHLCV data without specified dates.
			# Create a Quotes instance.
			>>> quotes = Quotes(tradier_acct, tradier_token)

			# Retrieve weekly data over past 12 weeks data for M&T Bank.
			>>> quotes.get_historical_quotes('MTB', interval='weekly')
					 date    open     high       low    close   volume
			0  2024-07-29  174.88  175.180  159.1400  162.290  4814917
			1  2024-08-05  157.93  164.420  155.1000  162.230  4275015
			2  2024-08-12  163.23  164.880  158.1850  163.090  4210555
			3  2024-08-19  163.09  170.880  161.1450  168.680  3549740
			4  2024-08-26  169.79  172.380  166.0700  170.760  3817783
			5  2024-09-02  172.17  173.285  165.1500  166.510  3855579
			6  2024-09-09  166.66  169.670  161.4000  168.880  4504412
			7  2024-09-16  169.61  180.250  168.9701  179.560  7557681
			8  2024-09-23  179.21  180.635  172.5050  175.420  5170467
			9  2024-09-30  175.48  178.970  170.1000  178.740  4891957
			10 2024-10-07  177.86  185.980  176.6700  185.190  4331667
			11 2024-10-14  185.19  190.230  182.9050  189.305  1789851

		Example 3: Retrieve monthly data for M&T Bank between April 14, 2021 and January 18, 2023.
			# Create a Quotes instance.
			>>> quotes = Quotes(tradier_acct, tradier_token)

			# Retrieve monthly data for M&T Bank.
			>>> quotes.get_historical_quotes('MTB', interval='monthly', start_date='2021-04-14', end_date='2023-01-18')
					 date    open      high      low   close    volume
			0  2021-05-01  159.30  168.2700  154.950  160.69  16054621
			1  2021-06-01  162.19  164.3900  142.550  145.31  18924640
			2  2021-07-01  147.08  147.8601  128.460  133.85  21389600
			3  2021-08-01  134.47  143.2200  131.290  140.01  17082781
			4  2021-09-01  140.31  154.7100  131.420  149.34  20531510
			5  2021-10-01  152.32  162.5400  145.720  147.12  20544524
			6  2021-11-01  149.03  163.2600  145.930  146.61  19397263
			7  2021-12-01  150.04  156.6600  141.490  153.58  18800067
			8  2022-01-01  155.77  186.9300  155.240  169.38  37110333
			9  2022-02-01  169.11  186.9500  168.100  182.23  24752575
			10 2022-03-01  179.84  186.6900  167.020  169.50  36583333
			11 2022-04-01  172.14  184.2900  157.950  166.64  42242399
			12 2022-05-01  168.25  181.1200  159.410  179.97  24896017
			13 2022-06-01  180.51  181.7900  156.190  159.39  23126308
			14 2022-07-01  158.63  178.0750  148.800  177.45  22184583
			15 2022-08-01  176.64  193.4200  173.400  181.78  17840093
			16 2022-09-01  181.76  191.3500  173.540  176.32  22113940
			17 2022-10-01  178.83  192.5600  159.395  168.37  33214286
			18 2022-11-01  169.72  172.5850  162.070  170.02  25592408
			19 2022-12-01  170.68  171.2700  138.430  145.06  31316314
			20 2023-01-01  145.00  158.0000  139.030  156.00  25906356

		Example 4: Retrieve monthly data for M&T Bank between April 14, 2021 and January 18, 2023.
			# Create a Quotes instance.
			>>> quotes = Quotes(tradier_acct, tradier_token)

			# Retrieve historical daily OHLCV bar data for option contract
			# Note - by setting the start_date to a reasonably far back past date, we obtain the prices for the entirety of the contract's life.
			# Note - this example may not work in the future because Tradier does not maintain historical data for expired options.
			>>> quotes.get_historical_quotes('MTB241115C00190000', start_date='2024-06-01')
					 date  open  high   low  close  volume
			0  2024-07-18  3.39  3.39  3.39   3.39       1
			1  2024-07-19  2.85  2.89  2.85   2.89       4
			2  2024-07-22  3.67  3.67  3.67   3.67       2
			3  2024-07-24  3.80  4.40  3.80   4.40      87
			4  2024-07-31  4.70  4.70  4.70   4.70      50
			5  2024-08-05  2.25  2.30  2.25   2.30       4
			6  2024-08-09  1.75  1.75  1.75   1.75       1
			7  2024-08-14  1.18  1.18  1.18   1.18       1
			8  2024-08-21  0.94  0.94  0.94   0.94       1
			9  2024-08-22  1.14  1.14  1.14   1.14       1
			10 2024-08-28  1.75  1.85  1.75   1.85       2
			11 2024-09-03  2.65  2.65  2.65   2.65       2
			12 2024-09-05  2.30  2.30  2.30   2.30      36
			13 2024-09-06  1.75  1.75  1.75   1.75      35
			14 2024-09-11  1.30  1.30  1.30   1.30       2
			15 2024-09-16  2.15  2.30  2.15   2.30      65
			16 2024-09-17  2.75  2.75  2.75   2.75       1
			17 2024-09-18  3.08  3.08  3.08   3.08      10
			18 2024-09-19  3.90  3.90  3.90   3.90      10
			19 2024-09-20  3.45  3.80  3.45   3.80      10
			20 2024-09-23  3.20  3.20  3.20   3.20      16
			21 2024-09-24  3.50  3.50  3.50   3.50      34
			22 2024-09-25  2.22  2.22  2.22   2.22       1
			23 2024-09-30  3.90  3.90  3.90   3.90       1
			24 2024-10-01  2.64  2.67  2.64   2.67       2
			25 2024-10-02  2.25  2.29  2.20   2.20      31
			26 2024-10-03  2.00  2.00  2.00   2.00       3
			27 2024-10-07  3.67  3.67  3.67   3.67       1
			28 2024-10-08  3.40  3.60  3.40   3.60      31
			29 2024-10-09  3.37  3.37  3.37   3.37       1
			30 2024-10-10  3.17  3.17  3.17   3.17       1
			31 2024-10-11  5.82  6.00  5.35   5.84      63
			32 2024-10-14  5.80  5.92  5.62   5.92      26
			33 2024-10-15  6.70  8.10  6.70   8.00      54
		'''

		#
		# Ensure that provided symbol is a string in uppercase format
		#

        if not isinstance(symbol, str):
            print(f"Symbol argument must be a string. Received: ({symbol}, {type(symbol)})")
            return pd.DataFrame()

        symbol = symbol.upper()

		#
		# Check that the interval is legit (daily, weekly, monthly)
		#

        if not isinstance(interval, str):
            print(f"Interval must be string. One of: daily, weekly, monthly")
            return pd.DataFrame()

        interval = interval.lower()
        if interval not in ['daily', 'weekly', 'monthly']:
            print(f"Invalid interval. One of: daily, weekly, monthly")
            return pd.DataFrame()

		#
		# Helper function used to index the start of the trading week
		#

        def last_monday(input_date):
            ''' Hand over last Monday's date '''
            if input_date.weekday() == 0:
                return input_date - timedelta(days=7)
            else:
                return input_date - timedelta(days=input_date.weekday())

		#
		# Infer the appropriate start/end dates (per interval) if not supplied by user
		#   • if end_date not supplied -> end_date set to current date
		#   • if start_date not supplied:
		#       • interval='daily'     -> start_date = most recent monday
		#       • interval='weekly'    -> start_date = 12 weeks ago
		#       • interval='monthly'   -> start_date = last year
		#

        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        if start_date is None:
            if interval == 'monthly':
                tmp_dt = (datetime.strptime(end_date, '%Y-%m-%d')-timedelta(weeks=52)).replace(day=1)
            elif interval == 'weekly':
                tmp_dt = (datetime.strptime(end_date, '%Y-%m-%d')-timedelta(weeks=12))
            else:
                tmp_dt = last_monday(datetime.strptime(end_date, '%Y-%m-%d'))

            start_date = tmp_dt.strftime('%Y-%m-%d')

        try:
            if verbose:
                print("Sending HTTP GET Request...")
                print(f"HTTP Request Params: (symbol={symbol}, interval={interval}, start={start_date}, end={end_date})\n")

			#
			# HTTP GET Request
			#

            r = requests.get(
            url     = f"{self.BASE_URL}/{self.QUOTES_HISTORICAL_ENDPOINT}",
            params  = {'symbol':symbol, 'interval':interval, 'start':start_date, 'end':end_date},
            headers = self.REQUESTS_HEADERS
            )
            r.raise_for_status()

            data = r.json()

            if verbose:
                print(pd.DataFrame(data['history']))
                print(f"DATA:\n{data}\n")

			#
			# Validate the structure of the API Response from Tradier
			#

            if not data:
                raise ValueError(f"Empty API Response. Status Code: {r.status_code}")
            if 'history' not in data:
                raise KeyError(f"API response missing 'history'. Received: {data}")
            if data['history'] is None:
                return pd.DataFrame()  # No historical data available
                #raise ValueError(f"No historical data for (symbol={symbol}, start_date={start_date}, end_date={end_date})")
            if 'day' not in data['history']:
                raise KeyError(f"API Response history data missing 'day': {data}")

			#
			# Extract the sought information from the API response and prepare returned dataframe
			#

            df = pd.json_normalize(data['history']['day'])
            df['date'] = pd.to_datetime(df['date'])

            if verbose:
                print(f"Returning Structure:\n{df.info()}\n")

			#
			# Off you go!
			#

            return df

        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            raise RuntimeError(f"[Historical Quotes] ... {e}")
