# import yfinance as yf

# # Download earnings data for AAPL
# aapl = yf.Ticker("AAPL")
# # earnings_data = aapl.earnings

# # # Print the data
# # print(earnings_data)
# earnings_dates = aapl.get_earnings_dates()
# print(earnings_dates)

import yahoo_fin.stock_info as si

aapl_data = si.get_earnings_history(ticker="AAPL")

# Iterate through dictionaries
for row in aapl_data["context"]["dispatcher"]["stores"]["ScreenerResultsStore"]["results"]["rows"]:
    date = row["date"]
    eps = row["eps"]
    # Access and process other data points as needed

    # Example: print data
    print(f"Date: {date}, EPS: {eps}")
