#############################################################################
#############################################################################
#Author: Thilo Kraft
#Date: 2024-11-19
#Purpose: This script is used for the paper: Decoding Blockchain Data for Research in Marketing: 
#New Insights Through an Analysis of Share of Wallet
#People of interest can dowload the code and run it 
#############################################################################
#############################################################################
#Description: This script is used to retrieve data from Flipside, a blockchain data provider,

# Import the required libraries
import pandas as pd
from datetime import datetime
import numpy as np

#############################################################################
#############################################################################
#RETRIEVE DATA FROM FLIPSIDE --> NEW: ROW LIMIT REPLACED BY 1 GB SELECT LIMIT
#SELECTED DATA ONLY 150 MB, SINGLE SELECT POSSIBLE, NO LOOPING & MERGING
#############################################################################
#############################################################################
from flipside import Flipside

# Initialize `Flipside` with your API Key and API Url
flipside = Flipside("..........", "https://api-v2.flipsidecrypto.xyz")
#### API KEY ENTFERNT 

sql = """
SELECT
  SELLER_ADDRESS,
  COUNT(DISTINCT TX_HASH) AS unique_tx_count,
  SUM(PLATFORM_FEE_USD) AS total_platform_fee_usd,
  PLATFORM_NAME
FROM
  ethereum.core.ez_nft_sales
WHERE
  block_Timestamp::date >= '2022-01-01'
  AND block_timestamp::date < '2023-01-01'
GROUP BY
  SELLER_ADDRESS, PLATFORM_NAME
"""

result_set = flipside.query(sql)

records = result_set.records

print(pd.DataFrame(records))

current_page_number = 1
page_size = 100000
total_pages = 10

all_rows = []

while current_page_number <= total_pages:
  results = flipside.get_query_results(
    result_set.query_id,
    page_number=current_page_number,
    page_size=page_size
  )

  total_pages = results.page.totalPages
  if results.records:
      all_rows = all_rows + results.records
  
  current_page_number += 1

df = pd.DataFrame(all_rows, columns=['seller_address', 'unique_tx_count','total_platform_fee_usd','platform_name', '__row_index'])

print(df.info)

print(f"All DONE")

#Save .csv file on your computer
#path = 
#df.to_csv(path,sep=';',encoding='utf-8',escapechar='\\' )
