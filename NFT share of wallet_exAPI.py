#############################################################################
#############################################################################
#RETRIEVE DATA FROM FLIPSIDE --> NEW: ROW LIMIT REPLACED BY 1 GB SELECT LIMIT
#SELECTED DATA ONLY 150 MB, SINGLE SELECT POSSIBLE, NO LOOPING & MERGING
#############################################################################
#############################################################################

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
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
  ethereum.core.ez_nft_salesP
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

path = f"C:/Users/Hanneke/Documents/Research Proposals/NFT Share of Wallet/Data\Seller_2022-01-01-2022-12-31.csv"
df.to_csv(path,sep=';',encoding='utf-8',escapechar='\\' )



#############################################################################
#############################################################################
#############################################################################
# AUSWERTUNGEN 
#############################################################################
#############################################################################
#############################################################################


import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

path1 = f"C:/Users/Hanneke/Documents/Research Proposals/NFT Share of Wallet/Data\Seller_2022-01-01-2022-12-31.csv"
df = pd.read_csv(path1,sep=';',encoding='utf-8',escapechar='\\' )

print(df.info())
print(df.describe())
print(df['total_platform_fee_usd'].sum())
  #--> 1.025.131.364,9055185 same as query result on top level

print(df.shape)
  #--> 1.903.440 rows, same as query result on top level

print(df['seller_address'].nunique())
  #--> 1.327.468 unique seller

print(df['unique_tx_count'].sum())
  #--> 22.886.156 transactions

### total spending per seller: SUM(total_platform_fee_usd) per unique seller
total_spending_per_seller = df.groupby('seller_address')['total_platform_fee_usd'].sum()
print(total_spending_per_seller)
  #--> Length: 1.327.468 --> quality check ok!
print(total_spending_per_seller.sum())
  #--> 1.025.131.364,9055185 same as query result on top level

### total transactions per seller:SUM(unique_tx_count) per unique seller 
total_transactions_per_seller = df.groupby('seller_address')['unique_tx_count'].sum()
## has to be recalculated later, because transactions on larva labs will be filtered out

print(total_transactions_per_seller)
  #--> Length: 1.327.468 --> quality check ok!
print(total_transactions_per_seller.sum())
  #--> 22.886.156 transactions

####Filter data out: Exlude Larva Labs and if total_spending_per_seller 0)
total_spending_per_seller = total_spending_per_seller.rename('total_spending_per_seller')
df = pd.merge(df, total_spending_per_seller, on=['seller_address'])

# Create a boolean mask for the conditions
mask = (df['platform_name'] != 'larva labs') & (df['total_spending_per_seller'] > 0)

# Apply the mask to the DataFrame
df = df[mask]


###################################################
# Quality SUM check 
###################################################
print(df['seller_address'].nunique())
  #--> 1.327.468 -> Filtered: 1.286.893 unique seller

print(df['unique_tx_count'].sum())
  #--> 22.886.156 transactions --> Filtered: 22.788.707

print(df['total_platform_fee_usd'].sum())
  #--> 1.025.131.364 --> same as before - good, because we only filtered 0-Fee! 
 
print(df.shape)
  #--> filtered: 1.860.649
 
print(df.info())
################################################################################
### SoW(fee) per seller on platform: total_platform_fee_usd per seller on platform / SUM(total_platform_fee_usd) per unique seller
################################################################################

# Group the data by seller_address and platform_name and calculate the total platform fees per seller on each platform
platform_fees_per_seller_per_platform = df.groupby(['seller_address', 'platform_name'])['total_platform_fee_usd'].sum()

# Compute the share-of-wallet per seller on each platform
sow_fee_per_seller_per_platform = platform_fees_per_seller_per_platform / total_spending_per_seller

print(sow_fee_per_seller_per_platform)

# Compute the number of platforms a seller_address was active on
seller_platform_count = df.groupby('seller_address')['platform_name'].nunique()

# Print the results
print(seller_platform_count)


################################################################################
### SoW(trx) per seller on platform: unique_tx_count per seller on platform / SUM(unique_tx_count) per unique seller
################################################################################

# Group the data by seller_address and platform_name and calculate the total transactions per seller on each platform
transactions_per_seller_per_platform = df.groupby(['seller_address', 'platform_name'])['unique_tx_count'].sum()

# Calculate the total transactions per unique seller
total_transactions_per_seller = df.groupby('seller_address')['unique_tx_count'].sum()

# Compute the share-of-wallet per seller on each platform for transactions
sow_transactions_per_seller_per_platform = transactions_per_seller_per_platform / total_transactions_per_seller

# Print the result
print(sow_transactions_per_seller_per_platform)

################################################################################
### CREATE potential wallet per seller and platform
################################################################################

print(df.info())

# Calculate the potential wallet
df['potential_wallet'] = df['total_spending_per_seller'] - df['total_platform_fee_usd']


################################################################################
### MERGING & SUMMARY STATISTICS
################################################################################

# Rename the columns to avoid duplicates before merging
sow_fee_per_seller_per_platform = sow_fee_per_seller_per_platform.rename('sow_fee_per_seller_per_platform')
sow_transactions_per_seller_per_platform = sow_transactions_per_seller_per_platform.rename('sow_transactions_per_seller_per_platform')
seller_platform_count = seller_platform_count.rename('seller_platform_count')
total_transactions_per_seller = total_transactions_per_seller.rename('total_transactions_per_seller')

# Merge the share-of-wallet columns with the original data frame
df_merged = pd.merge(df, sow_fee_per_seller_per_platform, on=['seller_address', 'platform_name'])
df_merged = pd.merge(df_merged, sow_transactions_per_seller_per_platform, on=['seller_address', 'platform_name'])
df_merged = pd.merge(df_merged, seller_platform_count, on=['seller_address'])
df_merged = pd.merge(df_merged, total_transactions_per_seller, on=['seller_address'])

df_merged.drop(columns=["Unnamed: 0", "__row_index"], inplace=True)
print(df_merged.info())

path2 = f"C:/Users/Hanneke/Documents/Research Proposals/NFT Share of Wallet/Data\dfmerged_2022-01-01-2022-12-31.csv"
df_merged.to_csv(path2,sep=';',encoding='utf-8',escapechar='\\' )

###################################################
# Quality SUM check 
###################################################

print(df_merged['sow_fee_per_seller_per_platform'].sum())
# --> 1286893.0--> good
print(df_merged['seller_address'].nunique())
# --> 1286893--> good
print(df_merged['total_platform_fee_usd'].sum())
# --> 1025131364.905517 --> good
print(df_merged['sow_transactions_per_seller_per_platform'].sum())
# --> 1286893.0 --> good

################################################################################
###### CREATE Total Wallet Quintiles
################################################################################

# Calculate the sum of total_platform_fee_usd for each unique seller_address
seller_fee_sum = df_merged.groupby('seller_address')['total_platform_fee_usd'].sum()

# Compute the quintiles based on the fee sums
quintiles = pd.qcut(seller_fee_sum, q=5, labels=['1st_quintile', '2nd_quintile', '3rd_quintile', '4th_quintile', '5th_quintile'], duplicates='drop')

# Map the quintiles to the seller_address
df_merged['total_quintile'] = df_merged['seller_address'].map(quintiles)

# Count the NaN values in the "quintile" column
nan_count = df_merged['total_quintile'].isna().sum()

# Print the count
print("Number of NaN in quintile column:", nan_count)

# Group by seller_address and calculate the required information
grouped_sellers = df_merged.groupby('seller_address').agg({
    'total_platform_fee_usd': 'sum',
    'unique_tx_count': 'sum',
    'total_quintile': 'max'
}).reset_index()

print(grouped_sellers.head())
print(grouped_sellers.columns)

# Group by quintiles and calculate the required information
grouped_quintiles = grouped_sellers.groupby('total_quintile').agg({
    'seller_address': 'nunique',
    'total_platform_fee_usd': ['sum', 'mean'],
    'unique_tx_count': 'sum'
})

# Calculate the overall sums
overall_sums = grouped_quintiles.sum()

# Print the grouped information
print("Grouped by Quintiles:")
print(grouped_quintiles)

# Print the overall sums
print("\nOverall Sums:")
print(overall_sums)

print(df_merged.info())

# # Drop the "Unnamed: 0.1" and "Unnamed: 0" columns
# df_merged.drop(columns=["Unnamed: 0"], inplace=True)

# # Reset the index of the DataFrame
# df_merged.reset_index(drop=True, inplace=True)

# print(df_merged.info())
# print(df_merged.head())



path3 = f"C:/Users/Hanneke/Documents/Research Proposals/NFT Share of Wallet/Data\dfmerged_quintiles.csv"
df_merged.to_csv(path3,sep=';',encoding='utf-8',escapechar='\\' )

################################################################################
###### CREATE SoW BINS per Platform
################################################################################

#path3 = f"C:/Users/Hanneke/Documents/Research Proposals/NFT Share of Wallet/Data\dfmerged_quintiles.csv"
#df = pd.read_csv(path3,sep=';',encoding='utf-8',escapechar='\\' )
# df.drop(columns=["Unnamed: 0"], inplace=True)
# df.reset_index(drop=True, inplace=True)

df = df_merged

print(df.info())
print(df.head())


df['platform_bin'] = ""

# Iterate over each platform
for platform in df['platform_name'].unique():
    # Filter the DataFrame for the current platform
    platform_df = df[df['platform_name'] == platform]
    
    # Compute the bin for sow_per_seller_per_platform using pd.cut
    bin_labels = [f"Bin {i+1}" for i in range(5)]
    platform_df['platform_bin'] = pd.cut(platform_df['sow_fee_per_seller_per_platform'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.], labels=bin_labels, include_lowest=True)
    
    # Update the corresponding rows in the original DataFrame
    df.update(platform_df['platform_bin'])

# Print the updated DataFrame
print(df)

print(df.info())

# Rename the columns
df = df.rename(columns={
    'seller_address': 'customer_id',
    'unique_tx_count': 'firm_tx_count',
    'platform_name':'firm_name',
    'total_platform_fee_usd': 'firm_size_wallet',
    'potential_wallet':'firm_potential_wallet',
    'total_spending_per_seller': 'total_wallet',
    'sow_fee_per_seller_per_platform' : 'firm_sow_fee',
    'sow_transactions_per_seller_per_platform': 'firm_sow_tx',
    'total_transactions_per_seller': 'total_tx_count',     
    'seller_platform_count': 'customer_firms_count',   
    'platform_bin': 'sow_bin',
    'total_quintile':'total_wallet_quintile'
})


path4 = f"C:/Users/Hanneke/Documents/Research Proposals/NFT Share of Wallet/Data\df_quintiles_bins.csv"
df.to_csv(path4,sep=';',encoding='utf-8',escapechar='\\' )



################################################################################
###### END DATA PREPARATION
################################################################################

################################################################################
###### START DATA OUTPUT
################################################################################



################################################################################
###### OUTPUT SUMMARY STATISTICS
################################################################################

path4 = f"C:/Users/Hanneke/Documents/Research Proposals/NFT Share of Wallet/Data\df_quintiles_bins.csv"
df = pd.read_csv(path4,sep=';',encoding='utf-8',escapechar='\\' )
df.drop(columns=["Unnamed: 0"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.info())
print(df.head())
print(df.columns)

customer_summary_stats = df.groupby('total_wallet_quintile').agg(
    # Number of customer_ID
    customer_unique=('customer_id', 'nunique'),
    customer_count=('customer_id', 'count'),

    # Sum and Mean firm_size_wallet
    firm_size_wallet_sum=('firm_size_wallet', 'sum'),
    firm_size_wallet_mean=('firm_size_wallet', 'mean'),
    # Sum and Mean firm_tx_count
    firm_tx_count_sum=('firm_tx_count', 'sum'),
    firm_tx_count_mean=('firm_tx_count', 'mean')
)

totals = customer_summary_stats.sum(numeric_only=True)
customer_summary_stats = pd.concat([customer_summary_stats, totals.rename('Total')])


# Print the summary statistics table
print(customer_summary_stats)

# Export the summary statistics table to Excel
customer_summary_stats.to_excel('customer_summary_stats.xlsx', index=True)



firms_summary_stats = df.groupby('firm_name').agg({
    'customer_id': 'nunique',
    'firm_size_wallet': ['mean', 'sum'],
    'customer_firms_count': 'mean',
    'firm_tx_count': ['sum', 'mean']
})

# Rename the columns for better readability
firms_summary_stats.columns = ['Unique Customers', 'Mean Fees', 'Sum Fees', 'Mean Firms Count', 'Sum Transactions', 'Mean Transactions']

totals = firms_summary_stats.sum(numeric_only=True)
firms_summary_stats = pd.concat([firms_summary_stats, totals.rename('Total')])


# Print the summary statistics table
print(firms_summary_stats)

# Export the summary statistics table to Excel
firms_summary_stats.to_excel('firms_summary_stats.xlsx', index=True)


################################################################################
###### OUTPUT MARKET VIEW - DU ET AL 2007 FIGURE 1 REPLICATION PER PLATFORM
################################################################################

path4 = f"C:/Users/Hanneke/Documents/Research Proposals/NFT Share of Wallet/Data\df_plat_quintiles_2022-01-01-2022-12-31.csv"
df = pd.read_csv(path4,sep=';',encoding='utf-8',escapechar='\\' )
df.drop(columns=["Unnamed: 0"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.info())
print(df.head())
print(df.columns)

# Create an Excel writer object
writer = pd.ExcelWriter('Firms_TotalWallet_SoW.xlsx')

# Filter the DataFrame for each platform and create pivot tables
for platform in df['firm_name'].unique():
    platform_df = df[df['firm_name'] == platform]
    
    # Create pivot table for 'customer_id'
    pivot_table_customer_id = pd.pivot_table(platform_df, index='total_wallet_quintile', columns='sow_bin', values='customer_id', aggfunc='nunique', fill_value=0, margins=True)
    
    # Create pivot table for 'firm_size_wallet'
    pivot_table_firm_size_wallet = pd.pivot_table(platform_df, index='total_wallet_quintile', columns='sow_bin', values='firm_size_wallet', aggfunc='sum', fill_value=0, margins=True)
    
    # Create pivot table for 'firm_tx_count'
    pivot_table_firm_tx_count = pd.pivot_table(platform_df, index='total_wallet_quintile', columns='sow_bin', values='firm_tx_count', aggfunc='sum', fill_value=0, margins=True)
    
    # Add custom headings to separate the pivot tables
    heading_customer_id = pd.DataFrame({('CUSTOMER ID', ''): ['']})
    heading_firm_size_wallet = pd.DataFrame({('FIRM SIZE WALLET', ''): ['']})
    heading_firm_tx_count = pd.DataFrame({('FIRM TX COUNT', ''): ['']})
    
    # Concatenate the headings with the respective pivot tables horizontally (along axis=1)
    combined_df = pd.concat([heading_customer_id, pivot_table_customer_id, heading_firm_size_wallet, pivot_table_firm_size_wallet, heading_firm_tx_count, pivot_table_firm_tx_count], axis=1)
    
    # Calculate the percentage of total sum for each pivot table
    pivot_table_customer_id_percent = pivot_table_customer_id / pivot_table_customer_id.loc['All', 'All']
    pivot_table_firm_size_wallet_percent = pivot_table_firm_size_wallet / pivot_table_firm_size_wallet.loc['All', 'All']
    pivot_table_firm_tx_count_percent = pivot_table_firm_tx_count / pivot_table_firm_tx_count.loc['All', 'All']
    
    # Add custom headings for percentage pivots
    heading_customer_id_percent = pd.DataFrame({('CUSTOMER ID (%)', ''): ['']})
    heading_firm_size_wallet_percent = pd.DataFrame({('FIRM SIZE WALLET (%)', ''): ['']})
    heading_firm_tx_count_percent = pd.DataFrame({('FIRM TX COUNT (%)', ''): ['']})
    
    # Concatenate the percentage pivots with the corresponding headings horizontally (along axis=1)
    combined_percent_df = pd.concat([heading_customer_id_percent, pivot_table_customer_id_percent,
                                     heading_firm_size_wallet_percent, pivot_table_firm_size_wallet_percent,
                                     heading_firm_tx_count_percent, pivot_table_firm_tx_count_percent], axis=1)
    
    # Write the combined DataFrame to the Excel file with platform name as the sheet name
    combined_df.to_excel(writer, sheet_name=f'{platform}_Combined')
    combined_percent_df.to_excel(writer, sheet_name=f'{platform}_Combined', startcol=len(combined_df.columns) + 2)

# Save the Excel file
writer.save()


################################################################################
###### OUTPUT CORRELATIONS
################################################################################

path4 = f"C:/Users/Hanneke/Documents/Research Proposals/NFT Share of Wallet/Data\df_plat_quintiles_2022-01-01-2022-12-31.csv"
df = pd.read_csv(path4,sep=';',encoding='utf-8',escapechar='\\' )
df.drop(columns=["Unnamed: 0"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.info())
print(df.head())
print(df.columns)


def compute_correlations(df):
    cols = ['firm_size_wallet', 'total_wallet', 'firm_potential_wallet', 'firm_sow_fee', 'firm_tx_count', 'total_tx_count' , 'firm_sow_tx',]
    return df[cols].corr()

# Calculate correlations for each firm_name separately using pivot table
firm_correlations = df.groupby('firm_name').apply(compute_correlations)

# Calculate correlations for the entire sample
sample_correlations = compute_correlations(df)

# Step 3: Write the correlations to separate sheets in an Excel file
with pd.ExcelWriter('Correlations.xlsx') as writer:
    # Write the firm correlations to the first sheet
    firm_correlations.to_excel(writer, sheet_name='Firm Correlations')
    
    # Write the sample correlations to the second sheet
    sample_correlations.to_excel(writer, sheet_name='Sample Correlations')


######################## manual double check ########################

# Create an empty DataFrame to store the correlation results

sample_correlations_manual = df['firm_tx_count'].corr(df['firm_size_wallet'])
print(sample_correlations_manual)

df_platform = df[df['firm_name'] == 'blur']
sample_correlations_manual2 = df_platform['firm_tx_count'].corr(df_platform['firm_size_wallet'])
print(sample_correlations_manual2)


################################################################################
###### OUTPUT REGRESSIONS
################################################################################

## create one output excel with regression results per platform on every sheet

path4 = f"C:/Users/Hanneke/Documents/Research Proposals/NFT Share of Wallet/Data\df_plat_quintiles_2022-01-01-2022-12-31.csv"
df = pd.read_csv(path4,sep=';',encoding='utf-8',escapechar='\\' )
df.drop(columns=["Unnamed: 0"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.info())
print(df.head())
print(df.columns)

import pandas as pd
import statsmodels.api as sm
from openpyxl.utils.dataframe import dataframe_to_rows
from statsmodels.iolib.summary2 import summary_col


def perform_regression(data):
    # Dependent variable
    y = data['firm_potential_wallet']
    
    # Independent variables
    X = sm.add_constant(data[['firm_size_wallet', 'firm_tx_count']])
    
    # Interaction term: firm_size_wallet * firm_tx_count
    X['interaction'] = data['firm_size_wallet'] * data['firm_tx_count']
    
    # Perform the multiple linear regression
    model = sm.OLS(y, X).fit(cov_type='HC3')
    
    return model

# Step 1: Create a list to store the summary information for each firm
summary_list = []

# Step 2: Loop through each firm and perform the regression
for firm in df['firm_name'].unique():
    firm_data = df[df['firm_name'] == firm]
    model = perform_regression(firm_data)
    
    # Extract the necessary information from the model
    firm_name = firm
    r_squared = model.rsquared
    coefficients = model.params
    p_values = model.pvalues
    print(firm)
    print(model.summary())
    # Append the summary information to the list
    summary_list.append([firm_name, r_squared] + coefficients.tolist() + p_values.tolist())

# Step 3: Perform the regression for the total sample (without filtering for firm_name)
total_sample_data = df
total_model = perform_regression(total_sample_data)
print(f'Total Sample')
print(total_model.summary())

# Extract the necessary information for the total sample
total_r_squared = total_model.rsquared
total_coefficients = total_model.params
total_p_values = total_model.pvalues

# Append the total sample summary information to the list with label "Total Sample"
summary_list.append(["Total Sample", total_r_squared] + total_coefficients.tolist() + total_p_values.tolist())

# Step 4: Create a DataFrame from the summary list with more specific column names
columns = ['Firm Name', 'R^2'] + ['Coefficient (Constant)', 'Coefficient (firm_size_wallet)', 'Coefficient (firm_tx_count)', 'Coefficient (Interaction)'] + ['P-Value (Constant)', 'P-Value (firm_size_wallet)', 'P-Value (firm_tx_count)', 'P-Value (Interaction)']
summary_df = pd.DataFrame(summary_list, columns=columns)

# Step 5: Export the summary table to Excel
summary_df.to_excel('regression_summary.xlsx', index=False)
