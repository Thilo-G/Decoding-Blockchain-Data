#############################################################################
#############################################################################
#Author: Thilo Kraft
#Date: 2024-11-19
#Purpose: This script is used for the paper: Decoding Blockchain Data for Research in Marketing: 
#New Insights Through an Analysis of Share of Wallet
#People of interest can dowload the code and run it on a sample data or data of their own
#############################################################################
#############################################################################


# Import the required libraries
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


#Upload the data merged from Flipside Crypto#
#Jump to Data Output with the sample data to replicate results
#upath1 = f""
#df = pd.read_csv(upath1,sep=';',encoding='utf-8',escapechar='\\' )

print(df["seller_address"].nunique())

#exclude looksrare
exlooksrare = True
firm_name = "looksrare"
if exlooksrare:
    # Step 1: Filter the DataFrame to isolate entries for the firm "looksrare"
    looksrare_df = df[df["platform_name"] == "looksrare"]

    # Step 2: Calculate the 99.9th percentile (top decile) of 'firm_size_wallet' for "looksrare"
    # and identify the customer IDs to be excluded
    top_decile_threshold = looksrare_df["total_platform_fee_usd"].quantile(0.999)
    top_decile_customers = looksrare_df[looksrare_df["total_platform_fee_usd"] > top_decile_threshold]["seller_address"]
    print(len(top_decile_customers))
    # Step 3: Remove these customer IDs from the original DataFrame
    df = df[~df["seller_address"].isin(top_decile_customers)]
    print(df[df["platform_name"] == "looksrare"]["total_platform_fee_usd"].sum())

print(df["seller_address"].nunique() + len(top_decile_customers))
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

#Save .csv file on your computer
dpath1 = f"C:/Users/thkraft/eCommerce-Goethe Dropbox/Thilo Kraft/Thilo(privat)/Privat/Research/NFT/GitHubCode/df_complete-2024-11-19a-TK.csv"
df.to_csv(dpath1,sep=';',encoding='utf-8',escapechar='\\' )


################################################################################
###### END DATA PREPARATION
################################################################################

################################################################################
###### START DATA OUTPUT
################################################################################



################################################################################
###### OUTPUT SUMMARY STATISTICS
################################################################################
#Upload the merged and updated data or sample file or continue with the data from above
#upath2 = f""
#df = pd.read_csv(upath2,sep=';',encoding='utf-8',escapechar='\\' )

df.drop(columns=["Unnamed: 0"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.info())
print(df.head())
print(df.columns)

###################################################
# Table 5: Distribution of Customers and Their Size of Wallets across Firms
###################################################
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

# Path to save the Excel file
dpath2 = f"C:/Users/thkraft/eCommerce-Goethe Dropbox/Thilo Kraft/Thilo(privat)/Privat/Research/NFT/GitHubCode/firm_summary_stats_exlooks-2024-11-19a-TK.xlsx"
# Create a Pandas Excel writer using openpyxl as the engine
with pd.ExcelWriter(dpath22, engine='openpyxl') as writer:
  firms_summary_stats.to_excel(writer, index=True)


################################################################################
###### Table 7 Correlations between unobservable metrics
################################################################################
# Define the columns of interest for correlation
columns_of_interest = [
    'firm_size_wallet', 'total_wallet',
    'firm_potential_wallet', 'firm_sow_fee', 'firm_sow_tx',
    'firm_tx_count'
]

# Path to save the Excel file
dpath3 = f"C:/Users/thkraft/eCommerce-Goethe Dropbox/Thilo Kraft/Thilo(privat)/Privat/Research/NFT/Second_Round/GithubCode/firm_correlation_matrices_exlooks-2024-11-19a-TK.xlsx"
# Create a Pandas Excel writer using openpyxl as the engine
with pd.ExcelWriter(dpath3, engine='openpyxl') as writer:
    correlation_matrix = df[columns_of_interest].corr()
    correlation_matrix.to_excel(writer, sheet_name="Correlation_all")
    # Iterate through each firm
    for firm_name, group_df in df.groupby('firm_name'):
        # Calculate the correlation matrix for the current group
        correlation_matrix = group_df[columns_of_interest].corr(method='spearman')
        
        # Write the correlation matrix to a sheet named after the firm
        sheet_name = firm_name[:31]  # Excel sheet names are limited to 31 characters
        correlation_matrix.to_excel(writer, sheet_name=sheet_name)

################################################################################
###### Figure 4 Distribution of Share of Wallet Across all Customers
################################################################################
###### Figure 4a Distribution of Share of Wallet Across all Customers
# Create the histogram for 'firm_sow_fee'
plt.figure(figsize=(10, 6))
plt.hist(df['firm_sow_fee'], bins=30, edgecolor='black', alpha=0.7)

# Add title and labels
plt.title('Distribution of Share of Wallet Across all Customers', fontsize=14)
plt.xlabel('Share of Wallet', fontsize=12)
plt.ylabel('Percentage of Customers', fontsize=12)

# Format the y-axis as percentages with a maximum of 60% and steps of 20%
total_count = len(df)  # Total number of customers
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / total_count * 100:.0f}%'))
plt.gca().yaxis.set_ticks([i * total_count / 5 for i in range(4)])  # 0%, 20%, 40%, 60%

# Set the y-axis limit to 60%
plt.ylim(0, total_count * 0.6)

# Show the plot
plt.tight_layout()
plt.show()

###### Figure 4b Distribution of Share of Wallet Across all Customers, who trade at more than one firm
# Filter the dataset for customers trading at more than one firm
df_filtered = df[df['customer_firms_count'] > 1]

# Create the histogram for 'firm_sow_fee' across filtered customers
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['firm_sow_fee'], bins=30, edgecolor='black', alpha=0.7)

# Add title and labels
plt.title('Distribution of Share of Wallet among Customers who trade at more than one firm', fontsize=14)
plt.xlabel('Share of Wallet', fontsize=12)
plt.ylabel('Percentage of Customers', fontsize=12)

# Format the y-axis as percentages with steps of 20% until 40%
total_count_filtered = len(df_filtered)  # Total number of filtered customers
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / total_count_filtered * 100:.0f}%'))
plt.gca().yaxis.set_ticks([i * total_count_filtered / 5 for i in range(3)])  # 0%, 20%, 40%

# Set the y-axis limit to 40%
plt.ylim(0, total_count_filtered * 0.4)

# Show the plot
plt.tight_layout()
plt.show()