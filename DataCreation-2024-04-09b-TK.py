
################################################################################
###### load packages and data
################################################################################
import pandas as pd
import numpy as np
inputpath1 = f"C:/Users/thkraft/eCommerce-Goethe Dropbox/Thilo Kraft/Thilo(privat)/Privat/Research/NFT/Data/df_quintiles_bins.csv"
df = pd.read_csv(inputpath1,sep=';',encoding='utf-8'
                 #escapechar='//'
                  )

################################################################################
######  Bins and Quintiles
################################################################################


# Convert the 'sow_bin' column to integers
# Convert the 'sow_bin' column to integers
# Convert the 'sow_bin' column to integers

# Function to convert quintile string to integer
def quintile_to_int(quintile_str):
    # Extract the first character and convert to integer
    return int(quintile_str[-1])

# Apply the function to the 'sow_bin' column
df['sow_bin'] = df['sow_bin'].apply(quintile_to_int)



# Add size of wallet bin quintiles	
# Add size of wallet bin quintiles
# Add size of wallet bin quintiles

# Initialize the new column for wallet size bins
df['wallet_size_bin_q'] = pd.NA  # Using pandas' NA for missing data handling

# Define numeric bin labels as per the quintiles
bin_labels = [1, 2, 3, 4, 5]
percentiles_q = [0, 20, 40, 60, 80, 100]

# Iterate over each firm
for firm_name in df['firm_name'].unique():
 # Filter the DataFrame for the current firm and exclude firm_size_wallet == 0
    firm_df = df[(df['firm_name'] == firm_name) & (df['firm_size_wallet'] > 0.0)]
    
    # Calculate the percentile values to use as bin edges for the current firm's 'size_of_wallet'
    bin_edges_q = np.percentile(firm_df['firm_size_wallet'], percentiles_q)

    # Ensure there are enough unique edges to form bins
    if len(bin_edges_q) > 1:
        # Compute quantile bins for 'size_of_wallet' within each firm using pd.cut
        # This method is manually defining bins based on calculated percentile edges
        firm_df['wallet_size_bin_q'] = pd.cut(firm_df['firm_size_wallet'],
                                                bins=bin_edges_q, 
                                                labels=bin_labels, 
                                                include_lowest=True)
    
         # Update the original DataFrame with the binned data
        df.update(firm_df[['wallet_size_bin_q']])

# Add transaction bin quintiles	
# Add transaction bin quintiles
# Add transaction bin quintiles
# Initialize the new column for wallet size bins
df['transaction_bin_q'] = pd.NA  # Using pandas' NA for missing data handling

# Iterate over each firm
for firm_name in df['firm_name'].unique():
 # Filter the DataFrame for the current firm and exclude firm_tx_count == 0
    firm_df = df[(df['firm_name'] == firm_name) & (df['firm_tx_count'] > 0.0)]

    # Ensure there are enough unique edges to form bins
    if len(bin_edges_q) > 1:
        # Compute quantile bins for 'size_of_wallet' within each firm using pd.cut
        # This method is manually defining bins based on calculated percentile edges
        firm_df['transaction_bin_q'] = pd.cut(firm_df['firm_tx_count'],
                                                bins=bin_edges_q, 
                                                labels=bin_labels, 
                                                include_lowest=True)
         # Update the original DataFrame with the binned data
        df.update(firm_df[['transaction_bin_q']])



outputpath1 = f"C:/Users/thkraft/eCommerce-Goethe Dropbox/Thilo Kraft/Thilo(privat)/Privat/Research/NFT/Data/DataCreation-Bins.csv"
df.to_csv(outputpath1,sep=';',encoding='utf-8'
          #escapechar='\\'
           )
