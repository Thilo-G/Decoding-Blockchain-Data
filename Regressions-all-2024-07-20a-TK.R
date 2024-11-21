rm(list = ls())
# Avoid scientific notation
options(scipen = 999)

# Load required libraries
library(dplyr)
library(ggplot2)
library(readr)
library(scales)
library(plm)
library(stargazer)
library(caret)
library(lmtest)

# Input path
inputpath1 <- "C:/Users/thkraft/eCommerce-Goethe Dropbox/Thilo Kraft/Thilo(privat)/Privat/Research/NFT/Second_Round/Exlooksrare/df_quintiles_bins-2024-04-24a-TK.csv"
# Read the CSV file with the correct delimiter
df <- read_delim(inputpath1, delim = ";")

# Check the first few rows to ensure it is read correctly
print(head(df))

#################################################################################
# Data preparation
#################################################################################

# Market Share
total_firm_wallets <- df %>%
  group_by(firm_name) %>%
  summarise(total_firm_size_wallet = sum(firm_size_wallet))

total_market_wallet <- sum(df$firm_size_wallet)

market_share <- total_firm_wallets %>%
  mutate(market_share = total_firm_size_wallet / total_market_wallet)

df <- df %>%
  left_join(market_share %>% select(firm_name, market_share), by = "firm_name")

# Ensure market_share is numeric
df <- df %>%
  mutate(market_share = as.numeric(market_share))

# Create new percentage columns
df <- df %>%
  mutate(
    firm_sow_fee_perc = firm_sow_fee * 100,
    market_share_perc = market_share * 100
  )

# Create a label encoder for 'firm_name'
df <- df %>%
  mutate(all_firm_name_encoded = as.numeric(factor(firm_name)))

# Ensure customer_id and market_share are present in df_panel_with_dummies
df_panel_with_dummies <- df %>%
  select(customer_id, all_firm_name_encoded, firm_sow_fee, firm_size_wallet, firm_potential_wallet, total_wallet, firm_name, market_share, firm_sow_fee_perc, market_share_perc) %>%
  mutate(firm_name = factor(firm_name))

# Create the formula with interaction terms
interaction_formula <- as.formula("~ firm_name * firm_size_wallet + total_wallet + firm_sow_fee + firm_potential_wallet + market_share + firm_sow_fee_perc + market_share_perc - 1")

# Use model.matrix to create dummy variables and interaction terms
model_matrix <- model.matrix(interaction_formula, data = df_panel_with_dummies)

# Combine the model matrix with the customer_id and all_firm_name_encoded
df_panel_with_dummies <- cbind(df_panel_with_dummies[, c("customer_id", "all_firm_name_encoded")], model_matrix)

# Convert to panel data
df_panel <- pdata.frame(df_panel_with_dummies, index = c("customer_id", "all_firm_name_encoded"))

# View the structure of the panel data frame to confirm
str(df_panel)

#################################################################################
# Potential Wallet
#################################################################################

pow_m_1 <- plm(firm_potential_wallet ~ 1+firm_size_wallet, data = df_panel, model = "pooling")
pow_m_2 <-  plm(firm_potential_wallet ~ 1+ firm_nameblur:firm_size_wallet + firm_namegem:firm_size_wallet + firm_namelooksrare:firm_size_wallet + firm_namenftx:firm_size_wallet + firm_nameopensea:firm_size_wallet + firm_namerarible:firm_size_wallet + firm_namesudoswap:firm_size_wallet + firm_namex2y2:firm_size_wallet+ firm_namegem+firm_namelooksrare+firm_namenftx+firm_nameopensea+firm_namerarible+firm_namesudoswap+firm_namex2y2, data = df_panel, model = "pooling")
pow_m_3 <- plm(firm_potential_wallet ~ 1+firm_size_wallet + firm_size_wallet:market_share_perc, data = df_panel, model = "pooling")

#################################################################################
# firm_sow_fee
#################################################################################

sow_m_1 <- plm(firm_sow_fee_perc ~ 1+firm_size_wallet, data = df_panel, model = "pooling")
sow_m_2 <-  plm(firm_sow_fee_perc ~ 1+ firm_nameblur:firm_size_wallet + firm_namegem:firm_size_wallet + firm_namelooksrare:firm_size_wallet + firm_namenftx:firm_size_wallet + firm_nameopensea:firm_size_wallet + firm_namerarible:firm_size_wallet + firm_namesudoswap:firm_size_wallet + firm_namex2y2:firm_size_wallet+ firm_namegem+firm_namelooksrare+firm_namenftx+firm_nameopensea+firm_namerarible+firm_namesudoswap+firm_namex2y2, data = df_panel, model = "pooling")
sow_m_3 <- plm(firm_sow_fee_perc ~ 1+firm_size_wallet + firm_size_wallet:market_share_perc, data = df_panel, model = "pooling")

#################################################################################
# Total Wallet
#################################################################################

tow_m_1 <- plm(total_wallet ~ 1+firm_size_wallet, data = df_panel, model = "pooling")
tow_m_2 <-  plm(total_wallet ~ 1+ firm_nameblur:firm_size_wallet + firm_namegem:firm_size_wallet + firm_namelooksrare:firm_size_wallet + firm_namenftx:firm_size_wallet + firm_nameopensea:firm_size_wallet + firm_namerarible:firm_size_wallet + firm_namesudoswap:firm_size_wallet + firm_namex2y2:firm_size_wallet+ firm_namegem+firm_namelooksrare+firm_namenftx+firm_nameopensea+firm_namerarible+firm_namesudoswap+firm_namex2y2, data = df_panel, model = "pooling")
tow_m_3 <- plm(total_wallet ~ 1+firm_size_wallet + firm_size_wallet:market_share_perc, data = df_panel, model = "pooling")

# Generate HTML output with stargazer
stargazer(pow_m_1, pow_m_2, pow_m_3, type = "html", title = "Regression Results Market", out = "pow_regression_results_by_market-2024-07-20a.html")
stargazer(sow_m_1, sow_m_2, sow_m_3, type = "html", title = "Regression Results Market", out = "sow_regression_results_by_market-2024-07-20a.html")
stargazer(tow_m_1, tow_m_2, tow_m_3, type = "html", title = "Regression Results Market", out = "tow_regression_results_by_market-2024-07-20a.html")



