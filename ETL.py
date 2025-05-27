import argparse
import os
import json
import pandas as pd
from collections import defaultdict
from azure.cosmos import CosmosClient
from CosmosObjects import CosmosObjects as co

# Set up argument parser
parser = argparse.ArgumentParser(description="Process company symbol for ETL pipeline.")
parser.add_argument("company_symbol", type=str, help="Company symbol (e.g., WIND.N0000)")
args = parser.parse_args()

# Assign the parsed company symbol to a variable
comp_symbol = args.company_symbol.strip()

# Query Cosmos DB container for all items
endpoint = os.getenv("COSMOS_ENDPOINT")
key = os.getenv("COSMOS_KEY")
database_name = os.getenv("COSMOS_DATABASE_NAME")
container_name = os.getenv("container_name")

client = CosmosClient(endpoint, credential=key)
database = client.get_database_client(database_name)
container = database.get_container_client(container_name)

# Fetch all items from the container
query = "SELECT * FROM c"
results = list(container.query_items(query=query, enable_cross_partition_query=True))

# Find the data for the given symbol
company_data = next((item for item in results if item.get("id") == comp_symbol), None)

# Save to JSON
if company_data:
    with open(f"{comp_symbol}.json", "w", encoding="utf-8") as f:
        json.dump(company_data, f, indent=4)
else:
    print(f"{comp_symbol} not found in the Cosmos results.")
    exit()

# Step 2: Load the JSON data
with open(f"{comp_symbol}.json", 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

# Extract and convert to DataFrame
if comp_symbol in data:
    records = data[comp_symbol]
    df = pd.DataFrame(records)
else:
    raise ValueError(f"Key '{comp_symbol}' not found in the JSON file.")

# Clean BOM if present
df.columns = [col.replace('\ufeff', '') for col in df.columns]

# Filter columns
df_filtered = df[['Trade Date', 'Close (Rs.)']].copy()

# Convert Trade Date to datetime
df_filtered['Trade Date'] = pd.to_datetime(df_filtered['Trade Date'])

# Step 3: Append additional data from daily uploads (IDs starting with '20')
endpoint = os.getenv("COSMOS_ENDPOINT")
key = os.getenv("COSMOS_KEY")
database_name = os.getenv("COSMOS_DATABASE_NAME")
container_name = os.getenv("container_name")

client = CosmosClient(endpoint, credential=key)
database = client.get_database_client(database_name)
container = database.get_container_client(container_name)

# Query for IDs starting with '20'
query = "SELECT c.id FROM c WHERE STARTSWITH(c.id, '20')"
items = list(container.query_items(query=query, enable_cross_partition_query=True))
id_list = [item["id"] for item in items]

# Collect company-wise data
company_data = defaultdict(list)
for doc_id in id_list:
    document = container.read_item(item=doc_id, partition_key=doc_id)
    upload_date = document["upload_date"]
    for company in document["data"]:
        symbol = company["Symbol"]
        if symbol != comp_symbol:
            continue  # Skip other symbols
        last_trade_str = company.get("Last Trade Rs", "0").replace(",", "")
        try:
            last_trade = float(last_trade_str)
        except ValueError:
            last_trade = None
        company_data[symbol].append({
            "Trade Date": pd.to_datetime(upload_date, format="%Y%m%d"),
            "Close (Rs.)": last_trade
        })

# Create DataFrame from new data
if comp_symbol in company_data:
    df_new = pd.DataFrame(company_data[comp_symbol])
    df_combined = pd.concat([df_filtered, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset="Trade Date", keep="last", inplace=True)
    df_combined.sort_values(by="Trade Date", inplace=True)
else:
    print(f"No additional data found for {comp_symbol}")
    df_combined = df_filtered

# Save to CSV
df_combined.to_csv("Company_stock_price.csv", index=False, encoding='utf-8-sig')

# Show output
print(f"\nFinal combined stock price data for {comp_symbol}:")
print(df_combined.tail())
print(df_combined)