from dotenv import load_dotenv, find_dotenv
import os
import json
from CosmosObjects import CosmosObjects as co
import pandas as pd

# Load environment variables
load_dotenv(find_dotenv())

# Query Cosmos DB
cosmos_query = "select * from c"
container = co.getCosmosContainer(os.environ["container_name"])
results = list(container.query_items(query=cosmos_query, enable_cross_partition_query=True))

# Get company symbol from user
comp_symbol = input("Enter the company symbol (e.g., WIND.N0000): ").strip()

# Find the data for the given symbol
company_data = next((item for item in results if item.get("id") == comp_symbol), None)

# Save to file
if company_data:
    with open(f"{comp_symbol}.json", "w", encoding="utf-8") as f:
        json.dump(company_data, f, indent=4)
else:
    print(f"{comp_symbol} not found in the Cosmos results.")
    exit()

# Load the JSON file
with open(f"{comp_symbol}.json", 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

# Extract the relevant data from the dynamic key (comp_symbol)
if comp_symbol in data:
    records = data[comp_symbol]
    df = pd.DataFrame(records)
else:
    raise ValueError(f"Key '{comp_symbol}' not found in the JSON file.")

# Clean up BOM if any
df.columns = [col.replace('\ufeff', '') for col in df.columns]

# Filter required columns
df_filtered = df[['Trade Date', 'Close (Rs.)']]
print(df_filtered)

# Save the filtered DataFrame to a CSV file
df_filtered.to_csv("Company_stock_price.csv", index=False, encoding='utf-8-sig')



