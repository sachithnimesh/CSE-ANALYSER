from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # read local .env file
import os
import json
from CosmosObjects import CosmosObjects as co
import pandas as pd

cosmos_query = "select * from c"
container = co.getCosmosContainer(os.environ["container_name"])
results = list(container.query_items(query=cosmos_query, enable_cross_partition_query=True))

# # Write the results to a JSON file
# with open("output.json", "w") as f:
#     json.dump(results, f, indent=4)
comp_symbol = input("Enter the company symbol (e.g., WIND.N0000): ").strip()
# Check if the input is valid

#Access the JSON file WIND.N0000.json from Cosmos
wind_data = next((item for item in results if item.get("id") == comp_symbol), None)

if wind_data:
        with open(f"{comp_symbol}.json", "w") as f:
            json.dump(wind_data, f, indent=4)
else:
        print("WIND.N0000 not found in the Cosmos results.")


#Read the WIND.N0000.json file

# Load the JSON file
with open(f"{comp_symbol}.json", 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

# Extract the list under the key "WIND.N0000"
if "WIND.N0000" in data:
    records = data["WIND.N0000"]
    df = pd.DataFrame(records)
else:
    raise ValueError("Key 'WIND.N0000' not found in the JSON file.")

# Clean up the BOM from the column name if it exists
df.columns = [col.replace('\ufeff', '') for col in df.columns]

# Now you can safely access the correct columns
df_filtered = df[['Trade Date', 'Close (Rs.)']]
print(df_filtered)
