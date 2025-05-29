import os
import json
from datetime import datetime
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
endpoint = os.getenv("COSMOS_ENDPOINT")
key = os.getenv("COSMOS_KEY")
database_name = os.getenv("COSMOS_DATABASE_NAME")
container_name = os.getenv("container_name")

# Connect to Cosmos DB
client = CosmosClient(endpoint, key)
db = client.create_database_if_not_exists(id=database_name)
container = db.create_container_if_not_exists(
    id=container_name,
    partition_key=PartitionKey(path="/id"),
    offer_throughput=400
)

# Today's date formatted as "YYYYMMDD.json"
today_filename = datetime.today().strftime("%Y%m%d.json")

# Directory to save the file
output_dir = "D:\Browns\CSE ANALYSER\Local Storage"
os.makedirs(output_dir, exist_ok=True)

# Try to read the document with today's date as id
try:
    document = container.read_item(item=today_filename, partition_key=today_filename)
    
    file_path = os.path.join(output_dir, today_filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=4)
    
    print(f"Downloaded document '{today_filename}' to '{file_path}'")

except exceptions.CosmosResourceNotFoundError:
    print(f"No document found with id '{today_filename}' in the container.")
