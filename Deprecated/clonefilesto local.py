import os
import json
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv
from json_repair import repair_json

# Load environment variables
load_dotenv()
endpoint = os.getenv("COSMOS_ENDPOINT")
key = os.getenv("COSMOS_KEY")
database_name = os.getenv("COSMOS_DATABASE_NAME")
container_name = os.getenv("container_name")
#database_name = "your_database_name"  # Replace with your database name
# Connect to Cosmos DB
client = CosmosClient(endpoint, key)
db = client.create_database_if_not_exists(id=database_name)
container = db.create_container_if_not_exists(
    id=container_name,
    partition_key=PartitionKey(path="/id"),
    offer_throughput=400
)

# Directory to save downloaded files
output_dir = "D:\Browns\CSE ANALYSER\Local Storage"

os.makedirs(output_dir, exist_ok=True)

# Read all documents in the container
for item in container.read_all_items():
    item_id = item.get("id", "unknown_id")
    
    # Sanitize file name and prepare JSON output path
    file_path = os.path.join(output_dir, f"{item_id}.json")
    
    # Save each document as a JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=4)

print(f"Downloaded all documents to '{output_dir}' directory.")
