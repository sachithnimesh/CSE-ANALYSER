import os

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos import exceptions as AzureCosmosExceptions
from azure.cosmos import database as AzureCosmosDatabase
from azure.cosmos import container as AzureCosmosContainer

    
class CosmosObjects:
    """Singleton class to hold cosmos objects. (So that wasteful similar calls to cosmos are not made, each time a reference to an object is required).
    
    Attributes:
        __cosmosClient: CosmosClient = None
        __cosmosDatabase: AzureCosmosDatabase.DatabaseProxy = None
        __cosmosContainer: AzureCosmosContainer.ContainerProxy = None

    Methods:
        3 Methods to get each of the above attributes
    """

    __cosmosClient: CosmosClient = None
    __cosmosDatabase: AzureCosmosDatabase.DatabaseProxy = None
    __cosmosContainers: dict[str,AzureCosmosContainer.ContainerProxy] = {}
    
    @classmethod
    def __getCosmosClient(cls) -> CosmosClient:
        """Singleton method to return a Cosmos object"""
        if CosmosObjects.__cosmosClient == None:
            ENDPOINT = os.environ["COSMOS_ENDPOINT"]
            KEY = os.environ["COSMOS_KEY"]
            CosmosObjects.__cosmosClient = CosmosClient(url=ENDPOINT, credential=KEY)

        return CosmosObjects.__cosmosClient


    @classmethod
    def __getCosmosDatabase(cls) -> AzureCosmosDatabase.DatabaseProxy:
        """Singleton method to return a Cosmos object"""
        if CosmosObjects.__cosmosDatabase == None:
            try:
                cosmosClient = CosmosObjects.__getCosmosClient()
                CosmosObjects.__cosmosDatabase = cosmosClient.create_database_if_not_exists(id=os.environ["COSMOS_DATABASE_NAME"])

            except AzureCosmosExceptions.CosmosHttpResponseError:
                print("CosmosHttpResponseError!")
        
        return CosmosObjects.__cosmosDatabase
    

    @classmethod
    def getCosmosContainer(cls, container_name: str) -> AzureCosmosContainer.ContainerProxy:
        """Singleton method to return a Cosmos object

        Args:
            release (str): The release (each release has its own container)

        Returns:
            AzureCosmosContainer.ContainerProxy: container
        """
        if container_name not in cls.__cosmosContainers:
            try:
                partition_key_path = PartitionKey(path="/id")
                database = CosmosObjects.__getCosmosDatabase()
                cls.__cosmosContainers[container_name] = database.create_container_if_not_exists(
                    id=container_name,
                    partition_key=partition_key_path
                )

            except AzureCosmosExceptions.CosmosHttpResponseError:
                print("Request to the Azure Cosmos database service failed.")

        return cls.__cosmosContainers[container_name]