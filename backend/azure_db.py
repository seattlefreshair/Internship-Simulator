import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from azure.cosmos import CosmosClient, PartitionKey
from azure.storage.blob import BlobServiceClient
import json

class AzureResumeService:
    def __init__(self):
        # Initialize Cosmos DB client
        self.cosmos_endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")
        self.cosmos_key = os.getenv("AZURE_COSMOS_KEY")
        self.database_name = os.getenv("AZURE_COSMOS_DATABASE", "resume_database")
        self.container_name = os.getenv("AZURE_COSMOS_CONTAINER", "resumes")
        
        # Initialize Azure Storage client
        self.storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.storage_container_name = os.getenv("AZURE_STORAGE_CONTAINER", "resume-files")
        
        # Initialize clients
        self.cosmos_client = None
        self.database = None
        self.container = None
        self.blob_service_client = None
        self.blob_container_client = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Azure Cosmos DB and Storage clients"""
        try:
            # Initialize Cosmos DB
            if self.cosmos_endpoint and self.cosmos_key:
                self.cosmos_client = CosmosClient(self.cosmos_endpoint, self.cosmos_key)
                self.database = self.cosmos_client.get_database_client(self.database_name)
                self.container = self.database.get_container_client(self.container_name)
                print("✅ Azure Cosmos DB client initialized successfully")
            else:
                print("⚠️  Azure Cosmos DB credentials not found")
            
            # Initialize Azure Storage
            if self.storage_connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    self.storage_connection_string
                )
                self.blob_container_client = self.blob_service_client.get_container_client(
                    self.storage_container_name
                )
                print("✅ Azure Storage client initialized successfully")
            else:
                print("⚠️  Azure Storage credentials not found")
                
        except Exception as e:
            print(f"❌ Error initializing Azure clients: {e}")
    
    def create_database_and_container(self):
        """Create the database and container if they don't exist"""
        try:
            # Create database
            self.database = self.cosmos_client.create_database_if_not_exists(self.database_name)
            print(f"✅ Database '{self.database_name}' ready")
            
            # Create container with partition key (serverless mode)
            self.container = self.database.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path="/user_id")
                # No offer_throughput specified = serverless mode
            )
            print(f"✅ Container '{self.container_name}' ready")
            
        except Exception as e:
            print(f"❌ Error creating database/container: {e}")
    
    def upload_resume_file(self, file_content: bytes, file_name: str) -> str:
        """Upload resume file to Azure Blob Storage and return the URL"""
        try:
            if not self.blob_container_client:
                raise Exception("Azure Storage not configured")
            
            # Generate unique blob name
            blob_name = f"{uuid.uuid4()}_{file_name}"
            blob_client = self.blob_container_client.get_blob_client(blob_name)
            
            # Upload file
            blob_client.upload_blob(file_content, overwrite=True)
            
            # Return the blob URL
            return blob_client.url
            
        except Exception as e:
            print(f"❌ Error uploading file to Azure Storage: {e}")
            raise
    
    def save_resume_metadata(self, user_id: str, file_url: str, file_name: str, 
                           file_size: int, extracted_text: str = "", 
                           parsed_data: Dict[str, Any] = None) -> str:
        """Save resume metadata to Cosmos DB"""
        try:
            if not self.container:
                raise Exception("Cosmos DB not configured")
            
            # Create resume document
            resume_doc = {
                "id": str(uuid.uuid4()),
                "type": "resume",  # Distinguish from user documents
                "user_id": user_id,
                "file_url": file_url,
                "file_name": file_name,
                "file_size": file_size,
                "extracted_text": extracted_text,
                "parsed_data": parsed_data or {},
                "uploaded_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Save to Cosmos DB
            result = self.container.create_item(body=resume_doc)
            print(f"✅ Resume metadata saved with ID: {result['id']}")
            return result['id']
            
        except Exception as e:
            print(f"❌ Error saving resume metadata: {e}")
            raise
    
    def get_user_resumes(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all resumes for a specific user"""
        try:
            if not self.container:
                raise Exception("Cosmos DB not configured")
            
            # Query resumes by user_id and type
            query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.type = 'resume' ORDER BY c.uploaded_at DESC"
            parameters = [{"name": "@user_id", "value": user_id}]
            
            resumes = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=False
            ))
            
            return resumes
            
        except Exception as e:
            print(f"❌ Error getting user resumes: {e}")
            return []
    
    def get_resume_by_id(self, resume_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific resume by ID"""
        try:
            if not self.container:
                raise Exception("Cosmos DB not configured")
            
            # Get resume by ID and user_id (partition key)
            resume = self.container.read_item(
                item=resume_id,
                partition_key=user_id
            )
            
            return resume
            
        except Exception as e:
            print(f"❌ Error getting resume by ID: {e}")
            return None
    
    def delete_resume(self, resume_id: str, user_id: str) -> bool:
        """Delete a resume from both Cosmos DB and Blob Storage"""
        try:
            if not self.container or not self.blob_container_client:
                raise Exception("Azure services not configured")
            
            # Get resume metadata first
            resume = self.get_resume_by_id(resume_id, user_id)
            if not resume:
                return False
            
            # Delete from Cosmos DB
            self.container.delete_item(
                item=resume_id,
                partition_key=user_id
            )
            
            # Delete from Blob Storage
            file_url = resume.get('file_url', '')
            if file_url:
                # Extract blob name from URL
                blob_name = file_url.split('/')[-1]
                blob_client = self.blob_container_client.get_blob_client(blob_name)
                blob_client.delete_blob()
            
            print(f"✅ Resume {resume_id} deleted successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting resume: {e}")
            return False
    
    def update_resume_metadata(self, resume_id: str, user_id: str, 
                             updates: Dict[str, Any]) -> bool:
        """Update resume metadata"""
        try:
            if not self.container:
                raise Exception("Cosmos DB not configured")
            
            # Get current resume
            resume = self.get_resume_by_id(resume_id, user_id)
            if not resume:
                return False
            
            # Update fields
            resume.update(updates)
            resume['updated_at'] = datetime.utcnow().isoformat()
            
            # Save updated resume
            self.container.replace_item(
                item=resume_id,
                body=resume,
                partition_key=user_id
            )
            
            print(f"✅ Resume {resume_id} updated successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error updating resume: {e}")
            return False
    
    def search_resumes(self, user_id: str, search_term: str) -> List[Dict[str, Any]]:
        """Search resumes by text content"""
        try:
            if not self.container:
                raise Exception("Cosmos DB not configured")
            
            # Simple text search in extracted_text
            query = """
            SELECT * FROM c 
            WHERE c.user_id = @user_id 
            AND c.type = 'resume'
            AND CONTAINS(c.extracted_text, @search_term, true)
            ORDER BY c.uploaded_at DESC
            """
            parameters = [
                {"name": "@user_id", "value": user_id},
                {"name": "@search_term", "value": search_term}
            ]
            
            resumes = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=False
            ))
            
            return resumes
            
        except Exception as e:
            print(f"❌ Error searching resumes: {e}")
            return []

# Global instance
azure_service = AzureResumeService() 