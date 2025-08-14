import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from azure.cosmos import CosmosClient, PartitionKey

class UserService:
    def __init__(self):
        # Initialize Cosmos DB client for users
        self.cosmos_endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")
        self.cosmos_key = os.getenv("AZURE_COSMOS_KEY")
        self.database_name = os.getenv("AZURE_COSMOS_DATABASE", "resume_database")
        self.container_name = os.getenv("AZURE_COSMOS_CONTAINER", "resumes")  # Use same container as resumes
        
        # Initialize clients
        self.cosmos_client = None
        self.database = None
        self.container = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Azure Cosmos DB client for users"""
        try:
            if self.cosmos_endpoint and self.cosmos_key:
                self.cosmos_client = CosmosClient(self.cosmos_endpoint, self.cosmos_key)
                self.database = self.cosmos_client.get_database_client(self.database_name)
                self.container = self.database.get_container_client(self.container_name)
                print("✅ User service Cosmos DB client initialized successfully")
            else:
                print("⚠️  User service: Azure Cosmos DB credentials not found")
                
        except Exception as e:
            print(f"❌ Error initializing user service clients: {e}")
    
    def create_users_container(self):
        """Create the users container if it doesn't exist (now uses same container as resumes)"""
        try:
            # Users will be stored in the same container as resumes, distinguished by 'type' field
            # The container is created by azure_db.py, so we just need to ensure it exists
            print(f"✅ Users will be stored in container '{self.container_name}' (shared with resumes)")
            
        except Exception as e:
            print(f"❌ Error setting up users container: {e}")
    
    def create_user(self, email: str, password_hash: str, name: str = "") -> str:
        """Create a new user"""
        try:
            if not self.container:
                raise Exception("Users container not configured")
            
            # Check if user already exists
            existing_user = self.get_user_by_email(email)
            if existing_user:
                raise Exception("User with this email already exists")
            
            # Create user document with type field to distinguish from resumes
            user_doc = {
                "id": str(uuid.uuid4()),
                "type": "user",  # Distinguish from resume documents
                "email": email,
                "password_hash": password_hash,
                "name": name,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "is_active": True,
                "user_id": str(uuid.uuid4())  # Use as partition key
            }
            
            # Save to Cosmos DB
            result = self.container.create_item(body=user_doc)
            print(f"✅ User created with ID: {result['id']}")
            return result['id']
            
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            raise
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            if not self.container:
                raise Exception("Users container not configured")
            
            # Query user by email and type
            query = "SELECT * FROM c WHERE c.email = @email AND c.type = 'user'"
            parameters = [{"name": "@email", "value": email}]
            
            users = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True  # Enable cross-partition query for users
            ))
            
            return users[0] if users else None
            
        except Exception as e:
            print(f"❌ Error getting user by email: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            if not self.container:
                raise Exception("Users container not configured")
            
            # Query user by ID and type
            query = "SELECT * FROM c WHERE c.id = @user_id AND c.type = 'user'"
            parameters = [{"name": "@user_id", "value": user_id}]
            
            users = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            return users[0] if users else None
            
        except Exception as e:
            print(f"❌ Error getting user by ID: {e}")
            return None
    
    def authenticate_user(self, email: str, password_hash: str) -> Optional[Dict[str, Any]]:
        """Authenticate user by email and password"""
        try:
            user = self.get_user_by_email(email)
            if user and user.get("password_hash") == password_hash and user.get("is_active", True):
                return user
            return None
            
        except Exception as e:
            print(f"❌ Error authenticating user: {e}")
            return None
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user information"""
        try:
            if not self.container:
                raise Exception("Users container not configured")
            
            # Get current user
            user = self.get_user_by_id(user_id)
            if not user:
                return False
            
            # Update fields
            user.update(updates)
            user["updated_at"] = datetime.utcnow().isoformat()
            
            # Save updated user
            self.container.replace_item(item=user_id, body=user)
            print(f"✅ User {user_id} updated successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error updating user: {e}")
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user (soft delete by setting is_active to False)"""
        try:
            if not self.container:
                raise Exception("Users container not configured")
            
            # Get current user
            user = self.get_user_by_id(user_id)
            if not user:
                return False
            
            # Soft delete by setting is_active to False
            user["is_active"] = False
            user["updated_at"] = datetime.utcnow().isoformat()
            
            # Save updated user
            self.container.replace_item(item=user_id, body=user)
            print(f"✅ User {user_id} deleted successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting user: {e}")
            return False

# Create global instance
user_service = UserService() 