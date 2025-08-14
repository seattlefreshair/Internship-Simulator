#!/usr/bin/env python3
"""
Azure Setup Script for Resume Upload System
This script helps you configure and test your Azure Cosmos DB and Storage setup.
"""

import os
import sys
from dotenv import load_dotenv

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🚀 Azure Resume Upload System Setup")
    print("=" * 60)
    print()

def check_env_file():
    """Check if .env file exists and has required variables"""
    print("📋 Checking environment configuration...")
    
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("   Please copy env_example.txt to .env and fill in your Azure credentials.")
        return False
    
    load_dotenv()
    
    required_vars = [
        'AZURE_COSMOS_ENDPOINT',
        'AZURE_COSMOS_KEY', 
        'AZURE_STORAGE_CONNECTION_STRING'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("   Please update your .env file with the missing values.")
        return False
    
    print("✅ Environment variables configured")
    return True

def test_azure_connection():
    """Test Azure services connection"""
    print("\n🔗 Testing Azure services connection...")
    
    try:
        from azure_db import azure_service
        from user_service import user_service
        
        # Test Cosmos DB connection
        print("   Testing Cosmos DB connection...")
        azure_service.create_database_and_container()
        print("   ✅ Cosmos DB connection successful")
        
        # Test user service
        print("   Testing user service...")
        user_service.create_users_container()
        print("   ✅ User service connection successful")
        
        # Test Azure Storage connection
        print("   Testing Azure Storage connection...")
        if azure_service.blob_container_client:
            print("   ✅ Azure Storage connection successful")
        else:
            print("   ⚠️  Azure Storage not configured")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("\n📝 Creating sample data...")
    
    try:
        from azure_db import azure_service
        from user_service import user_service
        import hashlib
        
        # Create a test user
        test_email = "test@example.com"
        test_password = "testpassword123"
        password_hash = hashlib.sha256(test_password.encode()).hexdigest()
        
        # Check if user already exists
        existing_user = user_service.get_user_by_email(test_email)
        if existing_user:
            print("   ✅ Test user already exists")
            user_id = existing_user['id']
        else:
            user_id = user_service.create_user(test_email, password_hash, "Test User")
            print(f"   ✅ Created test user with ID: {user_id}")
        
        # Create a sample resume document
        sample_resume = {
            "user_id": user_id,
            "file_url": "https://example.com/sample-resume.pdf",
            "file_name": "sample-resume.pdf",
            "file_size": 1024,
            "extracted_text": "Sample resume content for testing purposes.",
            "parsed_data": {
                "skills": ["Python", "JavaScript", "SQL"],
                "experience": ["Software Engineer", "Data Analyst"]
            }
        }
        
        # Check if sample resume exists
        user_resumes = azure_service.get_user_resumes(user_id)
        if user_resumes:
            print("   ✅ Sample resume data already exists")
        else:
            azure_service.save_resume_metadata(
                user_id=user_id,
                file_url=sample_resume["file_url"],
                file_name=sample_resume["file_name"],
                file_size=sample_resume["file_size"],
                extracted_text=sample_resume["extracted_text"],
                parsed_data=sample_resume["parsed_data"]
            )
            print("   ✅ Created sample resume data")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create sample data: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Start your FastAPI server: python app.py")
    print("2. Test the API endpoints:")
    print("   - POST /register - Register a new user")
    print("   - POST /login - Login with credentials")
    print("   - POST /upload-resume - Upload a resume (requires auth)")
    print("   - GET /resumes - List user resumes (requires auth)")
    print()
    print("Test credentials:")
    print("   Email: test@example.com")
    print("   Password: testpassword123")
    print()
    print("API Documentation:")
    print("   http://localhost:8000/docs")
    print()

def main():
    """Main setup function"""
    print_banner()
    
    # Check environment
    if not check_env_file():
        sys.exit(1)
    
    # Test connections
    if not test_azure_connection():
        print("\n❌ Azure connection failed. Please check your credentials.")
        sys.exit(1)
    
    # Create sample data
    if not create_sample_data():
        print("\n⚠️  Sample data creation failed, but setup can continue.")
    
    print_next_steps()

if __name__ == "__main__":
    main() 