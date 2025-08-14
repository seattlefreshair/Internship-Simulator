# Azure Setup Guide for Resume Upload System

This guide will help you set up Azure Cosmos DB and Azure Storage for storing resumes and user data.

## Prerequisites

- Azure account (free tier available)
- Python 3.7+ installed
- FastAPI backend running

## Step 1: Create Azure Resources

### 1.1 Create Azure Cosmos DB Account

1. Go to [Azure Portal](https://portal.azure.com)
2. Click "Create a resource"
3. Search for "Azure Cosmos DB" and select it
4. Click "Create"
5. Fill in the details:
   - **Subscription**: Your subscription
   - **Resource Group**: Create new or use existing
   - **Account Name**: `resume-db-{your-unique-name}`
   - **API**: Select "Core (SQL)"
   - **Location**: Choose closest to you
   - **Capacity mode**: "Provisioned throughput"
   - **Apply Free Tier Discount**: Yes (if available)
6. Click "Review + create" then "Create"

### 1.2 Create Azure Storage Account

1. In Azure Portal, click "Create a resource"
2. Search for "Storage account" and select it
3. Click "Create"
4. Fill in the details:
   - **Subscription**: Your subscription
   - **Resource Group**: Same as Cosmos DB
   - **Storage account name**: `resumestorage{your-unique-name}`
   - **Location**: Same as Cosmos DB
   - **Performance**: Standard
   - **Redundancy**: LRS (for free tier)
5. Click "Review + create" then "Create"

## Step 2: Get Connection Information

### 2.1 Get Cosmos DB Connection String

1. Go to your Cosmos DB account
2. Click "Keys" in the left menu
3. Copy the **URI** and **PRIMARY KEY**
4. These will be your `AZURE_COSMOS_ENDPOINT` and `AZURE_COSMOS_KEY`

### 2.2 Get Storage Connection String

1. Go to your Storage account
2. Click "Access keys" in the left menu
3. Copy the **Connection string** (key1)
4. This will be your `AZURE_STORAGE_CONNECTION_STRING`

## Step 3: Configure Environment Variables

1. Copy `env_example.txt` to `.env`:
   ```bash
   cp env_example.txt .env
   ```

2. Edit `.env` and add your Azure credentials:
   ```env
   # Azure Cosmos DB Configuration
   AZURE_COSMOS_ENDPOINT=https://your-cosmos-account.documents.azure.com:443/
   AZURE_COSMOS_KEY=your_cosmos_db_primary_key_here
   AZURE_COSMOS_DATABASE=resume_database
   AZURE_COSMOS_CONTAINER=resumes

   # Azure Storage Configuration
   AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=yourstorageaccount;AccountKey=yourstoragekey;EndpointSuffix=core.windows.net
   AZURE_STORAGE_CONTAINER=resume-files
   ```

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 5: Run Setup Script

```bash
python setup_azure.py
```

This script will:
- Check your environment configuration
- Test connections to Azure services
- Create sample data for testing

## Step 6: Start the Application

```bash
python app.py
```

## API Endpoints

### Authentication
- `POST /register` - Register a new user
- `POST /login` - Login user
- `POST /logout` - Logout user

### Resume Management
- `POST /upload-resume` - Upload a resume (requires auth)
- `GET /resumes` - List user resumes (requires auth)
- `GET /resumes/{resume_id}` - Get specific resume (requires auth)
- `DELETE /resumes/{resume_id}` - Delete resume (requires auth)
- `GET /search-resumes?search_term=python` - Search resumes (requires auth)

## Testing the Setup

### 1. Register a User
```bash
curl -X POST "http://localhost:8000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "password123",
    "name": "Test User"
  }'
```

### 2. Login
```bash
curl -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "password123"
  }'
```

### 3. Upload Resume (use the token from login)
```bash
curl -X POST "http://localhost:8000/upload-resume" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -F "resume=@/path/to/your/resume.pdf"
```

### 4. List Resumes
```bash
curl -X GET "http://localhost:8000/resumes" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Database Schema

### Users Container
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "password_hash": "hashed_password",
  "name": "User Name",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "is_active": true
}
```

### Resumes Container
```json
{
  "id": "uuid",
  "user_id": "user_uuid",
  "file_url": "https://storage.blob.core.windows.net/container/file.pdf",
  "file_name": "resume.pdf",
  "file_size": 1024,
  "extracted_text": "Resume content...",
  "parsed_data": {
    "skills": ["Python", "JavaScript"],
    "experience": ["Software Engineer"]
  },
  "uploaded_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check your connection strings
   - Ensure your Azure resources are created and running
   - Verify your IP is not blocked by firewall

2. **Authentication Errors**
   - Check your Cosmos DB key is correct
   - Ensure your storage account key is correct

3. **Container Not Found**
   - Run the setup script to create containers
   - Check container names match your .env file

4. **File Upload Fails**
   - Check storage connection string
   - Ensure storage container exists
   - Verify file size is within limits

### Free Tier Limits

- **Cosmos DB**: 1000 RU/s, 25GB storage
- **Storage**: 5GB storage, 20,000 transactions/month

## Security Notes

- Never commit your `.env` file to version control
- Use strong passwords for production
- Consider using Azure Key Vault for production secrets
- Enable CORS appropriately for your frontend

## Next Steps

1. Add frontend integration
2. Implement file validation
3. Add resume parsing with AI
4. Set up monitoring and logging
5. Configure backup and disaster recovery 