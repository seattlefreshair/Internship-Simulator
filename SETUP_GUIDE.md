# Interview Simulator Setup Guide

## Overview
This application now includes comprehensive user authentication and user-specific resume management. Each user can have their own account and manage their personal resumes and interview sessions.

## Features Added

### 🔐 User Authentication
- User registration and login system
- Secure password hashing
- JWT-style session tokens
- Protected routes and endpoints

### 👤 User Profile Management
- Personal profile information
- Update name and contact details
- View account creation date

### 📄 Resume Management
- User-specific resume storage
- Upload, view, and delete resumes
- Resume-based interview questions
- Secure file storage per user

### 💬 Interview Sessions
- Personalized interview questions
- Session history tracking
- User-specific chat history

## Setup Instructions

### 1. Environment Variables
Create a `.env` file in the `backend/` directory with the following variables:

```bash
# Azure Cosmos DB Configuration
AZURE_COSMOS_ENDPOINT=your_cosmos_db_endpoint_here
AZURE_COSMOS_KEY=your_cosmos_db_key_here
AZURE_COSMOS_DATABASE=resume_database
AZURE_COSMOS_CONTAINER=resumes

# Azure Blob Storage Configuration
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string_here
AZURE_STORAGE_CONTAINER=resumes

# AI API Keys (at least one is required)
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Application Settings
SECRET_KEY=your_secret_key_here
ENVIRONMENT=development
```

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the Application
```bash
cd backend
python app.py
```

The application will be available at:
- **Login/Register**: `http://localhost:8000/`
- **Main Application**: `http://localhost:8000/app`

## Usage Flow

### 1. User Registration
- Navigate to `/` (root)
- Click "Don't have an account? Register"
- Fill in name, email, and password
- Submit registration

### 2. User Login
- Use registered email and password
- System validates credentials
- Creates secure session token
- Redirects to main application

### 3. Main Application
- **Basic Questions Tab**: Generate interview questions by subject/topic
- **Resume Upload Tab**: Upload and manage personal resumes
- **My Profile Tab**: Manage account information and view resumes

### 4. Resume Management
- Upload resumes (PDF, DOC, DOCX, TXT)
- Generate personalized interview questions
- View and delete uploaded resumes
- All data is user-specific and secure

## Security Features

- **Password Hashing**: SHA-256 encryption
- **Session Tokens**: Secure, time-limited access
- **User Isolation**: Users can only access their own data
- **Input Validation**: Sanitized file uploads and form data
- **Protected Endpoints**: Authentication required for all user actions

## Database Schema

### Users Collection
```json
{
  "id": "unique_user_id",
  "type": "user",
  "email": "user@example.com",
  "password_hash": "hashed_password",
  "name": "Full Name",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "is_active": true,
  "user_id": "partition_key"
}
```

### Resumes Collection
```json
{
  "id": "unique_resume_id",
  "type": "resume",
  "user_id": "owner_user_id",
  "file_url": "blob_storage_url",
  "file_name": "resume.pdf",
  "file_size": 1024000,
  "extracted_text": "resume content...",
  "upload_date": "2024-01-01T00:00:00Z",
  "parsed_data": {}
}
```

## API Endpoints

### Authentication
- `POST /register` - User registration
- `POST /login` - User login
- `POST /logout` - User logout

### User Management
- `GET /users/{user_id}` - Get user profile
- `PUT /users/{user_id}` - Update user profile

### Resume Management
- `POST /upload-resume` - Upload resume file
- `GET /resumes` - Get user's resumes
- `GET /resumes/{resume_id}` - Get specific resume
- `DELETE /resumes/{resume_id}` - Delete resume
- `GET /search-resumes` - Search resume content

### Interview Questions
- `POST /get-question` - Generate basic question
- `POST /generate-resume-question` - Generate resume-based question
- `POST /submit-response` - Submit interview response
- `POST /submit-resume-response` - Submit resume response

## Troubleshooting

### Common Issues

1. **Template Not Found Error**
   - Ensure you're running from the `backend/` directory
   - Check that `frontend/` directory exists at the same level

2. **Authentication Errors**
   - Verify environment variables are set correctly
   - Check Azure Cosmos DB connection
   - Ensure user service is properly initialized

3. **File Upload Issues**
   - Verify Azure Blob Storage connection
   - Check file size limits (5MB max)
   - Ensure supported file types

4. **AI Service Errors**
   - Check API keys are valid
   - Verify internet connectivity
   - Check API rate limits

### Debug Mode
Set `ENVIRONMENT=development` in your `.env` file to see detailed error messages.

## Support

For issues or questions:
1. Check the console output for error messages
2. Verify all environment variables are set
3. Ensure Azure services are properly configured
4. Check network connectivity and firewall settings 