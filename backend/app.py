from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import random
from typing import List, Optional
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import PyPDF2
import io
import docx
import hashlib
import secrets

# Load environment variables
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="../frontend")

# Import Azure services
from azure_db import azure_service
from user_service import user_service

# Security
security = HTTPBearer()

# In-memory session storage (in production, use Redis or database)
user_sessions = {}

# Initialize AI clients
GEMINI_AVAILABLE = False
OPENAI_AVAILABLE = False
gemini_client = None
openai_client = None

# Try to initialize Google Gemini (primary choice)
try:
    import google.generativeai as genai
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_client = genai.GenerativeModel('gemini-1.5-flash')
        GEMINI_AVAILABLE = True
        print("✅ Google Gemini API configured successfully")
    else:
        print("⚠️  No Google API key found")
except Exception as e:
    print(f"❌ Error configuring Gemini: {e}")

# Try to initialize OpenAI (fallback)
try:
    import openai
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai_client = openai.OpenAI(api_key=openai_api_key)
        OPENAI_AVAILABLE = True
        print("✅ OpenAI API configured successfully")
    else:
        print("⚠️  No OpenAI API key found")
except Exception as e:
    print(f"❌ Error configuring OpenAI: {e}")

# Determine which AI service to use
AI_AVAILABLE = GEMINI_AVAILABLE or OPENAI_AVAILABLE
if AI_AVAILABLE:
    if GEMINI_AVAILABLE:
        print("🤖 Using Google Gemini for AI-generated questions")
    else:
        print("🤖 Using OpenAI for AI-generated questions")
else:
    print("📚 No AI services available, using local templates only")

# Subject categories for question generation
SUBJECTS = {
    "Software Engineering": {
        "Algorithms": ["Sorting", "Searching", "Graph Algorithms", "Dynamic Programming", "Greedy Algorithms"],
        "Data Structures": ["Arrays", "Linked Lists", "Trees", "Graphs", "Hash Tables", "Stacks and Queues"],
        "System Design": ["Scalability", "Databases", "Caching", "Load Balancing", "Microservices", "API Design"],
        "Programming": ["Object-Oriented Design", "Design Patterns", "Testing", "Debugging", "Code Review"]
    },
    "Data Science": {
        "Machine Learning": ["Supervised Learning", "Unsupervised Learning", "Deep Learning", "Model Evaluation", "Feature Engineering"],
        "Statistics": ["Probability", "Hypothesis Testing", "Regression Analysis", "Time Series", "Experimental Design"],
        "Data Engineering": ["Data Pipelines", "ETL Processes", "Data Warehousing", "Big Data", "Data Quality"],
        "Analytics": ["Exploratory Data Analysis", "Data Visualization", "Business Intelligence", "A/B Testing"]
    },
    "Product Management": {
        "Strategy": ["Product Vision", "Market Analysis", "Competitive Analysis", "Go-to-Market Strategy"],
        "Development": ["Agile Methodologies", "User Stories", "Sprint Planning", "Product Roadmaps"],
        "Analytics": ["User Metrics", "Product Analytics", "A/B Testing", "Data-Driven Decisions"],
        "User Experience": ["User Research", "UX Design", "Customer Feedback", "Usability Testing"]
    }
}

# Local question templates for fallback when AI is not available
LOCAL_QUESTION_TEMPLATES = {
    "Software Engineering": {
        "Algorithms": [
            "Explain the {algorithm} algorithm and its time complexity.",
            "How would you implement {algorithm} in your preferred programming language?",
            "What are the advantages and disadvantages of {algorithm}?",
            "Compare {algorithm} with other similar algorithms.",
            "How would you optimize {algorithm} for better performance?"
        ],
        "Data Structures": [
            "Explain how {data_structure} works and when to use it.",
            "What are the time complexities for common operations on {data_structure}?",
            "How would you implement {data_structure} from scratch?",
            "What are the trade-offs of using {data_structure}?",
            "How does {data_structure} compare to other data structures?"
        ],
        "System Design": [
            "How would you design a system for {system_type}?",
            "What are the key considerations when building {system_type}?",
            "How would you scale {system_type} to handle millions of users?",
            "What technologies would you choose for {system_type} and why?",
            "How would you ensure reliability and availability for {system_type}?"
        ]
    },
    "Data Science": {
        "Machine Learning": [
            "Explain the concept of {ml_concept} and its applications.",
            "How would you approach a {ml_problem} problem?",
            "What are the key considerations when implementing {ml_concept}?",
            "How would you evaluate the performance of {ml_concept}?",
            "What are the challenges in applying {ml_concept} to real-world data?"
        ],
        "Statistics": [
            "Explain the statistical concept of {stat_concept}.",
            "How would you apply {stat_concept} in a data analysis project?",
            "What are the assumptions behind {stat_concept}?",
            "How would you interpret results involving {stat_concept}?",
            "What are the limitations of {stat_concept}?"
        ]
    }
}

# Fill-in words for local templates
TEMPLATE_FILLERS = {
    "algorithm": ["quicksort", "mergesort", "binary search", "Dijkstra's", "BFS", "DFS", "dynamic programming"],
    "data_structure": ["binary tree", "hash table", "linked list", "stack", "queue", "graph", "heap"],
    "system_type": ["social media platform", "e-commerce site", "video streaming service", "ride-sharing app", "messaging system"],
    "ml_concept": ["overfitting", "cross-validation", "feature selection", "ensemble methods", "neural networks"],
    "ml_problem": ["classification", "regression", "clustering", "recommendation", "time series forecasting"],
    "stat_concept": ["p-value", "confidence intervals", "correlation", "regression", "hypothesis testing"]
}

class QuestionRequest(BaseModel):
    subject: str = ""
    topic: str = ""
    chapter: str = ""

class ChatResponse(BaseModel):
    question: str
    answer: str
    subject: Optional[str] = ""
    topic: Optional[str] = ""
    chapter: Optional[str] = ""

class ResumeChatResponse(BaseModel):
    question: str
    answer: str
    resume_data: Optional[str] = ""

class ChatHistory(BaseModel):
    session_id: str
    responses: List[ChatResponse]
    timestamp: str

class UserRegister(BaseModel):
    email: str
    password: str
    name: str = ""

class UserLogin(BaseModel):
    email: str
    password: str

class ResumeUpload(BaseModel):
    user_id: str
    file_name: str
    file_size: int
    extracted_text: str = ""
    parsed_data: Optional[dict] = None

class ResumeQuestionRequest(BaseModel):
    resume_text: str
    question_type: str = "technical"

# In-memory storage for chat sessions (in production, use a database)
chat_sessions = {}
resume_chat_sessions = {}

# Authentication helper functions
def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_session_token(user_id: str) -> str:
    """Create a session token for a user"""
    token = secrets.token_urlsafe(32)
    user_sessions[token] = {
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat()
    }
    return token

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from session token"""
    token = credentials.credentials
    if token not in user_sessions:
        raise HTTPException(status_code=401, detail="Invalid session token")
    return user_sessions[token]["user_id"]

def initialize_azure_services():
    """Initialize Azure services on startup"""
    try:
        # Create database and containers
        azure_service.create_database_and_container()
        user_service.create_users_container()
        print("✅ Azure services initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing Azure services: {e}")

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF content"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_content: bytes) -> str:
    """Extract text from DOCX content"""
    try:
        doc = docx.Document(io.BytesIO(docx_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_file(file_content: bytes, file_type: str) -> str:
    """Extract text from uploaded file based on type"""
    if file_type == "application/pdf":
        return extract_text_from_pdf(file_content)
    elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        return extract_text_from_docx(file_content)
    elif file_type == "text/plain":
        return file_content.decode('utf-8', errors='ignore')
    else:
        return ""

async def generate_ai_question(subject: str = "", topic: str = "", chapter: str = "") -> str:
    """Generate a unique question using AI (Gemini preferred, OpenAI fallback)"""
    if not AI_AVAILABLE:
        return generate_local_question(subject, topic, chapter)
    
    # Build the prompt based on provided parameters
    if subject and topic and chapter:
        prompt = f"Generate a unique, challenging interview question for {subject} specifically about {topic} - {chapter}. The question should be suitable for a technical interview and encourage detailed discussion."
    elif subject and topic:
        prompt = f"Generate a unique, challenging interview question for {subject} specifically about {topic}. The question should be suitable for a technical interview and encourage detailed discussion."
    elif subject:
        prompt = f"Generate a unique, challenging interview question for {subject}. The question should be suitable for a technical interview and encourage detailed discussion."
    else:
        prompt = "Generate a unique, challenging technical interview question. The question should encourage detailed discussion and be suitable for a software engineering, data science, or product management interview."
    
    # Try Gemini first (preferred)
    if GEMINI_AVAILABLE and gemini_client:
        try:
            system_prompt = "You are an expert technical interviewer. Generate unique, thought-provoking questions that encourage candidates to think deeply and demonstrate their knowledge. Keep responses concise and focused."
            
            response = gemini_client.generate_content([
                system_prompt,
                prompt
            ])
            
            if response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Error generating Gemini question: {e}")
    
    # Fallback to OpenAI
    if OPENAI_AVAILABLE and openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert technical interviewer. Generate unique, thought-provoking questions that encourage candidates to think deeply and demonstrate their knowledge."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating OpenAI question: {e}")
    
    # Fallback to local templates
    return generate_local_question(subject, topic, chapter)

async def generate_resume_ai_question(resume_text: str) -> str:
    """Generate a unique question based on resume content using AI"""
    if not AI_AVAILABLE:
        return generate_resume_local_question(resume_text)
    
    # Truncate resume text if too long (AI models have token limits)
    if len(resume_text) > 2000:
        resume_text = resume_text[:2000] + "..."
    
    prompt = f"""Based on this resume content, generate a personalized interview question that relates to the candidate's experience, skills, or background:

Resume Content:
{resume_text}

Generate a specific, relevant interview question that would help assess the candidate's qualifications and experience. The question should be tailored to their background and encourage detailed discussion."""

    # Try Gemini first (preferred)
    if GEMINI_AVAILABLE and gemini_client:
        try:
            system_prompt = "You are an expert technical interviewer. Generate personalized, relevant questions based on a candidate's resume that encourage them to elaborate on their experience and demonstrate their skills. Keep responses concise and focused."
            
            response = gemini_client.generate_content([
                system_prompt,
                prompt
            ])
            
            if response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Error generating Gemini resume question: {e}")
    
    # Fallback to OpenAI
    if OPENAI_AVAILABLE and openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert technical interviewer. Generate personalized, relevant questions based on a candidate's resume that encourage them to elaborate on their experience and demonstrate their skills."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating OpenAI resume question: {e}")
    
    # Fallback to local templates
    return generate_resume_local_question(resume_text)

def generate_resume_local_question(resume_text: str) -> str:
    """Generate a question using local templates when AI is not available for resume-based questions"""
    
    # Extract some basic information from resume text
    resume_lower = resume_text.lower()
    
    # Determine focus areas based on resume content
    focus_areas = []
    
    if any(tech in resume_lower for tech in ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust']):
        focus_areas.append('programming')
    
    if any(tech in resume_lower for tech in ['machine learning', 'ai', 'data science', 'statistics', 'analytics']):
        focus_areas.append('data science')
    
    if any(tech in resume_lower for tech in ['product', 'management', 'agile', 'scrum', 'user experience']):
        focus_areas.append('product management')
    
    if any(tech in resume_lower for tech in ['system design', 'architecture', 'scalability', 'microservices']):
        focus_areas.append('system design')
    
    if any(tech in resume_lower for tech in ['database', 'sql', 'nosql', 'mongodb', 'postgresql']):
        focus_areas.append('databases')
    
    # If no specific focus areas found, use general questions
    if not focus_areas:
        focus_areas = ['general']
    
    # Generate question based on focus areas
    focus_area = random.choice(focus_areas)
    
    resume_questions = {
        'programming': [
            "Can you walk me through a challenging programming problem you've solved recently?",
            "What programming languages are you most comfortable with, and how did you learn them?",
            "Describe a time when you had to debug a complex issue. What was your approach?",
            "How do you stay updated with the latest programming trends and technologies?",
            "What's your preferred development environment and why?"
        ],
        'data science': [
            "Can you describe a data analysis project you've worked on? What were the key insights?",
            "How do you approach feature engineering in machine learning projects?",
            "What metrics do you use to evaluate model performance?",
            "Describe a time when you had to explain complex data findings to non-technical stakeholders.",
            "How do you handle missing or noisy data in your analysis?"
        ],
        'product management': [
            "Can you walk me through a product feature you've launched from conception to release?",
            "How do you prioritize features in a product roadmap?",
            "Describe a time when you had to make a difficult product decision with limited data.",
            "How do you gather and incorporate user feedback into product decisions?",
            "What's your approach to defining and measuring product success metrics?"
        ],
        'system design': [
            "Can you describe a system you've designed or worked on? What were the key challenges?",
            "How do you approach designing a scalable architecture?",
            "What factors do you consider when choosing between different technologies?",
            "Describe a time when you had to optimize system performance. What was your approach?",
            "How do you ensure system reliability and handle failure scenarios?"
        ],
        'databases': [
            "Can you describe a database design you've worked on? What were the key considerations?",
            "How do you approach database optimization and performance tuning?",
            "What's your experience with different types of databases (SQL vs NoSQL)?",
            "Describe a time when you had to migrate or scale a database system.",
            "How do you ensure data integrity and handle data quality issues?"
        ],
        'general': [
            "Can you tell me about a challenging project you've worked on recently?",
            "What are your strongest technical skills, and how did you develop them?",
            "Describe a time when you had to learn a new technology quickly.",
            "How do you approach problem-solving when faced with an unfamiliar challenge?",
            "What are your career goals and how does this role align with them?",
            "Can you walk me through your experience and how it relates to this position?",
            "What are some of your recent achievements that you're most proud of?",
            "How do you handle working under pressure or tight deadlines?",
            "Describe a time when you had to collaborate with a difficult team member.",
            "What motivates you in your work and how do you stay productive?"
        ]
    }
    
    return random.choice(resume_questions[focus_area])

def generate_local_question(subject: str = "", topic: str = "", chapter: str = "") -> str:
    """Generate a question using local templates when AI is not available"""
    
    # If no subject specified, pick a random one
    if not subject:
        subject = random.choice(list(SUBJECTS.keys()))
    
    # If no topic specified, pick a random one from the subject
    if not topic:
        topic = random.choice(list(SUBJECTS[subject].keys()))
    
    # If no chapter specified, pick a random one from the topic
    if not chapter:
        chapter = random.choice(SUBJECTS[subject][topic])
    
    # Try to use templates if available
    if subject in LOCAL_QUESTION_TEMPLATES and topic in LOCAL_QUESTION_TEMPLATES[subject]:
        template = random.choice(LOCAL_QUESTION_TEMPLATES[subject][topic])
        
        # Fill in template placeholders
        for placeholder, fillers in TEMPLATE_FILLERS.items():
            if f"{{{placeholder}}}" in template:
                template = template.replace(f"{{{placeholder}}}", random.choice(fillers))
        
        return template
    
    # Fallback to generic questions
    generic_questions = [
        f"Explain a challenging problem you've solved related to {topic} in {subject}.",
        f"What are the key concepts in {chapter} within {topic}?",
        f"How would you approach a {subject} problem involving {topic}?",
        f"What recent developments in {subject} have you found interesting?",
        f"Describe a project where you applied {topic} concepts.",
        f"What are the trade-offs involved in {chapter}?",
        f"How would you explain {topic} to someone with no technical background?",
        f"What resources would you recommend for learning more about {chapter}?"
    ]
    
    return random.choice(generic_questions)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    initialize_azure_services()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get-question")
async def get_question(request: QuestionRequest):
    """Generate a unique question based on the provided parameters"""
    question = await generate_ai_question(request.subject, request.topic, request.chapter)
    return {"message": question}

@app.post("/generate-resume-question")
async def generate_resume_question(resume: UploadFile = File(...), question_type: str = Form("technical")):
    """Generate a question based on uploaded resume content"""
    try:
        # Read file content
        file_content = await resume.read()
        
        # Extract text from file
        resume_text = extract_text_from_file(file_content, resume.content_type)
        
        if not resume_text:
            return {"success": False, "error": "Could not extract text from the uploaded file"}
        
        # Generate question based on resume content and question type
        if question_type == "behavioral":
            question = await generate_resume_behavioral_question(resume_text)
        else:
            question = await generate_resume_ai_question(resume_text)
        
        return {"success": True, "question": question, "question_type": question_type}
        
    except Exception as e:
        print(f"Error processing resume: {e}")
        return {"success": False, "error": "Error processing resume file"}

@app.post("/generate-resume-question-from-text")
async def generate_resume_question_from_text(request: ResumeQuestionRequest):
    """Generate a question based on resume text content"""
    try:
        if not request.resume_text:
            return {"success": False, "error": "Resume text is required"}
        
        # Generate question based on resume content and question type
        if request.question_type == "behavioral":
            question = await generate_resume_behavioral_question(request.resume_text)
        else:
            question = await generate_resume_ai_question(request.resume_text)
        
        return {"success": True, "question": question, "question_type": request.question_type}
        
    except Exception as e:
        print(f"Error generating question from resume text: {e}")
        return {"success": False, "error": "Error generating question"}

async def generate_resume_behavioral_question(resume_text: str) -> str:
    """Generate a behavioral question based on resume content using AI"""
    if not AI_AVAILABLE:
        return generate_resume_behavioral_local_question(resume_text)
    
    # Truncate resume text if too long (AI models have token limits)
    if len(resume_text) > 2000:
        resume_text = resume_text[:2000] + "..."
    
    prompt = f"""Based on this resume content, generate a personalized behavioral interview question that focuses on workplace dynamics, teamwork, communication, and how the candidate works with others. Focus on soft skills and interpersonal relationships rather than technical problem-solving:

Resume Content:
{resume_text}

Generate a specific behavioral interview question that would help assess the candidate's ability to work well with others, handle workplace relationships, communicate effectively, and contribute to a positive team environment. The question should encourage them to share specific examples of how they interact with colleagues, handle conflicts, or contribute to team success."""

    # Try Gemini first (preferred)
    if GEMINI_AVAILABLE and gemini_client:
        try:
            system_prompt = "You are an expert behavioral interviewer. Generate personalized, relevant behavioral questions based on a candidate's resume that focus on workplace dynamics, teamwork, communication, and interpersonal relationships. Focus on how candidates work with others, handle conflicts, and contribute to team success. Keep responses concise and focused."
            
            response = gemini_client.generate_content([
                system_prompt,
                prompt
            ])
            
            if response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Error generating Gemini behavioral question: {e}")
    
    # Fallback to OpenAI
    if OPENAI_AVAILABLE and openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert behavioral interviewer. Generate personalized, relevant behavioral questions based on a candidate's resume that focus on workplace dynamics, teamwork, communication, and interpersonal relationships. Focus on how candidates work with others, handle conflicts, and contribute to team success."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating OpenAI behavioral question: {e}")
    
    # Fallback to local templates
    return generate_resume_behavioral_local_question(resume_text)

def generate_resume_behavioral_local_question(resume_text: str) -> str:
    """Generate a behavioral question using local templates when AI is not available for resume-based questions"""
    
    # Extract some basic information from resume text
    resume_lower = resume_text.lower()
    
    # Determine focus areas based on resume content
    focus_areas = []
    
    if any(word in resume_lower for word in ['lead', 'manage', 'supervise', 'team', 'direct', 'mentor']):
        focus_areas.append('leadership')
    
    if any(word in resume_lower for word in ['team', 'collaborate', 'coordinate', 'work with', 'group']):
        focus_areas.append('teamwork')
    
    if any(word in resume_lower for word in ['communicate', 'present', 'explain', 'feedback', 'discuss']):
        focus_areas.append('communication')
    
    if any(word in resume_lower for word in ['relationship', 'colleague', 'coworker', 'peer', 'partner']):
        focus_areas.append('workplace_relationships')
    
    if any(word in resume_lower for word in ['conflict', 'disagree', 'resolve', 'dispute', 'tension']):
        focus_areas.append('handling_conflicts')
    
    if any(word in resume_lower for word in ['collaborate', 'coordinate', 'work together', 'joint', 'shared']):
        focus_areas.append('collaboration')
    
    if any(word in resume_lower for word in ['adapt', 'change', 'flexible', 'adjust', 'different']):
        focus_areas.append('adaptability')
    
    # If no specific focus areas found, use general behavioral questions
    if not focus_areas:
        focus_areas = ['general']
    
    # Generate question based on focus areas
    focus_area = random.choice(focus_areas)
    
    behavioral_questions = {
        'leadership': [
            "Can you tell me about a time when you had to lead a team through a challenging situation? How did you keep everyone motivated?",
            "Describe a situation where you had to help a struggling team member improve their performance. What was your approach?",
            "Tell me about a time when you had to make a decision that wasn't popular with your team. How did you handle the pushback?",
            "Can you share an example of how you've helped create a positive team culture?",
            "Describe a time when you had to lead by example to inspire your colleagues."
        ],
        'teamwork': [
            "Tell me about a time when you had to work with someone who had a very different working style than yours. How did you adapt?",
            "Can you describe a situation where you had to resolve a conflict between two team members?",
            "Tell me about a time when you had to collaborate with people from different departments or backgrounds.",
            "Describe a situation where you had to step up to help a colleague who was overwhelmed.",
            "Can you share an example of how you've contributed to building a strong team dynamic?"
        ],
        'communication': [
            "Tell me about a time when you had to explain something complex to someone who didn't understand it initially.",
            "Describe a situation where you had to give constructive feedback to a colleague. How did you approach it?",
            "Can you share an example of how you've handled a difficult conversation with a coworker?",
            "Tell me about a time when you had to present an idea to a group that was initially skeptical.",
            "Describe a situation where you had to mediate between two people who disagreed."
        ],
        'workplace_relationships': [
            "Tell me about a time when you had to work with someone you didn't get along with personally. How did you handle it?",
            "Can you describe a situation where you had to build trust with a new team member?",
            "Tell me about a time when you had to work with someone who was much more experienced than you.",
            "Describe a situation where you had to work with someone who was much less experienced than you.",
            "Can you share an example of how you've helped a new colleague feel welcome on the team?"
        ],
        'handling_conflicts': [
            "Tell me about a time when you disagreed with your manager or supervisor. How did you handle it?",
            "Describe a situation where you had to work with someone who had a very different opinion than yours.",
            "Can you share an example of a time when you had to address a conflict between team members?",
            "Tell me about a time when you had to stand up for what you believed was right at work.",
            "Describe a situation where you had to find common ground with someone who had opposing views."
        ],
        'collaboration': [
            "Tell me about a time when you had to work with a team to achieve a common goal. What was your role?",
            "Can you describe a situation where you had to coordinate with multiple people to get something done?",
            "Tell me about a time when you had to share credit for a successful project with others.",
            "Describe a situation where you had to help someone else succeed even though it wasn't your responsibility.",
            "Can you share an example of how you've contributed to a team's success beyond your individual tasks?"
        ],
        'adaptability': [
            "Tell me about a time when you had to adapt to a major change in your work environment.",
            "Describe a situation where you had to work with a new team or group of people.",
            "Can you share an example of how you've handled working with people from different cultures or backgrounds?",
            "Tell me about a time when you had to adjust your communication style to work with different people.",
            "Describe a situation where you had to learn to work with someone whose approach was very different from yours."
        ],
        'general': [
            "Tell me about a time when you had to work with someone whose personality was very different from yours.",
            "Describe a situation where you had to help create a positive work environment.",
            "Can you share an example of how you've contributed to team morale during a difficult time?",
            "Tell me about a time when you had to work with someone who was having a hard time personally.",
            "Describe a situation where you had to be patient with a colleague who was learning something new.",
            "Can you share an example of how you've helped someone feel included in a group setting?",
            "Tell me about a time when you had to work with someone who was resistant to change.",
            "Describe a situation where you had to help bridge a gap between different groups or departments.",
            "Can you share an example of how you've helped resolve tension in a work environment?",
            "Tell me about a time when you had to work with someone who was much older or younger than you."
        ]
    }
    
    return random.choice(behavioral_questions[focus_area])

async def generate_resume_follow_up_question(response: ResumeChatResponse) -> str:
    """Generate a contextual follow-up question for resume-based responses"""
    
    if not AI_AVAILABLE:
        # Fallback to local follow-up questions
        follow_up_questions = [
            "Can you provide more specific details about that experience?",
            "What were the key challenges you faced in that situation?",
            "How did you measure the success of that project?",
            "What would you do differently if you had to do it again?",
            "Can you walk me through your decision-making process?",
            "What skills did you develop through that experience?",
            "How did you handle any setbacks or failures?",
            "What feedback did you receive from stakeholders?",
            "How did this experience prepare you for future challenges?",
            "What resources or support did you rely on?",
            "Can you give me a specific example of a problem you solved?",
            "How did you prioritize your responsibilities?",
            "What was the most valuable lesson you learned?",
            "How did you collaborate with others on this?",
            "What impact did your work have on the organization?",
            "How did you handle any resistance or pushback?",
            "What would you say was your biggest contribution in that situation?",
            "How did you ensure everyone was on the same page?",
            "What was the outcome, and how did you measure success?",
            "How did you handle any unexpected obstacles?"
        ]
        return random.choice(follow_up_questions)
    
    prompt = f"Based on this interview response about their resume experience: '{response.answer}', generate a thoughtful follow-up question that digs deeper into their interpersonal skills and how they work with others. The question should encourage the candidate to provide more specific details about their relationships with colleagues, communication style, or team dynamics."
    
    # Try Gemini first (preferred)
    if GEMINI_AVAILABLE and gemini_client:
        try:
            system_prompt = "You are an expert behavioral interviewer. Generate thoughtful follow-up questions that encourage candidates to elaborate on their interpersonal skills, communication style, and how they work with others. Focus on team dynamics, relationships with colleagues, and soft skills. Keep responses concise."
            
            ai_response = gemini_client.generate_content([
                system_prompt,
                prompt
            ])
            
            if ai_response.text:
                return ai_response.text.strip()
        except Exception as e:
            print(f"Error generating Gemini resume follow-up: {e}")
    
    # Fallback to OpenAI
    if OPENAI_AVAILABLE and openai_client:
        try:
            ai_response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert behavioral interviewer. Generate thoughtful follow-up questions that encourage candidates to elaborate on their interpersonal skills, communication style, and how they work with others. Focus on team dynamics, relationships with colleagues, and soft skills."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            return ai_response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating OpenAI resume follow-up: {e}")
    
    # Fallback to local follow-up questions
    follow_up_questions = [
        "Can you tell me more about how the other person reacted to your approach?",
        "What did you learn about working with people from that experience?",
        "How did you build trust with the people involved?",
        "What would you do differently if you faced a similar situation?",
        "How did you ensure everyone felt heard and valued?",
        "What was the impact on the team dynamic after this situation?",
        "How did you handle any emotional aspects of the situation?",
        "What feedback did you receive from others involved?",
        "How did this experience change how you work with others?",
        "What communication strategies worked best in that situation?",
        "How did you maintain positive relationships throughout?",
        "What was the most challenging part of working with others in this situation?",
        "How did you balance different perspectives and needs?",
        "What role did empathy play in how you handled this?",
        "How did you ensure the team stayed motivated and engaged?",
        "What did you learn about conflict resolution from this experience?",
        "How did you adapt your communication style for different people?",
        "What was the long-term impact on your working relationships?",
        "How did you help others feel comfortable and included?",
        "What would you say was your biggest interpersonal challenge in this situation?"
    ]
    
    return random.choice(follow_up_questions)

@app.post("/submit-response")
async def submit_response(response: ChatResponse):
    """Handle user responses and potentially generate follow-up questions"""
    
    # Store the response (in a real app, save to database)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    chat_sessions[session_id].append(response)
    
    # Generate a follow-up question based on context
    follow_up = await generate_follow_up_question(response)
    
    return {
        "status": "success",
        "follow_up": follow_up,
        "session_id": session_id
    }

@app.post("/submit-resume-response")
async def submit_resume_response(response: ResumeChatResponse):
    """Handle user responses for resume-based questions and generate follow-up questions"""
    
    # Store the response (in a real app, save to database)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if session_id not in resume_chat_sessions:
        resume_chat_sessions[session_id] = []
    
    resume_chat_sessions[session_id].append(response)
    
    # Generate a follow-up question based on context
    follow_up = await generate_resume_follow_up_question(response)
    
    return {
        "status": "success",
        "follow_up": follow_up,
        "session_id": session_id
    }

async def generate_follow_up_question(response: ChatResponse) -> str:
    """Generate a contextual follow-up question based on the user's response"""
    
    if not AI_AVAILABLE:
        # Fallback to local follow-up questions
        follow_up_questions = [
            "Can you elaborate on that?",
            "What specific examples can you provide?",
            "How would you handle a challenging situation related to this?",
            "What would you do differently next time?",
            "How does this relate to your previous experience?",
            "What resources would you use to improve in this area?",
            "How would you explain this to someone with no technical background?",
            "What are the trade-offs involved in this approach?",
            "How would you test or validate this solution?",
            "What are the potential risks or limitations?",
            "Can you walk me through a specific example?",
            "What metrics would you use to measure success?",
            "How would you scale this solution?",
            "What alternatives did you consider?",
            "How would you handle edge cases?"
        ]
        return random.choice(follow_up_questions)
    
    prompt = f"Based on this interview response: '{response.answer}', generate a thoughtful follow-up question that digs deeper into the topic. The question should encourage the candidate to elaborate or provide more specific examples."
    
    # Try Gemini first (preferred)
    if GEMINI_AVAILABLE and gemini_client:
        try:
            system_prompt = "You are an expert technical interviewer. Generate thoughtful follow-up questions that encourage candidates to elaborate and provide more specific examples. Keep responses concise."
            
            ai_response = gemini_client.generate_content([
                system_prompt,
                prompt
            ])
            
            if ai_response.text:
                return ai_response.text.strip()
        except Exception as e:
            print(f"Error generating Gemini follow-up: {e}")
    
    # Fallback to OpenAI
    if OPENAI_AVAILABLE and openai_client:
        try:
            ai_response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert technical interviewer. Generate thoughtful follow-up questions that encourage candidates to elaborate and provide more specific examples."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            return ai_response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating OpenAI follow-up: {e}")
    
    # Fallback to local follow-up questions
    follow_up_questions = [
        "Can you elaborate on that?",
        "What specific examples can you provide?",
        "How would you handle a challenging situation related to this?",
        "What would you do differently next time?",
        "How does this relate to your previous experience?",
        "What resources would you use to improve in this area?",
        "How would you explain this to someone with no technical background?",
        "What are the trade-offs involved in this approach?",
        "How would you test or validate this solution?",
        "What are the potential risks or limitations?",
        "Can you walk me through a specific example?",
        "What metrics would you use to measure success?",
        "How would you scale this solution?",
        "What alternatives did you consider?",
        "How would you handle edge cases?"
    ]
    
    return random.choice(follow_up_questions)

@app.get("/get-subjects")
async def get_subjects():
    return {"subjects": list(SUBJECTS.keys())}

@app.get("/get-topics/{subject}")
async def get_topics(subject: str):
    if subject in SUBJECTS:
        return {"topics": list(SUBJECTS[subject].keys())}
    return {"topics": []}

@app.get("/get-chapters/{subject}/{topic}")
async def get_chapters(subject: str, topic: str):
    if subject in SUBJECTS and topic in SUBJECTS[subject]:
        return {"chapters": SUBJECTS[subject][topic]}
    return {"chapters": []}

@app.get("/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    """Retrieve chat history for a session"""
    if session_id in chat_sessions:
        return {"history": chat_sessions[session_id]}
    return {"history": []}

@app.get("/resume-chat-history/{session_id}")
async def get_resume_chat_history(session_id: str):
    """Retrieve resume chat history for a session"""
    if session_id in resume_chat_sessions:
        return {"history": resume_chat_sessions[session_id]}
    return {"history": []}

# User Management Endpoints
@app.post("/register")
async def register_user(user_data: UserRegister):
    """Register a new user"""
    try:
        # Hash the password
        password_hash = hash_password(user_data.password)
        
        # Create user
        user_id = user_service.create_user(
            email=user_data.email,
            password_hash=password_hash,
            name=user_data.name
        )
        
        # Create session token
        token = create_session_token(user_id)
        
        return {
            "success": True,
            "message": "User registered successfully",
            "user_id": user_id,
            "token": token
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
async def login_user(user_data: UserLogin):
    """Login user"""
    try:
        # Hash the password
        password_hash = hash_password(user_data.password)
        
        # Authenticate user
        user = user_service.authenticate_user(user_data.email, password_hash)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Create session token
        token = create_session_token(user["id"])
        
        return {
            "success": True,
            "message": "Login successful",
            "user_id": user["id"],
            "user_name": user.get("name", ""),
            "token": token
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/logout")
async def logout_user(current_user: str = Depends(get_current_user)):
    """Logout user"""
    # In a real app, you'd invalidate the token
    # For now, we'll just return success
    return {"success": True, "message": "Logout successful"}

# Resume Management Endpoints
@app.post("/upload-resume")
async def upload_resume(
    resume: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    """Upload a resume file"""
    try:
        # Read file content
        file_content = await resume.read()
        
        # Upload file to Azure Blob Storage
        file_url = azure_service.upload_resume_file(file_content, resume.filename)
        
        # Extract text from file
        extracted_text = extract_text_from_file(file_content, resume.content_type)
        
        # Save resume metadata to Cosmos DB
        resume_id = azure_service.save_resume_metadata(
            user_id=current_user,
            file_url=file_url,
            file_name=resume.filename,
            file_size=len(file_content),
            extracted_text=extracted_text
        )
        
        return {
            "success": True,
            "message": "Resume uploaded successfully",
            "resume_id": resume_id,
            "file_url": file_url
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/resumes")
async def get_user_resumes(current_user: str = Depends(get_current_user)):
    """Get all resumes for the current user"""
    try:
        resumes = azure_service.get_user_resumes(current_user)
        return {
            "success": True,
            "resumes": resumes
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/resumes/{resume_id}")
async def get_resume(
    resume_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get a specific resume"""
    try:
        resume = azure_service.get_resume_by_id(resume_id, current_user)
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        return {
            "success": True,
            "resume": resume
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/resumes/{resume_id}")
async def delete_resume(
    resume_id: str,
    current_user: str = Depends(get_current_user)
):
    """Delete a resume"""
    try:
        success = azure_service.delete_resume(resume_id, current_user)
        if not success:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        return {
            "success": True,
            "message": "Resume deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/search-resumes")
async def search_resumes(
    search_term: str,
    current_user: str = Depends(get_current_user)
):
    """Search resumes by content"""
    try:
        resumes = azure_service.search_resumes(current_user, search_term)
        return {
            "success": True,
            "resumes": resumes
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint to verify AI availability"""
    ai_service = "none"
    if GEMINI_AVAILABLE:
        ai_service = "gemini"
    elif OPENAI_AVAILABLE:
        ai_service = "openai"
    
    return {
        "status": "healthy",
        "ai_available": AI_AVAILABLE,
        "ai_service": ai_service,
        "gemini_available": GEMINI_AVAILABLE,
        "openai_available": OPENAI_AVAILABLE,
        "subjects": list(SUBJECTS.keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 