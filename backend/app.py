from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import random
from typing import List, Optional
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="./frontend")

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

class ChatHistory(BaseModel):
    session_id: str
    responses: List[ChatResponse]
    timestamp: str

# In-memory storage for chat sessions (in production, use a database)
chat_sessions = {}

async def generate_ai_question(subject: str = "", topic: str = "", chapter: str = "") -> str:
    """Generate a unique question using AI (Gemini preferred, OpenAI fallback)"""
    # TODO: how AI_AVAILABLE is set
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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get-question")
# TODO: what is async mean?
async def get_question(request: QuestionRequest):
    """Generate a unique question based on the provided parameters"""
    # TODO: what is await mean?
    question = await generate_ai_question(request.subject, request.topic, request.chapter)
    return {"message": question}

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