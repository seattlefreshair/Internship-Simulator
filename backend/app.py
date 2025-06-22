from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import random

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Data structure for questions
questions_data = {
    "Software Engineering": {
        "Algorithms": {
            "Sorting": [
                "Explain the difference between bubble sort and quicksort.",
                "What is the time complexity of merge sort?",
                "How would you implement a binary search algorithm?"
            ],
            "Data Structures": [
                "What is the difference between a stack and a queue?",
                "Explain how a hash table works.",
                "What are the advantages of using a binary tree?"
            ]
        },
        "System Design": {
            "Scalability": [
                "How would you design a URL shortening service?",
                "Explain the concept of load balancing.",
                "What is horizontal vs vertical scaling?"
            ],
            "Databases": [
                "What is the difference between SQL and NoSQL?",
                "Explain ACID properties in databases.",
                "How would you handle database sharding?"
            ]
        }
    },
    "Data Science": {
        "Machine Learning": {
            "Supervised Learning": [
                "Explain the difference between classification and regression.",
                "What is overfitting and how do you prevent it?",
                "How does cross-validation work?"
            ],
            "Unsupervised Learning": [
                "What is clustering and when would you use it?",
                "Explain the k-means algorithm.",
                "What is dimensionality reduction?"
            ]
        },
        "Statistics": {
            "Probability": [
                "What is the difference between mean, median, and mode?",
                "Explain the concept of standard deviation.",
                "What is a p-value?"
            ],
            "Hypothesis Testing": [
                "What is a null hypothesis?",
                "Explain Type I and Type II errors.",
                "What is statistical significance?"
            ]
        }
    }
}

class QuestionRequest(BaseModel):
    subject: str = ""
    topic: str = ""
    chapter: str = ""

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get-question")
async def get_question(request: QuestionRequest):
    # If no parameters provided, return a random question from all categories
    if not request.subject and not request.topic and not request.chapter:
        all_questions = []
        for subject in questions_data.values():
            for topic in subject.values():
                for chapter in topic.values():
                    all_questions.extend(chapter)
        return {"message": random.choice(all_questions)}
    
    # Filter questions based on provided parameters
    available_questions = []
    
    for subject_name, subject_data in questions_data.items():
        if not request.subject or subject_name == request.subject:
            for topic_name, topic_data in subject_data.items():
                if not request.topic or topic_name == request.topic:
                    for chapter_name, questions in topic_data.items():
                        if not request.chapter or chapter_name == request.chapter:
                            available_questions.extend(questions)
    
    if available_questions:
        return {"message": random.choice(available_questions)}
    else:
        return {"message": "No questions found for the selected criteria."}

@app.get("/get-subjects")
async def get_subjects():
    return {"subjects": list(questions_data.keys())}

@app.get("/get-topics/{subject}")
async def get_topics(subject: str):
    if subject in questions_data:
        return {"topics": list(questions_data[subject].keys())}
    return {"topics": []}

@app.get("/get-chapters/{subject}/{topic}")
async def get_chapters(subject: str, topic: str):
    if subject in questions_data and topic in questions_data[subject]:
        return {"chapters": list(questions_data[subject][topic].keys())}
    return {"chapters": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 