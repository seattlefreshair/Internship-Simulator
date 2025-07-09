# Internship Simulator

An AI-powered interview practice application that generates unique, dynamic questions for technical interviews. Perfect for preparing for software engineering, data science, and product management internships.

## Features

- **AI-Generated Questions**: Uses Google Gemini (preferred) or OpenAI's GPT model to create unique, contextual interview questions
- **Multiple Subjects**: Covers Software Engineering, Data Science, and Product Management
- **Dynamic Follow-ups**: Generates intelligent follow-up questions based on your responses
- **Fallback System**: Works offline with local question templates when AI is unavailable
- **Interactive Interface**: Clean, modern web interface for practice sessions

## Setup

### Prerequisites
- Python 3.8+
- Google Gemini API key (recommended, free tier available) or OpenAI API key (optional, for AI-generated questions)

### Installation

1. Install the required dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. **Optional**: Set up AI API for AI-generated questions:
   - **Recommended**: Get a Google Gemini API key (free tier available) from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **Alternative**: Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create a `.env` file in the `backend` directory:
   ```bash
   cp env_example.txt .env
   ```
   - Edit `.env` and add your API key(s):
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   # or
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Application

1. Start the FastAPI server:
```bash
cd backend
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. Start practicing with dynamic interview questions!

## How It Works

### AI-Powered Questions
When an AI API key is configured, the app generates unique questions using Google Gemini (preferred) or OpenAI GPT. Questions are tailored to the selected subject, topic, and chapter. Gemini is recommended as it offers a generous free tier.

### Fallback System
If AI is unavailable, the app uses intelligent local templates with randomized components to create varied questions.

### Follow-up Questions
The system generates contextual follow-up questions based on your responses, either using AI or predefined templates.

## API Endpoints

- `GET /` - Main application interface
- `POST /get-question` - Generate a new question
- `POST /submit-response` - Submit your answer and get follow-up
- `GET /get-subjects` - Get available subjects
- `GET /get-topics/{subject}` - Get topics for a subject
- `GET /get-chapters/{subject}/{topic}` - Get chapters for a topic
- `GET /health` - Check system status and AI availability

## API Documentation

FastAPI provides automatic API documentation at:
```
http://localhost:8000/docs
```

## Subjects Covered

### Software Engineering
- **Algorithms**: Sorting, Searching, Graph Algorithms, Dynamic Programming
- **Data Structures**: Arrays, Linked Lists, Trees, Graphs, Hash Tables
- **System Design**: Scalability, Databases, Caching, Load Balancing
- **Programming**: OOP, Design Patterns, Testing, Debugging

### Data Science
- **Machine Learning**: Supervised/Unsupervised Learning, Deep Learning, Model Evaluation
- **Statistics**: Probability, Hypothesis Testing, Regression Analysis
- **Data Engineering**: Data Pipelines, ETL, Data Warehousing
- **Analytics**: EDA, Data Visualization, Business Intelligence

### Product Management
- **Strategy**: Product Vision, Market Analysis, Competitive Analysis
- **Development**: Agile, User Stories, Sprint Planning, Roadmaps
- **Analytics**: User Metrics, Product Analytics, A/B Testing
- **User Experience**: User Research, UX Design, Customer Feedback

## Contributing

Feel free to contribute by adding new question templates, improving the AI prompts, or enhancing the user interface! 