#!/usr/bin/env python3
"""
Test script to verify AI integration is working
"""

import os
import sys
from dotenv import load_dotenv

def test_ai_integration():
    print("🧪 Testing AI Integration for Interview Simulator")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv("backend/.env")
    
    # Check API keys
    gemini_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"🔑 Gemini API Key: {'✅ Found' if gemini_key and gemini_key != 'your_gemini_api_key_here' else '❌ Not found'}")
    print(f"🔑 OpenAI API Key: {'✅ Found' if openai_key and openai_key != 'your_openai_api_key_here' else '❌ Not found'}")
    
    # Test Gemini
    if gemini_key and gemini_key != 'your_gemini_api_key_here':
        print("\n🤖 Testing Google Gemini...")
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Test question generation
            prompt = "Generate a unique interview question about algorithms for software engineering."
            response = model.generate_content(prompt)
            
            if response.text:
                print("✅ Gemini working!")
                print(f"   Sample question: {response.text[:100]}...")
                return True
            else:
                print("❌ No response from Gemini")
                
        except Exception as e:
            print(f"❌ Gemini error: {e}")
    
    # Test OpenAI as fallback
    elif openai_key and openai_key != 'your_openai_api_key_here':
        print("\n🤖 Testing OpenAI...")
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Generate a unique interview question about algorithms."}
                ],
                max_tokens=100
            )
            
            if response.choices[0].message.content:
                print("✅ OpenAI working!")
                print(f"   Sample question: {response.choices[0].message.content[:100]}...")
                return True
            else:
                print("❌ No response from OpenAI")
                
        except Exception as e:
            print(f"❌ OpenAI error: {e}")
    
    print("\n❌ No working AI service found")
    print("Please set up your API keys in backend/.env")
    return False

def test_local_fallback():
    print("\n📚 Testing local question generation...")
    
    # Import the local question generation function
    sys.path.append('backend')
    try:
        from app import generate_local_question
        
        # Test local question generation
        question = generate_local_question("Software Engineering", "Algorithms", "Sorting")
        print(f"✅ Local question generated: {question[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Local generation error: {e}")
        return False

def main():
    print("🚀 Interview Simulator AI Test")
    print("=" * 30)
    
    # Test AI services
    ai_working = test_ai_integration()
    
    # Test local fallback
    local_working = test_local_fallback()
    
    print("\n" + "=" * 50)
    if ai_working:
        print("🎉 AI integration is working! Your app will generate unique AI-powered questions.")
    elif local_working:
        print("📚 Local templates are working. The app will work without AI, but questions will be from templates.")
        print("   To enable AI, set up your API keys in backend/.env")
    else:
        print("❌ Neither AI nor local generation is working. Please check your setup.")
    
    print("\nTo run the application:")
    print("1. cd backend")
    print("2. python3 app.py")
    print("3. Open http://localhost:8000")

if __name__ == "__main__":
    main() 