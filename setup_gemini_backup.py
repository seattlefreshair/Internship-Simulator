#!/usr/bin/env python3
"""
Setup script for Google Gemini API configuration
"""

import os
import sys

def setup_gemini():
    print("🚀 Setting up Google Gemini API for Interview Simulator")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = "backend/.env"
    if not os.path.exists(env_file):
        print("📝 Creating .env file...")
        with open(env_file, "w") as f:
            f.write("# Google Gemini API Key\n")
            f.write("GOOGLE_API_KEY=your_gemini_api_key_here\n")
            f.write("\n# Optional: OpenAI API Key (alternative)\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
    
    # Get API key from user
    print("\n🔑 Please enter your Google Gemini API key:")
    print("   Get it from: https://makersuite.google.com/app/apikey")
    print("   (The API key should start with 'AIza' and be about 39 characters long)")
    
    api_key = input("\nEnter your Gemini API key: ").strip()
    
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ No valid API key provided. Please get your API key from Google AI Studio.")
        return False
    
    if not api_key.startswith("AIza"):
        print("⚠️  Warning: API key doesn't start with 'AIza'. Please make sure you're using the correct API key.")
        print("   The number you provided earlier (736400257435) looks like a project ID, not an API key.")
        print("   You need the actual API key from Google AI Studio.")
    
    # Update .env file
    print("\n📝 Updating .env file...")
    with open(env_file, "r") as f:
        content = f.read()
    
    content = content.replace("GOOGLE_API_KEY=your_gemini_api_key_here", f"GOOGLE_API_KEY={api_key}")
    
    with open(env_file, "w") as f:
        f.write(content)
    
    print("✅ API key saved to .env file")
    
    # Test the API key
    print("\n🧪 Testing API connection...")
    try:
        # Install required packages if not already installed
        try:
            import google.generativeai
        except ImportError:
            print("📦 Installing required packages...")
            os.system("pip3 install google-generativeai python-dotenv")
        
        # Test the connection
        from dotenv import load_dotenv
        import google.generativeai as genai
        
        # Load the API key from the .env file
        # TODO: does it use the API key from the .env file?
        load_dotenv(env_file)

        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Test with a simple prompt
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'Hello, Gemini is working!'")
        
        if response.text:
            print("✅ Gemini API connection successful!")
            print(f"   Response: {response.text}")
            return True
        else:
            print("❌ No response from Gemini API")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Gemini API: {e}")
        print("\nPossible issues:")
        print("1. Invalid API key - make sure you copied it correctly")
        print("2. Network connection issues")
        print("3. API quota exceeded - check your Google AI Studio dashboard")
        return False

def main():
    if setup_gemini():
        print("\n🎉 Setup complete!")
        print("\nTo run the application:")
        print("1. cd backend")
        print("2. python3 app.py")
        print("3. Open http://localhost:8000 in your browser")
        print("\nThe app will now generate unique AI-powered questions!")
    else:
        print("\n❌ Setup failed. Please check your API key and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 