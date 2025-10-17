import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Configure the API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file")
    exit(1)

# Try using the requests library to call the API directly
print("Available Gemini Models:\n" + "="*50)
try:
    import requests
    
    # Call the ListModels API
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        models = data.get('models', [])
        
        print(f"\nTotal models available: {len(models)}\n")
        
        for model in models:
            print(f"Model: {model.get('name', 'N/A')}")
            print(f"  Display Name: {model.get('displayName', 'N/A')}")
            print(f"  Description: {model.get('description', 'N/A')}")
            print(f"  Input Token Limit: {model.get('inputTokenLimit', 'N/A')}")
            print(f"  Output Token Limit: {model.get('outputTokenLimit', 'N/A')}")
            print(f"  Supported Generation Config: {model.get('supportedGenerationConfig', {})}")
            print()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Error listing models: {e}")