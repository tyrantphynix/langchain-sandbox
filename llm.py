from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    api_key=API_KEY,
    model = "gemini-2.0-flash-lite-001",
    temperature= 1.9,
    max_tokens=400)
    
response = llm.stream("Write a storry about AI taking over M0di Govt.")

for chunk in response:
    print(chunk.content, end="", flush=True)