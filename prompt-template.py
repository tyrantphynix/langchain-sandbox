from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate 
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Instatiate template

llm = ChatGoogleGenerativeAI(
    api_key = API_KEY,
    model = "gemini-2.0-flash-lite-001",
    temperature = 1,
    max_tokens = 400
    )

# Prompt Templates


# Template prompt
# prompt = ChatPromptTemplate.from_template("Tell me a joke about a {subject}")

# Message prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of 10 synonyms of the following word, Seprate all the synonyms by a comma"),
        ("human", "{input}")
    ]
)

# Create LLM chain

chain = prompt | llm

response = chain.invoke({"input": "Happy"})
print(response.content)