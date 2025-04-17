import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults

model = ChatGoogleGenerativeAI(
    api_key=API_KEY,
    model="gemini-2.0-flash-lite-001",
    max_tokens=300,
    temperature=0.3
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are friendly assistant max."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

search = TavilySearchResults()

tools = [search]

agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools
)

response = agentExecutor.invoke({
    "input" : "What is the weather in Zirakhpur today?"
})

print(response)