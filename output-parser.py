import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser, JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Instatiate template
llm = ChatGoogleGenerativeAI(
    api_key = API_KEY,
    model = "gemini-2.0-flash-lite-001",
    temperature = 0.2,
    max_tokens = 400
)
def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Tell me a joke about the following input"),
            ("human", "{input}")
        ]
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser

    return chain.invoke({
        "input": "Modi"
    })

def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate a list of 10 synonyms of the following word, Seprate all the synonyms by a comma"),
            ("human", "{input}")
        ]
    )

    parser = CommaSeparatedListOutputParser()

    chain = prompt | llm | parser

    return chain.invoke({
        "input" : "AI"
    })

def call_JSON_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an AI that extracts structured information from natural language. "
         "Use the following format instructions strictly to return a JSON object:\n{format_instructions}"),
        ("human", "{phrase}")
    ])

    class Person(BaseModel):
        recipie: str =  Field(description="the name of the recipie")
        ingridient: list = Field(description="the ingridients of the recipie")

    parser = JsonOutputParser(pydantic_object=Person)

    chain = prompt | llm | parser

    return chain.invoke({
        "phrase" : "The ingridients for a margarita pizza are tomato, cheese, basil, onion",
        "format_instructions" : parser.get_format_instructions()
    })



print(call_JSON_output_parser())