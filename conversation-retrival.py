import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval import create_retrieval_chain


def get_document_from_loader(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitdocs = splitter.split_documents(docs)
    return splitdocs

def create_db(docs):
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        # google_api_key=API_KEY
    )
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = ChatGoogleGenerativeAI(
    api_key=API_KEY,
    model="gemini-2.0-flash-lite-001",
    temperature=0.2,
    max_tokens=200
)

    prompt = ChatPromptTemplate.from_template("""
        Answer the user's question:
        Context : {context}
        Question : {input}
    """)


    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 2})

    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain

docs = (get_document_from_loader('https://python.langchain.com/v0.1/docs/expression_language/'))
vectorStore = create_db(docs)
chain = create_chain(vectorStore)


response =  chain.invoke({
    "input": "What is LCEL?"
})

print(response["context"])