import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME")

if not OPEN_AI_KEY:
    raise ValueError("OPENAI_API_KEY not set!")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set!")

if not INDEX_NAME:
    raise ValueError("INDEX_NAME not set!")

embeddings = OpenAIEmbeddings()
vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
Answer the following question only from the context:
{context}
Question:
{question}
Provide a detailed answer.
"""
)

llm = ChatOpenAI(api_key=OPEN_AI_KEY, model="gpt-4o")


def format_document(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_without_lcel(query):
    docs = retriever.invoke(query)
    context = format_document(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    raw_response = llm.invoke(messages)
    return raw_response.content


def rag_with_lcel():
    return (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_document
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )


def main():

    query = "What is pinecone in machine learning?"
    # print("-" * 60)
    # print("IMPLEMENTATION 0 - RAW LLM invokation (NO RAG)")
    # print("-" * 60)
    # raw_response = llm.invoke([HumanMessage(query)])
    # print("Answer: \n")
    # print(raw_response.content)

    # print("-" * 60)
    # print("IMPLEMENTATION 1 - RAW LLM invokation (WITH RAG, WITHOUT LCEL)")
    # print("-" * 60)
    # response = rag_without_lcel(query)
    # print("Answer: \n")
    # print(response)

    print("-" * 60)
    print("IMPLEMENTATION 2 - LLM invokation (WITH RAG, WITH LCEL)")
    print("-" * 60)
    llm = rag_with_lcel()
    response = llm.invoke({"question": query})
    print("Answer: \n")
    print(response)


if __name__ == "__main__":
    main()
