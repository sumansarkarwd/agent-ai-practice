import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


def main():
    OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    INDEX_NAME = os.environ.get("INDEX_NAME")

    if not OPEN_AI_KEY:
        raise ValueError("OPENAI_API_KEY not set!")

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set!")

    if not INDEX_NAME:
        raise ValueError("INDEX_NAME not set!")

    print("Ingesting...")
    doc_path = Path(__file__).parent / "mediumblog.txt"
    doc = [
        Document(
            page_content=doc_path.read_text(encoding="utf-8"),
            metadata={"source": str(doc_path)},
        )
    ]

    print("Spliting...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splitter.split_documents(doc)
    print(f"Create {len(texts)} chunks!")

    print("Ingesting...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)
    PineconeVectorStore.from_documents(texts, embeddings, index_name=INDEX_NAME)


if __name__ == "__main__":
    main()
