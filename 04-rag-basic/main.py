import os

from dotenv import load_dotenv

load_dotenv()


def main():
    OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

    if not OPEN_AI_KEY:
        raise ValueError("OPENAI_API_KEY not set!")

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set!")


if __name__ == "__main__":
    main()
