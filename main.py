import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main():
    OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
    print(f"OPEN_AI_KEY: {OPEN_AI_KEY}")
    open_ai_client = OpenAI(
        api_key=OPEN_AI_KEY, base_url="https://openai.vocareum.com/v1"
    )


if __name__ == "__main__":
    main()
