import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")

    if not OPEN_AI_KEY:
        raise ValueError("OPENAI_API_KEY not set!")

    introduction_path = Path(__file__).parent / "data" / "stephen_hawking.txt"
    introduction = introduction_path.read_text()

    template_text = """Here is the details of a person.
    Details: {introduction}
    Your task is to provide a brief summary and 2 interesting facts about the person.
    """

    prompt_template = PromptTemplate(
        input_variables=["introduction"], template=template_text
    )

    llm = ChatOpenAI(
        base_url="https://openai.vocareum.com/v1",
        api_key=OPEN_AI_KEY,
        temperature=0,
        model="gpt-5-nano",
    )

    chain = prompt_template | llm

    reponse = chain.invoke(input={"introduction": introduction})

    print(reponse.content)


if __name__ == "__main__":
    main()
