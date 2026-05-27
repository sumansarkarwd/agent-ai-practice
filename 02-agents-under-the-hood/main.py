import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langsmith import traceable

load_dotenv()

MODEL = "gpt-4.1-nano"
OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")
MAX_ITERATION = 10


@tool
def get_product_price(name: str) -> float:
    """Tool to get product price by product name"""
    print(f"Looking up price for product {name}")
    products = {"laptop": 1000, "bag": 5, "charger": 50}
    return products.get(name, 0)


@tool
def get_discounted_price(amount: float, tier: str) -> float:
    """Tool to get discounted price after applying discount based on tier
    available tiers are: gold, silver and bronze
    """
    print(f"Applying discount on amount: {amount}")
    tiers = {"gold": 70, "silver": 30, "bronze": 10}
    discount_percentage = tiers.get(tier)
    return round(amount * (1 - discount_percentage / 100), 2)


@traceable(describe="Langchain agent loop")
def run_agent(query: str):
    tools = [get_product_price, get_discounted_price]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(
        model=f"openai:{MODEL}",
        temperature=0,
        base_url="https://openai.vocareum.com/v1",
        api_key=OPEN_AI_KEY,
    )
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content="""
            You are a shopping cart assistant.
            You search for product price then apply discount then return the final price.
            You never guess the products price, you always call tools to calculate the price.
            STRICT RULES:
            1. You use `get_product_price` tool to know about a product price
            2. You use `get_discounted_price` to apply discount and return the final price
            3. You never call `get_discounted_price` before you have got the price from `get_product_price`
            4. You never calculate the product discount using math, you use `get_discounted_price` only to get the discounted price.
            5. If the user has not mentioned the discount tier, then you ask the user for the discount tier.
            """
        ),
        HumanMessage(content=query),
    ]

    print(f"Query: {query}")
    print("*" * 60)

    for iteration in range(1, MAX_ITERATION + 1):
        print(f"Current iteration: {iteration}")

        response = llm_with_tools.invoke(input=messages)

        if not response.tool_calls:
            print(f"Final result: {response.content}")
            return response.content

        tool_name = response.tool_calls[0].get("name")
        tool_to_call = tools_dict.get(tool_name)
        tool_call_id = response.tool_calls[0].get("id")

        if not tool_to_call:
            raise ValueError("Invalid tool name called")

        args = response.tool_calls[0].get("args")

        print(f"Tool selected: {tool_name} with args: {args}")

        observation = tool_to_call.invoke(args)

        print(f"Tool call result: {observation}")

        messages.append(response)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("Max iterations exhausted!")
    return None


def main():
    OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")

    if not OPEN_AI_KEY:
        raise ValueError("OPENAI_API_KEY not set!")

    run_agent("What is the price of the laptop after applying gold discount?")


if __name__ == "__main__":
    main()
