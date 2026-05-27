import inspect
import os
import re

from dotenv import load_dotenv
from langsmith import traceable
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4.1-nano"
OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")
MAX_ITERATION = 5


@traceable(run_type="tool")
def get_product_price(name: str) -> float:
    """Tool to get product price by product name"""
    print(f"Looking up price for product {name}")
    products = {"laptop": 1000, "bag": 5, "charger": 50}
    return products.get(name, 0)


@traceable(run_type="tool")
def get_discounted_price(amount: float, tier: str) -> float:
    """Tool to get discounted price after applying discount based on tier
    available tiers are: gold, silver and bronze
    """
    print(f"Applying discount on amount: {amount}")
    tiers = {"gold": 70, "silver": 30, "bronze": 10}
    discount_percentage = tiers.get(tier)
    return round(amount * (1 - discount_percentage / 100), 2)


tools_dict = {
    "get_product_price": get_product_price,
    "get_discounted_price": get_discounted_price,
}


def _underlying_callable(tool_obj):
    if hasattr(tool_obj, "func"):
        return tool_obj.func
    return getattr(tool_obj, "__wrapped__", tool_obj)


def get_tool_description(tools):
    description = []

    for tool_name, tool_obj in tools.items():
        func = _underlying_callable(tool_obj)
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        description.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(description)


tools_description = get_tool_description(tools_dict)
tool_names = ", ".join(tools_dict.keys())


def _coerce_param(value: str, param: inspect.Parameter):
    annotation = param.annotation
    if annotation is float:
        return float(value)
    if annotation is int:
        return int(value)
    return value


def parse_action_input(tool_name: str, raw_input: str) -> dict:
    """Parse Action Input into kwargs for the tool, using its signature."""
    func = _underlying_callable(tools_dict[tool_name])
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    raw = raw_input.strip()

    if "=" in raw:
        kwargs = {}
        for part in re.split(r",\s*", raw):
            key, _, value = part.partition("=")
            key = key.strip()
            if key not in sig.parameters:
                raise ValueError(f"Unknown parameter {key!r} for {tool_name}")
            kwargs[key] = _coerce_param(value.strip(), sig.parameters[key])
        return kwargs

    if len(param_names) == 1:
        name = param_names[0]
        return {name: _coerce_param(raw, sig.parameters[name])}

    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != len(param_names):
        raise ValueError(
            f"Expected {len(param_names)} values ({', '.join(param_names)}), "
            f"got {len(parts)} in {raw_input!r}"
        )
    return {
        name: _coerce_param(val, sig.parameters[name])
        for name, val in zip(param_names, parts)
    }


react_prompt = f"""
STRICT RULES:
1. You use `get_product_price` tool to know about a product price
2. You use `get_discounted_price` to apply discount and return the final price
3. You never call `get_discounted_price` before you have got the price from `get_product_price`
4. You never calculate the product discount using math, you use `get_discounted_price` only to get the discounted price.
5. If the user has not mentioned the discount tier, then you ask the user for the discount tier.
            
Answer the following questions as best you can. You have access to the following tools:

{tools_description}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""


client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=OPEN_AI_KEY,
)


@traceable(name="Chat trace", run_type="llm")
def chat_traced(messages, stop=None, temperature=0):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        stop=stop,
    )
    return response.choices[0].message.content


@traceable(name="Agent loop")
def run_agent(query: str):
    print(f"Query: {query}")
    print("*" * 60)

    prompt = react_prompt.format(question=query)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATION + 1):
        print(f"Current iteration: {iteration}")

        full_prompt = prompt + scratchpad

        output = chat_traced(
            messages=[{"role": "user", "content": full_prompt}],
            stop=["\nObservation"],
        )

        print("*" * 60)
        print(f"LLM output: {output}")
        print("*" * 60)

        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("*" * 60)
            print(f"Parsed final answer: {final_answer}")
            print("*" * 60)
            return final_answer

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("Unable to parse action match or action input match")
            break

        tool = action_match.group(1).strip()
        tool_args = action_input_match.group(1).strip()

        if tool not in tools_dict:
            observation = (
                f"Unknown tool: {tool}. Available tools: {list[str](tool_names)}"
            )
            scratchpad += output + f"\nObservation: {observation}\n"
            break

        try:
            kwargs = parse_action_input(tool, tool_args)
        except ValueError as e:
            observation = f"Failed to parse action input: {e}"
            scratchpad += output + f"\nObservation: {observation}\n"
            break

        print(f"Tool selection: {tool}, kwargs: {kwargs}")
        print(f"Executing {tool} with args: {kwargs}")
        observation = tools_dict[tool](**kwargs)
        print("*" * 60)
        print(f"Observation: {observation}")
        print("*" * 60)

        scratchpad += output + f"\nObservation: {observation}\n"

    print("Max iterations exhausted!")
    return None


def main():
    OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")

    if not OPEN_AI_KEY:
        raise ValueError("OPENAI_API_KEY not set!")

    run_agent("What is the price of the laptop after applying gold discount?")


if __name__ == "__main__":
    main()
