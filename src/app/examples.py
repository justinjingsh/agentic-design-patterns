"""Example patterns for using LangChain with AWS Bedrock."""

from langchain.prompts import ChatPromptTemplate
from .bedrock import llm


def simple_prompt_example() -> str:
    """Simple prompt example."""
    response = llm.invoke("What is AWS Bedrock?")
    return response.content


def chain_example() -> str:
    """Example using LangChain prompt + chain."""
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in 2-3 sentences for a beginner."
    )
    chain = prompt | llm
    response = chain.invoke({"topic": "AWS Bedrock"})
    return response.content


def multi_turn_example() -> str:
    """Example with conversation history."""
    from langchain.schema import HumanMessage, AIMessage

    messages = [
        HumanMessage(content="What is LangChain?"),
    ]

    response = llm.invoke(messages)
    messages.append(response)

    messages.append(HumanMessage(content="How does it work with AWS?"))
    response = llm.invoke(messages)

    return response.content


if __name__ == "__main__":
    print("=== Simple Prompt ===")
    print(simple_prompt_example())

    print("\n=== Chain Example ===")
    print(chain_example())

    print("\n=== Multi-turn Example ===")
    print(multi_turn_example())
