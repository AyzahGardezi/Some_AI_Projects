from langchain_core.messages import HumanMessage    # high level framework that allows us to build AI applications
from langchain_openai import ChatOpenAI             # allow us to use open AI within LangChain and LangGraph
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent   # complex framework that allows us to build AI agents
from dotenv import load_dotenv

# AI agent has access to tools - tool can be anything, calculate, logical operations, etc

load_dotenv()

def main():
    model = ChatOpenAI(temperature=0)   # low randomness of the model

    tools = []
    agent_executor = create_react_agent(model, tools)

    print("AI assistant lauched. Type 'quit' to exit.")
    print("I can perform calculations and chat with you.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input == "quit":
            break

        print("\nAssistant: ", end="") # end stops new line to be added

        # chunks are parts of a response coming from the agent
        # loop through the chunks to see if there is a response from the agent
        # if there is a response then we see if there are any messages in the response
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")

        print()

if __name__ == "__main__":
    main()

