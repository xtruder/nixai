import operator

from typing import Annotated, List, Optional, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_openai import ChatOpenAI

from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.globals import set_verbose
from langchain.agents import create_openai_tools_agent
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.tools.vectorstore.tool import VectorStoreQATool

from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_community.tools.github.tool import GitHubAction
from langchain_community.tools.github.prompt import SEARCH_CODE_PROMPT
from langchain_community.llms.openai import OpenAI


from langgraph.prebuilt import ToolExecutor
from langgraph.graph import StateGraph, END

from fastapi import FastAPI
from langserve import add_routes

from nixai.nixdoc import retriever as nixdoc_retriever, vectorstore as nixdoc_vectorstore


set_verbose(True)

nixpkgs_github = GitHubAPIWrapper()

class SearchCode(BaseModel):
    """Schema for operations that require a search query as input."""

    instructions: str = Field(
        ...,
        description=(
            "A keyword-focused natural language search"
            "query for code, e.g. `mypackage` that would represents keywords in nix code syntax. Separate all tokens with spaces and skip common words like `latest`."
        ),
    )

class VectorStoreQATool(BaseTool, BaseModel):
    """Tool for the VectorDBQA chain. To be initialized with name and chain."""

    retriever: BaseRetriever = Field(exclude=True)
    llm: BaseLanguageModel = Field(default_factory=lambda: OpenAI(temperature=0))

    class Config(BaseTool.Config):
        pass

    # @staticmethod
    # def get_description(name: str, description: str) -> str:
    #     template: str = (
    #         "Useful for when you need to answer questions about {name}. "
    #         "Whenever you need information about {description} "
    #         "you should ALWAYS use this. "
    #         "**IMPORTANT** Input should be a fully formed question."
    #     )

    #     return template.format(name=name, description=description)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        template = """
You are an assistant for question-answering tasks abot nix package manager. You need to answer question in step-by-step fashion.
Use the following pieces of context to help you with answering the question at the end. Be aware that context could also
information that might not be related to answering the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

------
{context}
------

Question: {question}
Helpful Answer:"""

        prompt = PromptTemplate(
            template=template, 
            input_variables=[
                'context', 
                'question',
            ]
        )

        chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt})

        return chain.invoke(
            {chain.input_key: query},
            #config={"callbacks": [run_manager.get_child()] if run_manager else []},
        )[chain.output_key]

# Choose the LLM that will drive the agent
model = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0, streaming=True)

SEARCH_CODE_PROMPT = """
This tool will search for code in the nixos nixpkgs repository. **VERY IMPORTANT**: You must specify the search query as a string input parameter."""

tools: List[BaseTool] = [
    # seaches for github code in nixpkgs github repo
    # GitHubAction(
    #     name="SearchCode",
    #     description=SEARCH_CODE_PROMPT,
    #     mode="search_code",
    #     api_wrapper=nixpkgs_github,
    #     args_schema=SearchCode,
    # ),

    VectorStoreQATool(
        name="NixPackageManagerDocs",
        description="This tool answers questions how nix package manager works or how to use it. It also provides info on how to do packing for nixpkgs. Form a sentence that would look similar to a sentence inside nix package manager manual.",
        vectorstore=nixdoc_vectorstore,
        retriever=nixdoc_retriever,
        llm=model,
    ),
]

# Get the prompt to use
prompt = hub.pull("hwchase17/openai-tools-agent")

# Construct the OpenAI Functions agent
agent_runnable = create_openai_tools_agent(model, tools, prompt)

tool_executor = ToolExecutor(tools)

class AgentState(TypedDict):
    # Input string to an agent
    input: str

  # The list of previous messages in the conversation
    chat_history: list[BaseMessage]

    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[AgentAction, AgentFinish, None]

    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# Define the agent
def run_agent(data):
    inputs = data.copy()
    if len(inputs["intermediate_steps"]) > 5:
        inputs["intermediate_steps"] = inputs["intermediate_steps"][-5:]

    agent_outcome = agent_runnable.invoke(inputs)
    return {"agent_outcome": agent_outcome}

# Define the function to execute tools
def execute_tools(state: AgentState):
    agent_actions = state["agent_outcome"]

    results = []
    for action in agent_actions:
        print("invoking action", action)

        output = tool_executor.invoke(action)
        results.append((action, str(output)))

    return {"intermediate_steps": results}

    # messages = state['messages']
    # # Based on the continue condition
    # # we know the last message involves a function call
    # last_message = messages[-1]

    # tool_msgs = []
    # for tool_call in last_message.additional_kwargs["tool_calls"]:
    #     assert tool_call["type"] == "function", "invalid tool type"

    #     tool_call_id = tool_call["id"]
    #     func = tool_call["function"]
    #     name = func["name"]
    #     args = json.loads(func["arguments"])

    #     # We construct an ToolInvocation from the function_call
    #     action = ToolInvocation(
    #         tool=name,
    #         tool_input=args["__arg1"] if "__arg1" in args else args,
    #     )
    #     # We call the tool_executor and get back a response
    #     response = tool_executor.invoke(action)

    #     print("tool resp", response)

    #     tool_msgs.append(ToolMessage(tool_call_id=tool_call_id, content=str(response), name=action.tool))

    # # We return a list, because this will get added to the existing list
    # return {"messages": tool_msgs}


# Define logic that will be used to determine which conditional edge to go down
def should_continue(data):
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if isinstance(data["agent_outcome"], AgentFinish):
        return "end"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "continue"

# Choose the LLM that will drive the agent
model = model.bind_tools(tools)

# Define a new graph
workflow = StateGraph(AgentState)


# Define the two nodes we will cycle between
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END
    }
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge('action', 'agent')

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

# 4. App definition
api = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.

class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )


class Output(BaseModel):
    output: str

add_routes(
    api,
    app,
    path="/agent",
    input_type=Input,
    output_type=Output,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=8000)


# inputs = {"input": "How to uninstall nix on macos?", "chat_history": []}
# for output in app.stream(inputs):
#     # stream() yields dictionaries with output keyed by node name
#     for key, value in output.items():
#         result: AgentState = value
#         print(f"Output from node '{key}':")
#         print("---")
#         print(value)

#         if "agent_outcome" in result and "return_values" in result["agent_outcome"][0]:
#             print(result["agent_outcome"][0].return_values["output"])
#     print("\n---\n")


# Create an agent executor by passing in the agent and tools
# agent_executor = AgentExecutor(
#     agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
# )
