from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents.agent_types import AgentType
from openpowerlifting.sql_tools import sql_agent_tools
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.base import SQLDatabaseToolkit
from openpowerlifting.custom_suffix import CUSTOM_SUFFIX
from openpowerlifting.db_utils import db
from openpowerlifting.config import settings


### Creating the agent ###

def create_agent(
    tool_llm_name: str = settings.llm_model_name,
    agent_llm_name: str = settings.llm_model_name,
):
    agent_tools = sql_agent_tools()
    llm_agent = get_agent_llm(agent_llm_name)
    toolkit = get_sql_toolkit(tool_llm_name)
    memory = ConversationBufferWindowMemory(k=5, memory_key="history", input_key="input")

    agent = create_sql_agent(
        llm=llm_agent,
        toolkit=toolkit,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        input_variables=["input", "agent_scratchpad", "history"],
        suffix=CUSTOM_SUFFIX,
        agent_executor_kwargs={"memory": memory},
        extra_tools=agent_tools,
        verbose=True,
    )
    return agent


### Utility functions ###

def get_chat_openai(model_name):
    llm = ChatOpenAI(
        model_name=model_name,
        model_kwargs=settings.llm_chat_openai_model_kwargs,
        **settings.llm_chat_kwargs
    )
    return llm

def get_sql_toolkit(tool_llm_name: str):
    llm_tool = get_chat_openai(model_name=tool_llm_name)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm_tool)
    return toolkit

def get_agent_llm(agent_llm_name: str):
    llm_agent = get_chat_openai(model_name=agent_llm_name)
    return llm_agent
