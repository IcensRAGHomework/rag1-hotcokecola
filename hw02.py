import requests
import os
import json
import traceback
import re

from dotenv import load_dotenv


#from langchain_openai import ChatOpenAI
#from langchain.agents import AgentExecutor, create_openai_functions_agent
#from langchain import hub
from langchain_openai import AzureChatOpenAI
#from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from model_configurations import get_model_configuration
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
#from langchain_anthropic import ChatAnthropic
#from langchain_core.prompts import ChatPromptTemplate

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

CALENDARIFIC_API_KEY="VaIKuhkMlufwBdWje5KnmGByrxS34lN4"

def agent_hw02(question):

    output_national = {
        "Result": [
            {
                "date": "2024-10-10",
                "name": "國慶日"
            },
            {
                "date": "2024-10-09",
                "name": "重陽節"
            },
            {
                "date": "2024-10-21",
                "name": "華僑節"
            },
        ]
    }

    output_one = {
        "Result": [
            {
                "date": "2024-01-01",
                "name": "元旦"
            }
        ]
    }

    output_teacher = {
        "Result": [
            {
                "date": "2024-09-28",
                "name": "教師節"
            }
        ]
    }

    json_national = json.dumps(output_national, indent=4, ensure_ascii=False).encode('utf8').decode()
    json_one = json.dumps(output_one, indent=4, ensure_ascii=False).encode('utf8').decode()
    json_teacher = json.dumps(output_teacher, indent=4, ensure_ascii=False).encode('utf8').decode()
    
    examples = [
        { "input": "10月節日", "output": json_national},
        { "input": "1月節日", "output": json_one},
        { "input": "9月節日", "output": json_teacher},
    ]
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "這個任務把tool回傳的資料修改格式城我們定義的JSON"),
            few_shot_prompt,
            #("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    
    
    #prompt = hub.pull("hwchase17/openai-functions-agent")
    model = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )


    @tool
    def get_holidays(country: str, year: int, month: int) -> str:
        """
        Searches holidays 
        
        Args:
            country (str): The country name. ex. cn, tw, us
            year    (int): year
            month   (int): month
        
        Returns:
            str: The search results.
        """
        api_key = os.getenv('CALENDARIFIC_API_KEY')
        url = f'https://calendarific.com/api/v2/holidays?api_key={api_key}&country={country}&year={year}&month={month}'
    
        #print(url)
    
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(f'Error fetching holidays: {e}')
        return None


    tools = [get_holidays]
    
    agent = create_tool_calling_agent(model, tools, prompt)
    #agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    response = agent_executor.invoke({"input": question})
    
    # Using with chat history
    # from langchain_core.messages import AIMessage, HumanMessage
    # agent_executor.invoke(
    #     {
    #         "input": "what's my name?",
    #         "chat_history": [
    #             HumanMessage(content="hi! my name is bob"),
    #             AIMessage(content="Hello Bob! How can I assist you today?"),
    #         ],
    #     }
    # )
    
    return response
    