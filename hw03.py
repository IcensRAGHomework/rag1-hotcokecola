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
from langchain_core.messages import AIMessage, HumanMessage
#from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from model_configurations import get_model_configuration
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
#from langchain_anthropic import ChatAnthropic
#from langchain_core.prompts import ChatPromptTemplate

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

#CALENDARIFIC_API_KEY="VaIKuhkMlufwBdWje5KnmGByrxS34lN4"

def agent_hw03(question2, question3, response):
    output_oct = {
        "Result":
            {
                "add": 1,
                "reason": "蔣中正誕辰紀念日並未包含在十月的節日清單中。目前十月的現有節日包括國慶日、重陽節、華僑節、台灣光復節和萬聖節。因此，如果該日被認定為節日，應該將其新增至清單中。"
            }
    }

    output_may1 = {
        "Result":
            {
                "add": 0,
                "reason": "母親節紀念日已包含在五月的節日清單中。目前五月的現有節日包括勞動節、媽祖誕辰、文藝節、母親節和佛誕日。"
            }
    }

    output_may2 = {
        "Result":
            {
                "add": 0,
                "reason": "勞動節紀念日已包含在五月的節日清單中。目前五月的現有節日包括勞動節、媽祖誕辰、文藝節、母親節和佛誕日。"
            }
    }

    output_aug = {
        "Result":
            {
                "add": 1,
                "reason": "母親節紀念日並未包含在八月的節日清單中。目前八月的現有節日包括父親節、七夕情人節和中元節。因此，如果該日被認定為節日，應該將其新增至清單中。"
            }
    }

    json_oct = json.dumps(output_oct, indent=4, ensure_ascii=False).encode('utf8').decode()
    json_may1 = json.dumps(output_may1, indent=4, ensure_ascii=False).encode('utf8').decode()
    json_may2 = json.dumps(output_may2, indent=4, ensure_ascii=False).encode('utf8').decode()    
    json_aug = json.dumps(output_aug, indent=4, ensure_ascii=False).encode('utf8').decode()
    
    examples = [
        { "input": "蔣中正誕辰紀念日是否有含在10月節日", "output": json_oct},
        { "input": "母親節紀念日是否有含在5月節日", "output": json_may1},
        { "input": "勞動節紀念日是否有含在5月節日", "output": json_may2},
        { "input": "母親節紀念日是否有含在8月節日", "output": json_aug},
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
            ("system", "這個任務需看前一次的問題與歷史紀錄,並回答問題 true or false. \
                原因需要列出歷史紀錄內所有當月的節日. \
                所有回傳內容要按照我們定義的JSON格式"),
            few_shot_prompt,
            ("placeholder", "{chat_history}"),
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
        api_key = "VaIKuhkMlufwBdWje5KnmGByrxS34lN4" #os.getenv('CALENDARIFIC_API_KEY')
        #print("E")
        #print(api_key)
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
    
    #response = agent_executor.invoke({"input": question3})
    
    response = agent_executor.invoke(
        {
            "chat_history": [
                HumanMessage(content=question2),
                AIMessage(content=response),
            ],
            "input": question3,
        }
    )
    
    
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
    