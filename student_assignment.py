import json
import traceback
import re


from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    output_national = {
        "date": "2024-10-10",
        "name": "國慶日"
    }
    output_one = {
        "date": "2024-01-01",
        "name": "元旦"
    }
    output_teacher = {
        "date": "2024-09-28",
        "name": "教師節"
    }

    json_national = json.dumps(output_national, ensure_ascii=False).encode('utf8').decode()
    json_one = json.dumps(output_one, ensure_ascii=False).encode('utf8').decode()
    json_teacher = json.dumps(output_teacher, ensure_ascii=False).encode('utf8').decode()
    
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

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "把當月份的節日用指定格式輸出"),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
 
 
    chain = final_prompt | llm
    response = chain.invoke({"input": question})

    #print(response.content)

    #Formatting the output
    extract_array = re.findall(r'\{.*?\}', response.content)
    extract_array = [eval(item) for item in extract_array]
    #number_of_items = len(extract_array)
    #print(number_of_items)
    output_array = {"Result": extract_array}
    output_json = json.dumps(output_array, indent=2, ensure_ascii=False).encode('utf8').decode()
 
    #print(output_json) 

    return output_json
    
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response.content

