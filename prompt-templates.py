from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StringOutputParser, CommaSeparatedListOutputParser

# Instantiate Model
model = ChatOpenAI(
    temperature=0.7, # 0-Factual, 1-Creative
    model="gpt-3.5-turbo-1106",
)

# Prompt Template
def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Tell me a joke about the following subject."),
            ("human", "{input}")
        ]
    )

    parser = StringOutputParser()
    
    chain = prompt | model | parser

    return chain.invoke({
        "input": "dog"
    })

def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
     
     ("system", "Tell me a joke about the following subject."),
     ("human", "{input}")   
    ])
    
    parser = CommaSeparatedListOutputParser()
    
    chain = prompt | model | parser
    
    return chain.invoke({
        "input": "happy"
    })

print(call_list_output_parser())