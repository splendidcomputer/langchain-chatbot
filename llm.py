from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7, # 0-Factual, 1-Creative
    max_tokens=1000,
    verbose=True
)

# response = llm.invoke("What is the meaning of life?")

# response = llm.batch(["Hello, how are you?","Write a poem about AI"])
# print(response)

response = llm.stream(["Write a poem about AI"])

for chunk in response:
    print(chunk.content, end="", flush=True)