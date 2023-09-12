import openai
import json


# llm = OpenAI(openai_api_key="sk-RUYMa4nzjcQHBvVmPgYvsYR3A9Nd6OwRgtK1nRqCvFfOUusn",
#              openai_api_base="https://api.chatanywhere.com.cn/v1", temperature=0, max_tokens=1500)
# 设置 OpenAI API 密钥和基本 URL
openai.api_key = "sk-RUYMa4nzjcQHBvVmPgYvsYR3A9Nd6OwRgtK1nRqCvFfOUusn"
openai.api_base = "https://api.chatanywhere.com.cn/v1"

# 要嵌入的文本
text = "Hello, world!"

# 调用 OpenAI Embedding API
# response = openai.Embedding.create(
#     engine="text-davinci-002",
#     input=text
# )
response = openai.Embedding.create(
   model="text-embedding-ada-002",
   input=text
)
# 解析响应并提取嵌入向量
# embedding = json.loads(response.to_json())["data"][0]["embedding"]
embedding = response["data"][0]["embedding"]

print(embedding)
# print(response)