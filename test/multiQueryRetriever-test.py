from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever


# Set logging for the queries
# import logging

# logging.basicConfig()
# logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# unique_docs = retriever_from_llm.get_relevant_documents(query=question)
# len(unique_docs)


import logging

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)


question = "What are the approaches to Task Decomposition?"
llm = ChatOpenAI(temperature=0,    
                 openai_api_key='sk-RUYMa4nzjcQHBvVmPgYvsYR3A9Nd6OwRgtK1nRqCvFfOUusn',
                 openai_api_base='https://api.chatanywhere.com.cn/v1')
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)


# retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(),
#                                                   llm=ChatOpenAI(temperature=0))
unique_docs = retriever_from_llm.get_relevant_documents(query=question)
len(unique_docs)

