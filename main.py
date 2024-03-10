import os.path
import argparse

import dotenv
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from loader.loader import Loader
from history.history import ChatHistory
import chromadb.utils.embedding_functions as embedding_functions

OPENAI_KEY = 'OPENAI_API_KEY'
OPENAI_MODEL = 'OPENAI_MODEL'

TEMPLATE = ("You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answer concise."
            "You have also been provided with the chat history to keep track of the conversation. "
            "Chat history: {history}"
            "Context: {context}"
            "Question: {question}"
            "Answer:")

COLLECTION = "PDFChat"


def main():
    parser = argparse.ArgumentParser(
        prog='chatAI',
        description='An indexing application that allows you to chat with your PDF documents')
    parser.add_argument("-d", "--directory", help="If populated only the specified directory will be used in the chat "
                                                  "context. Enter the path relative to the repository root.")
    parser.add_argument("-f", "--file", help="If populated only the specified file will be used in the chat context. "
                                             "Enter the path relative to the repository root. Takes precedence over "
                                             "directory flag")
    parser.add_argument("-r", "--retention", default=15, help="Sets the retention for the chat history. The default is "
                                                              "15. This means that the model will remember only the "
                                                              "last 15 exchanges.")
    args = parser.parse_args()

    # TODO: Add option to use local models.
    dotenv.load_dotenv()

    # Create DB and load question
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ[OPENAI_KEY],
        model_name="text-embedding-ada-002"
    )
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=openai_ef,
    )
    Loader("Documents", collection).load()

    # PDF retriever
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(),
    )

    if args.file is not None:
        src = args.file.strip('/')
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6, 'filter': {'source': src}},
        )
    elif args.directory is not None:
        directory = args.directory.strip('/')
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6, 'filter': {'dir': directory}},
        )
    else:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    llm = ChatOpenAI(model_name=os.environ[OPENAI_MODEL], temperature=0)
    prompt_template = PromptTemplate.from_template(TEMPLATE)

    rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt_template
            | llm
            | StrOutputParser()
    )

    chat_history = ChatHistory(retention_window=args.retention)
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough(), "history": RunnableLambda(chat_history.text)}
    ).assign(answer=rag_chain_from_docs)

    while True:
        question = input("Please enter your question:\n")
        if 'exit' == question or 'q' == question:
            break

        output = rag_chain_with_source.invoke(question)
        documents = output["context"]
        print('Sources are:')
        for d in documents:
            page_content = d.page_content.replace("\\n", "\n")
            print(f'Metadata: {d.metadata}\n{page_content}\n')
        print(f'Question: {question}\nAnswer: {output["answer"]}')
        chat_history.add_history(question, output["answer"])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    main()
