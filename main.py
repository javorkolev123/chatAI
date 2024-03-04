import os.path
import argparse

import dotenv
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from loader.loader import Loader
import chromadb.utils.embedding_functions as embedding_functions

OPENAI_KEY = 'OPENAI_API_KEY'
OPENAI_MODEL = 'OPENAI_MODEL'

TEMPLATE = ("You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answer concise."
            "Question: {question}"
            "Context: {context}"
            "Answer:")

COLLECTION = "PDFChat"


def main():
    parser = argparse.ArgumentParser(
        prog='chatAI',
        description='An indexing program that allows you to chat with your PDF documents')
    _ = parser.parse_args()

    # TODO: Add option to use local models.
    dotenv.load_dotenv()

    # Create DB and load data
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

    # TODO: Add option to limit search to a certain directory
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    llm = ChatOpenAI(model_name=os.environ[OPENAI_MODEL], temperature=0)
    prompt_template = PromptTemplate.from_template(TEMPLATE)

    # TODO: Add dialogue memory
    rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt_template
            | llm
            | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    while True:
        data = input("Please enter your question:\n")
        if 'exit' == data or 'q' == data:
            break

        output = rag_chain_with_source.invoke(data)
        print(f'Sources: {output["context"]}\nQuestion: {output["question"]}\nAnswer: {output["answer"]}')


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    main()
