import os.path
import hashlib

import dotenv
import chromadb
from chromadb.errors import IDAlreadyExistsError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from loader.loader import DirectoryLoader
import chromadb.utils.embedding_functions as embedding_functions

OPENAI_KEY = 'OPENAI_API_KEY'

TEMPLATE = ("You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answer concise."
            "Question: {question}"
            "Context: {context}"
            "Answer:")

COLLECTION = "PDFChat"


def main():
    dotenv.load_dotenv()
    documents = DirectoryLoader("Documents").load_directory()
    print(documents[0].metadata)

    # load if it already exists
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ[OPENAI_KEY],
        model_name="text-embedding-ada-002"
    )
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=openai_ef,
    )

    try:
        ids = [hashlib.md5(i.page_content.encode()).hexdigest() for i in documents]
        metadatas = [i.metadata for i in documents]
        texts = [i.page_content for i in documents]
        collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=texts,
        )
    except IDAlreadyExistsError as id_error:
        # This means that some files were already loaded before.
        # This is expected for now.
        print(f"Files were already loaded => {id_error}")

    # PDF retriever
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(),
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    prompt_template = PromptTemplate.from_template(TEMPLATE)

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
