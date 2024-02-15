import os.path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents.base import Document


def _add_directory(d: Document):
    source = d.metadata['source']
    d.metadata['dir'] = os.path.dirname(source)


class DirectoryLoader:

    def __init__(self, directory_path: str):
        self._directory_path = directory_path

    def load_directory(self) -> list[Document]:
        loader = PyPDFDirectoryLoader(self._directory_path, recursive=True)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)
        [_add_directory(i) for i in all_splits]
        return all_splits
