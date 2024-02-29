import json
import os.path
import uuid

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_core.documents.base import Document


def _add_directory(d: Document):
    source = d.metadata['source']
    d.metadata['dir'] = os.path.dirname(source)


class Loader:
    """
    The Loader class takes a documents directory path relative
    to the execution path and a DB collection client. The class parses
    the directory and updates the DB collection entries as required.
    """

    def __init__(self, directory_path: str, collection: chromadb.Collection):
        self._directory_path = directory_path
        self._collection = collection

    def load(self):
        """
        Load the directory data into the DB collection.
        :return:
        """
        self._load_directory(self._directory_path)

    def _add_to_collection(self, documents: list[Document]):
        if len(documents) < 1:
            return
        ids = [uuid.uuid4().hex for _ in documents]
        metadatas = [i.metadata for i in documents]
        texts = [i.page_content for i in documents]
        [_add_directory(i) for i in documents]
        self._collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=texts,
        )

    def _delete_from_collection(self, document_names: list[str]):
        if len(document_names) == 0:
            return
        elif len(document_names) == 1:
            self._collection.delete(where={"source": document_names[0]})
        else:
            filters = []
            [(filters.append({"source": i})) for i in document_names]
            self._collection.delete(where={"$or": filters})

    def _load_directory(self, path: str):
        cur_files, dirs = {}, []
        special = ""

        for p in os.listdir(path):
            full_path = os.path.join(path, p)
            if os.path.isfile(full_path):
                if p == ".files":
                    special = full_path
                    continue
                # Hash the float the guarantee no weirdness from saving floats as text
                cur_files[full_path] = os.path.getmtime(full_path).hex()
                continue
            dirs.append(full_path)

        if special == "":
            # if directory has not been visited before
            # load all files and then dump the dictionary
            special = os.path.join(path, ".files")
            loader = PyPDFDirectoryLoader(path, recursive=False)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, add_start_index=True
            )
            all_splits = text_splitter.split_documents(docs)
            self._add_to_collection(all_splits)

            with open(special, 'x') as f:
                f.write(json.dumps(cur_files))
        else:
            with open(special, 'r') as f:
                old_files = json.loads(f.read())

            new_dot_file = {}
            docs_to_add = []
            docs_to_delete: list[str] = []

            for k, v in old_files.items():
                cur = cur_files.get(k)
                if cur is None:
                    # Deleted files
                    print(f'{k} to be deleted')
                    docs_to_delete.append(k)
                    continue

                if v != cur:
                    # Updated files
                    print(f'{k} to be updated')
                    docs_to_delete.append(k)
                    loader = PyPDFLoader(k)
                    docs = loader.load_and_split()
                    docs_to_add.extend(docs)

                new_dot_file[k] = cur
                # Leave only new files in cur_files
                del cur_files[k]

            for k, v in cur_files.items():
                # Those are all the new files we need to add
                print(f'{k} to be added')
                new_dot_file[k] = v
                loader = PyPDFLoader(k)
                docs = loader.load_and_split()
                docs_to_add.extend(docs)

            # Delete deleted files and old versions of
            # updated files. Then add the new versions
            # of updated files and newly added files.
            self._delete_from_collection(docs_to_delete)
            self._add_to_collection(docs_to_add)

            with open(special, 'w') as f:
                f.write(json.dumps(new_dot_file))

        for dir_path in dirs:
            self._load_directory(dir_path)
