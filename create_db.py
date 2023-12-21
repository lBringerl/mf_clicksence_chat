import os
from pathlib import Path
import time
from typing import List

import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import openai
import PyPDF2

from paths import OPENAI_TOKEN_PATH, CHROMA_STORE_PATH, DOCUMENTS_DIR_PATH
from embedder import get_embedder_function, query_embedding


zero_key =  OPENAI_TOKEN_PATH.read_text()

openai.api_key = zero_key
os.environ['OPENAI_API_KEY'] = zero_key
os.environ['VERBOSE'] = 'True' # To see what’s going on in the background

MODEL = 'gpt-3.5-turbo' # gpt-3.5-turbo-1106  gpt-3.5-turbo
MODEL_EMBEDDING = 'text-embedding-ada-002'


filenames = [
  '1. Начинаем с основ.pdf',
  '2. Построение приложения.pdf',
  '3. Построение скрипта для начинающих.pdf',
  '4. Построение скрипта для продолжающих.pdf',
  '5. Выражения диаграммы.pdf',
  '6. Создание приложений и визуализаций.pdf',
  '7. Управление данными.pdf',
  '8. Cинтаксис скрипта и функции диаграммы.pdf',
  '9. Исследуйте и анализируйте.pdf',
  '10. Совместная работа в Qlik Sense.pdf'
]


def exceptions_replacing(string):
  exceptions = ['\xa0',
                'Учебное пособие —построение скрипта дляначинающих -QlikSense,',
                'Учебноепособие—построение приложения -QlikSense,',
                'Учебноепособие—выражения диаграммы -QlikSense,',
                'Учебное пособие —начинаем соснов-QlikSense,',
                'Учебноепособие—следующие этапывпостроении скрипта-QlikSense,',
                'May2021']
  for exception in exceptions:
    string = string.replace(exception, '')
  return string


def parse_docs(directory: str, filenames: List[str]):
    documents = []
    metadatas = []
    for filename in filenames:
        with open(str(Path(directory).joinpath(filename)), 'rb') as doc:
            pdf_reader = PyPDF2.PdfReader(doc)
            for page in range(4, len(pdf_reader.pages)):
                extracted_text = pdf_reader.pages[page].extract_text()
                clean_text = extracted_text.replace('\n',' ')
                clean_text = exceptions_replacing(clean_text)
                clean_text = clean_text[:-3]
                documents.append(clean_text)
                metadata = {'doc' : filename, 'page' : str(page + 1)}
                metadatas.append(metadata)
    return documents, metadatas


def load_docs(filenames, path: str):
    documents = []
    for filename in filenames:
        doc_path = Path(path).joinpath(filename)
        loader = PyPDFLoader(str(doc_path))
        documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)
    return documents, chunked_documents


def create_db(documents, persist_directory: str) -> Chroma:
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=documents[:1000],
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb, embedding


def ask_llm(vectordb: Chroma, question: str):
    llm = ChatOpenAI(model_name=MODEL)
    chain = load_qa_chain(llm, chain_type='stuff')
    matching_docs = vectordb.similarity_search(question)
    matching_docs_documents = [i.page_content for i in matching_docs]
    matching_docs_metadatas = [i.metadata for i in matching_docs]
    tuple_matching_docs = tuple(
        zip(matching_docs_documents, matching_docs_metadatas)
    )
    answer = chain.run(input_documents=matching_docs,
                       question=question)
    return answer


def write_collection(name, client, openai_ef, ids, texts, metadatas):
    existing_collections = list(
        map(lambda c: c.name, client.list_collections())
    )
    if name in existing_collections:
        client.delete_collection(name=name)
    if not len(ids) == len(texts) == len(metadatas):
        raise ValueError('Length of ids, texts, metadatas is not the same')
    collection = client.create_collection(name,
                                          embedding_function=openai_ef,
                                          metadata={"hnsw:space": "cosine"})
    multiplier = 1000
    for i in range(len(ids) // multiplier):
        collection.add(documents=texts[multiplier * i:multiplier * (i + 1)],
                       ids=ids[multiplier * i:multiplier * (i + 1)],
                       metadatas=metadatas[multiplier * i:multiplier * (i + 1)])
        time.sleep(70)
    collection.add(documents=texts[multiplier * (i + 1):],
                    ids=ids[multiplier * (i + 1):],
                    metadatas=metadatas[multiplier * (i + 1):])


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_STORE_PATH))
    texts, metadatas = parse_docs(directory=str(DOCUMENTS_DIR_PATH),
                                  filenames=filenames)
    ids = [str(x) for x in range(len(texts))]

    write_collection(name='qlicksense',
                     client=client,
                     openai_ef=get_embedder_function(),
                     ids=ids,
                     texts=texts,
                     metadatas=metadatas)


if __name__ == '__main__':
    main()
