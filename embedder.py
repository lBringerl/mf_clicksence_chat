import openai

from chromadb.utils import embedding_functions

from paths import OPENAI_TOKEN_PATH


zero_key =  OPENAI_TOKEN_PATH.read_text()

MODEL = 'gpt-3.5-turbo' # gpt-3.5-turbo-1106  gpt-3.5-turbo
MODEL_EMBEDDING = 'text-embedding-ada-002'


def get_embedder_function():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=zero_key,
        model_name=MODEL_EMBEDDING
    )
    return openai_ef


def query_embedding(text) -> None:
    response = openai.Embedding.create(model=MODEL_EMBEDDING, input=text)
    return response['data'][0]['embedding']
