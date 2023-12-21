from pathlib import Path


PROJECT_HOME = Path(__file__).parent
SECRETS_HOME = PROJECT_HOME.joinpath('mf_clicksence_chat_secrets')
TG_TOKEN_PATH = SECRETS_HOME.joinpath('tg_key.key')
OPENAI_TOKEN_PATH = SECRETS_HOME.joinpath('openai_key.key')
DOCUMENTS_DIR_PATH = PROJECT_HOME.joinpath('PDFs')
CHROMA_STORE_PATH = PROJECT_HOME.joinpath('chroma_store')
