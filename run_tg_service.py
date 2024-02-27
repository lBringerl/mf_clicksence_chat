import os

from aiogram import Bot, Dispatcher, executor, types
import openai
import chromadb
from chromadb.utils import embedding_functions

from paths import CHROMA_STORE_PATH, OPENAI_TOKEN_PATH, TG_TOKEN_PATH


OPENAI_TOKEN = OPENAI_TOKEN_PATH.read_text().strip()
TG_TOKEN = TG_TOKEN_PATH.read_text().strip()

openai.api_key = OPENAI_TOKEN
os.environ['OPENAI_API_KEY'] = OPENAI_TOKEN
os.environ['VERBOSE'] = 'True'


class Reference:
    """
    <description>
    """
    def __init__(self) -> None:
        self.response = ''


reference = Reference()

repo = '/content/drive/MyDrive/nlp_qliksence'
MODEL_NAME = 'gpt-3.5-turbo'
MODEL_EMBEDDING = 'text-embedding-ada-002'

# Initialize bot and dispatcher
bot = Bot(token=TG_TOKEN)
dispatcher = Dispatcher(bot)


# ----------------MY CODE STARTS HERE----------------
def query_embedding(text) -> None:
    '''
    Creates the embedding of a given query.
    '''
    response = openai.Embedding.create(model = MODEL_EMBEDDING,
                                       input = text)
    return response['data'][0]['embedding']
# ----------------MY CODE ENDS HERE----------------


def clear_past():
    """
    A function to clear the previous conversation and context.
    """
    reference.response = ''


@dispatcher.message_handler(commands=['start'])
async def welcome(message: types.Message):
    """
    A handler to welcome the user and clear past conversation and context.
    """
    clear_past()
    await message.reply("Привет! \nЯ чат-бот, который поможет в работе с Qlik Sense!\
                        \nЧем я могу помочь?")


@dispatcher.message_handler(commands=['clear'])
async def clear(message: types.Message):
    """
    A handler to clear the previous conversation and context.
    """
    clear_past()
    await message.reply("Я очистил прошлый разговор и контекст.")


@dispatcher.message_handler(commands=['help'])
async def helper(message: types.Message):
    """
    A handler to display the help menu.
    """
    help_command = """
    Привет! Я чат-бот, который поможет в работе с QlikSense! Пожалуйста, используйте эти команды -
    /start - начать разговор
    /clear - очистить прошлый разговор и контекст
    /help - вызвать меню помощи
    """
    await message.reply(help_command)


@dispatcher.message_handler()
async def chatgpt(message: types.Message):
    """
    A handler to process the user's input and generate a response using the chatGPT API.
    """
    print(f">» USER: \n{message.text}")

    # ----------------MY CODE STARTS HERE----------------
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = openai.api_key,
        model_name = MODEL_EMBEDDING
    )

    client = chromadb.PersistentClient(path=str(CHROMA_STORE_PATH))
    collection = client.get_collection('qlicksense',
                                       embedding_function=openai_ef)

    query_vector = query_embedding(message.text)

    results = collection.query(query_embeddings=query_vector,
                               n_results=3,
                               include=['documents', 'metadatas', 'distances'])

    # tuple_results = tuple(zip(results['distances'][0], results['documents'][0], results['metadatas'][0]))

    # meta = set([f'Документ "{i[2]["doc"]}", страница {i[2]["page"]}' \
    #                                        for i in tuple_results if i[0] <= 0.16]) # threshold

    space = '\n'.join(str(item) for item in results['documents'][0]) # all OR limited by threshold

    # system_prompt
    replacements = {'жира' : 'Jira', 'Жира' : 'Jira'}
    system_prompt = f"""Используя контекст, дай ответ на вопрос в конце.
    Тебе пишут пользователи сервиса QlikSense. Ты помогаешь отвечать на вопросы, которые они не знают.
    Если ответа нет в контексте, не пытайся придумывать ответ. Если информация не найдена в контексте,\
    вежливо попроси задать вопрос про QlikSense.
    Если спрашивается какой-то общий вопрос не по контексту, например, "сколько будет 2 + 2", или вообще не спрашивается вопрос,\
    вежливо попроси задать вопрос про QlikSense.
    При ответе делай следующие замены слов:
    {replacements}
    Отвечай на том языке, на котором задан вопрос."""

    # user_prompt
    user_prompt = f""" Контекст: {space}\
    Вопрос: {message.text}
    Если в контексте не найдена информаци, не говори, что не получилось найти ответ в контексте."""

    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}]

    # ----------------MY CODE ENDS HERE----------------

    print(f">» chatGPT: \nПечатает...")
    await bot.send_message(chat_id=message.chat.id, text='Печатает...')

    response = openai.ChatCompletion.create(
        model = MODEL_NAME,
        messages = messages,
        temperature = 0
    )

    reference.response = response['choices'][0]['message']['content']

    print(f">» chatGPT: \n{reference.response}")
    await bot.send_message(chat_id=message.chat.id, text=f"{reference.response}")

    # if len(meta) != 0:
    #   meta = 'Узнать больше:\n' + '\n'.join(meta)

    #   print(f">» chatGPT: \n{meta}")
    #   await bot.send_message(chat_id=message.chat.id, text=f"{meta}")


if __name__ == '__main__':
    print("Starting...")
    executor.start_polling(dispatcher, skip_updates=True)
    print("Stopped")
