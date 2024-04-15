import pathlib
from operator import itemgetter

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.messages import get_buffer_string, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from file_utils import create_files, get_django_files_contents

# Prompt templates
DJANGO_DEVELOPER_PROMPT_TEMPLATE = """ You are python Django Senior Developer.\n
You know how to write a code in Django files.\n
Follow up user_story, rephrase the follow up user_story to be a standalone user_story, in its original language.\n

Follow Up Input: {user_story}\n
Standalone user_story:"""

GENERATE_APPS_AND_MODELS_FILES_PROMPT_TEMPLATE = """
\nThis is how to write an Python Django apps:
{context}\ns
\n
Remember: always give a code in "```python" markdown.\n
Code need to be in english - always!\n
In "```python" markdown always start with line "# <path/to/file.py>" line.\n
Root folder of application is "app", all other app folders you can named by your own.\n
Create a two files per each app: "apps.py" and "models.py" django app files based only on the following User Story:\n
\n
User Story: \n{user_story}
"""

GENERATE_ADMIN_AND_FILTER_FILES_PROMPT_TEMPLATE = """
\nGiven the existing Django app files `apps.py` and `models.py`, generate the corresponding `admin.py` and `filters.py` files.
\n
Every content file have information about path of the file, at the begin, example: # ../this/is/file/path.py \n
Here are apps.py and models.py content: \n
{first_pair}
\n
Generate admin.py and filters.py files for all classes in all models.py files:
"""

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
DJANGO_DEVELOPER_PROMPT = PromptTemplate.from_template(DJANGO_DEVELOPER_PROMPT_TEMPLATE)
GENERATE_APPS_AND_MODELS_FILES_PROMPT = ChatPromptTemplate.from_template(
    GENERATE_APPS_AND_MODELS_FILES_PROMPT_TEMPLATE
)
GENERATE_ADMIN_AND_FILTER_FILES_PROMPT = ChatPromptTemplate.from_template(
    GENERATE_ADMIN_AND_FILTER_FILES_PROMPT_TEMPLATE
)


def _combine_documents(
    docs: list[Document],
    document_prompt: PromptTemplate = DEFAULT_DOCUMENT_PROMPT,
    document_separator: str = "\n\n",
) -> str:
    """
    Combines multiple documents into a single document.
    :param docs: documents to combine.
    :param document_prompt: document prompt.
    :param document_separator: document separator.
    :return:
    """
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def get_retriever(paths: list[pathlib.Path]) -> VectorStoreRetriever:
    """
    Create a FAISS vector store retriever from files in the given paths.
    :param paths: paths to the reference projects.
    :return: FAISS retriever.
    """
    vectorstore = FAISS.from_texts(
        get_django_files_contents(paths),
        embedding=OpenAIEmbeddings(),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return retriever


def create_chains(
    retriever: VectorStoreRetriever, execution_output_path: pathlib.Path
) -> RunnableSerializable:
    """
    Create a chain of runnables for the Django app creation.
    :param retriever: vector store retriever.
    :param execution_output_path: root output path for the generated files.
    :return: runnable chain.
    """
    memory = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="user_story"
    )

    # chat history if we want to use it
    _ = RunnableLambda(memory.load_memory_variables) | itemgetter("history")

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables)
        | itemgetter("history"),
    )

    standalone_user_story: dict[str, RunnableSerializable] = {
        "standalone_user_story": {
            "user_story": lambda x: x["user_story"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | DJANGO_DEVELOPER_PROMPT
        | ChatOpenAI()
        | StrOutputParser(),
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_user_story") | retriever,
        "user_story": lambda x: x["standalone_user_story"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "user_story": itemgetter("user_story"),
    }

    def _create_files_from_answer(content: AIMessage) -> str:
        return create_files(content, execution_output_path)

    generate_first_pair = {
        "first_pair": final_inputs
        | GENERATE_APPS_AND_MODELS_FILES_PROMPT
        | ChatOpenAI()
        | _create_files_from_answer,
        "docs": itemgetter("docs"),
    }

    generate_second_pair = {
        "second_pair": GENERATE_ADMIN_AND_FILTER_FILES_PROMPT
        | ChatOpenAI()
        | _create_files_from_answer,
        "docs": itemgetter("docs"),
    }

    # And now we put it all together!
    final_chain: RunnableSerializable = (
        loaded_memory
        | standalone_user_story
        | retrieved_documents
        | generate_first_pair
        | generate_second_pair
    )
    return final_chain
