# django_langchain_accelerator


### Description

This project, django_langchain_accelerator, leverages a Language Model (LLM) to expedite the creation of Django projects. 
It generates necessary boilerplate code for Django projects, taking into account both the user's specifications 
and existing Django projects as a reference and context.
For this purpose, we will use [langchain](https://python.langchain.com/docs/get_started/introduction) 
with [OpenAI](https://python.langchain.com/docs/integrations/chat/openai/).


### Installation

Install poetry:

```bash
  pip install poetry
```

In root of the repository run the following command to install dependencies:

```bash
  poetry install --no-root
```


### Setup
In order for the code to work, you need to provide OpenAI API key. You can do this by setting the environment variable:

```bash
  export OPENAI_API_KEY=<your_openai_api_key>
```


### Usage

There is prepared example of usage in [`main.py`](main.py) file. To run it use:

```bash
  poetry run python main.py -t <file_with_user_story> -p <path_to_existing_project> -p <path_to_existing_project> -o <output_root_dir> 
```
where 
- `<file_with_user_story>` is a file with user story.
- `<path_to_existing_project>` is a path to existing Django project which will be used as a context for the generated code.
- `<output_root_dir>` is a path to the directory where the generated code will be saved. If not provided, the code will be saved under `result` subdirectory.


### Development

To install the linting and type checking tools, run the following command in the root of the repository:

```bash
  poetry install --only dev --no-root
```

To run the formatter execute:

```bash
  poetry run ruff format
```

To run the linter execute:

```bash
  poetry run ruff check
```

To run the type checker execute:

```bash
  poetry run mypy .
```
