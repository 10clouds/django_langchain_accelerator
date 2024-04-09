import argparse
import logging
import pathlib
from datetime import datetime

from file_utils import read_file
from llm_utils import create_chains, get_retriever

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Automate Django app creation with task description from a text file."
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task_file_path",
        type=str,
        help="Path to the text file containing the task description.",
    )
    parser.add_argument(
        "-p",
        "--path",
        action="append",
        dest="paths",
        type=str,
        help="Path to the reference projects.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=str,
        help="Path to the output directory.",
        default="result",
        required=False,
    )

    return parser.parse_args()


def generate_django_project() -> None:
    """
    Generate a Django project based on the task description and reference projects.
    """
    args = parse_args()

    description = read_file(args.task_file_path)
    if description is None:
        logger.error("Task file not found.")
        return None

    inputs = {"user_story": description}

    project_paths = [pathlib.Path(path) for path in args.paths]
    retriever = get_retriever(project_paths)

    root_output = pathlib.Path(args.output_path)
    #
    execution_output_path = (
        root_output / f"{datetime.now().strftime('%Y/%m/%d/%H/%M/')}"
    )

    final_chain = create_chains(retriever, execution_output_path)
    final_chain.invoke(inputs)


if __name__ == "__main__":
    generate_django_project()
