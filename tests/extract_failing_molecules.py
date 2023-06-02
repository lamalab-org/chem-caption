from typing import List, Tuple

import pandas as pd
from bs4 import BeautifulSoup
import os

PAGE = os.path.join(os.getcwd().split("tests")[0], "Test Results - .html")


def extract_code_content(page: str) -> List[str]:
    """
    Extract relevant errors.

    Args:
        page (str): HTML page of interest.

    Returns:
        errors (List[str]): List of specific errors.
    """
    response = open(page, "r").read()
    soup = BeautifulSoup(response, "lxml")

    errors = soup.findAll("span", class_="stderr")

    errors = [
        error.text.split("!=")
        for error in errors
        if error.text.__contains__("rotable")
        or error.text.__contains__("hdonor")
        or error.text.__contains__("hacceptor")
    ]

    return errors


def split_string(string_list: List[str]) -> Tuple[str, str, int, int]:
    """
    Split the strings in a list and returns information.

    Args:
        string_list (List): List to split.

    Returns:
        test_type (str) Type of test the string records.
        string (str): Molecular representation string.
        actual (int): Actual value returned from test.
        expected (int): Value test was expected to return.
    """
    first, second = string_list[:2]
    first = first.strip()
    second = second.strip()

    if "rotable" in first:
        test_type = "rotable"
    elif "hdonor" in first:
        test_type = "donor"
    elif "hacceptor" in first:
        test_type = "acceptor"
    else:
        test_type = ""

    actual = int(first.split("array")[-1][3:-3])
    expected = int(second.split("test_input")[0][7:-2])
    string = second.split("test_input")[1].split("expected")[0].replace(" = ", "")
    string = string.replace("'", "").replace(", ", "")

    return test_type, string, actual, expected


def return_extracted_error_data(page: str, string_type: str = "smiles") -> None:
    """Put together the final extraction pipeline.

    Args:
        page (str): HTML page name.
        string_type (str): Type of representation the page contains errors about.

    Returns:
        None
    """
    errors = extract_code_content(page)
    errors = [split_string(error) for error in errors]

    df = pd.DataFrame(columns=["Type", string_type.upper(), "Actual", "Expected"])

    test_type = list(map(lambda x: x[0], errors))
    strings = list(map(lambda x: x[1], errors))
    actual = list(map(lambda x: x[2], errors))
    expected = list(map(lambda x: x[3], errors))

    df["Type"] = test_type
    df[f"{string_type.upper()}"] = strings
    df["Actual"] = actual
    df["Expected"] = expected

    df.to_csv(f"data/error_throwing_{string_type}.csv", index=False)

    return

if __name__ == "__main__":
    string_type = "inchi"
    return_extracted_error_data(page=PAGE, string_type=string_type)
