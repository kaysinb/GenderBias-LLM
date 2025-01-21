from .utils import read_config
from datasets import Dataset, DatasetDict
import os
from typing import List, Dict, Tuple, Optional


def txt_file_to_list(file_path: str) -> List[str]:
    """
    Reads a text file and returns its contents as a list of lines.
    """
    with open(file_path, "r") as file:
        return file.read().splitlines()


def build_context_pronoun_list(
    templates: List[str],
    professions: List[str],
    pronoun_choices: Dict[str, List[str]],
    prompt_beginning: str,
    space_symbol: str,
) -> Tuple[List[str], List[List[str]]]:
    """
    Generates contexts and corresponding pronoun lists for each combination of template and profession.
    """
    contexts = []
    pronoun_lists = []

    for template in templates:
        pronoun_idx = template.find("{pronoun_")
        if pronoun_idx == -1:
            continue

        pronoun_start = pronoun_idx + 1
        pronoun_end = pronoun_idx + 10
        pronoun_key = template[pronoun_start:pronoun_end]

        template_stripped = template[:pronoun_idx].strip()

        for profession in professions:
            prof_lower = profession.lower()
            context_text = prompt_beginning.format(profession=prof_lower) + template_stripped.format(
                profession=prof_lower
            )

            pronoun_list = pronoun_choices[pronoun_key]

            if context_text.endswith("."):
                pronoun_list = [p.capitalize() for p in pronoun_list]

            pronoun_list = [space_symbol + p for p in pronoun_list]

            contexts.append(context_text)
            pronoun_lists.append(pronoun_list)

    return contexts, pronoun_lists


def prepare_dataset_gender(
    dataset_config_path: str,
    template_path: str,
    train_share: float = 0.8,
    validation_share: float = 0.1,
    space_symbol: str = "Ġ",
    print_dataset_info: bool = False,
    reduced_number_of_train_templates: Optional[int] = None,
) -> DatasetDict:
    """
    Creates a gender-specific dataset by reading configuration and template files, splitting the data into training,
    validation, and test sets, and organizing them into a DatasetDict.
    """
    dataset_config = read_config(dataset_config_path)
    templates = txt_file_to_list(template_path)

    train_professions = dataset_config["train_professions"]
    validation_professions = dataset_config["validation_professions"]
    test_professions = dataset_config["test_professions"]
    pronoun_choices = dataset_config["pronoun_choices"]
    prompt_beginning = dataset_config["prompt_beginning"]

    templates_num = len(templates)
    validation_beginning = int(templates_num * train_share)
    test_beginning = int(templates_num * (train_share + validation_share))

    train_templates = templates[:validation_beginning]
    if reduced_number_of_train_templates:
        train_templates = train_templates[:reduced_number_of_train_templates]
    validation_templates = templates[validation_beginning:test_beginning]
    test_templates = templates[test_beginning:]

    # Train
    train_contexts, train_pronoun_lists = build_context_pronoun_list(
        templates=train_templates,
        professions=train_professions,
        pronoun_choices=pronoun_choices,
        prompt_beginning=prompt_beginning,
        space_symbol=space_symbol,
    )
    train_dataset = Dataset.from_dict({"context": train_contexts, "pronoun_list": train_pronoun_lists})

    # Validation
    val_contexts, val_pronoun_lists = build_context_pronoun_list(
        templates=validation_templates,
        professions=validation_professions,
        pronoun_choices=pronoun_choices,
        prompt_beginning=prompt_beginning,
        space_symbol=space_symbol,
    )
    validation_dataset = Dataset.from_dict({"context": val_contexts, "pronoun_list": val_pronoun_lists})

    # Test
    test_contexts, test_pronoun_lists = build_context_pronoun_list(
        templates=test_templates,
        professions=test_professions,
        pronoun_choices=pronoun_choices,
        prompt_beginning=prompt_beginning,
        space_symbol=space_symbol,
    )
    test_dataset = Dataset.from_dict({"context": test_contexts, "pronoun_list": test_pronoun_lists})

    if print_dataset_info:
        print("Train dataset info:")
        print(f"    Templates: {len(train_templates)}")
        print(f"    Professions: {len(train_professions)}")
        print(f"    Dataset rows: {len(train_dataset)}")

        print("Validation dataset info:")
        print(f"    Templates: {len(validation_templates)}")
        print(f"    Professions: {len(validation_professions)}")
        print(f"    Dataset rows: {len(validation_dataset)}")

        print("Test dataset info:")
        print(f"    Templates: {len(test_templates)}")
        print(f"    Professions: {len(test_professions)}")
        print(f"    Dataset rows: {len(test_dataset)}")

    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset,
        }
    )

    return dataset_dict


def prepare_dataset_gender_stories(
    stories_path: str, reduced_number_of_stories_per_profession: Optional[int] = None
) -> DatasetDict:
    """
    Generates a dataset from story files, organizing instructions and responses based on professions.
    """
    file_paths = os.listdir(stories_path)
    instructions_list = []
    responses_list = []
    instruction_template = "Write a short story about the {profession}."
    for file_path in file_paths:
        with open(os.path.join(stories_path, file_path), "r") as f:
            stories = f.read().splitlines()
        stories = [story.strip() for story in stories if len(story) > 10]

        if reduced_number_of_stories_per_profession:
            stories = stories[:reduced_number_of_stories_per_profession]

        profession = file_path.split("_")[0].replace("-", " ")
        instructions_list.extend([instruction_template.format(profession=profession)] * len(stories))
        responses_list.extend(stories)

    dataset = Dataset.from_dict({"instruction": instructions_list, "response": responses_list})
    dataset_dict = DatasetDict({"train": dataset})
    return dataset_dict
