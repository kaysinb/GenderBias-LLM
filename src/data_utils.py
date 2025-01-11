from .utils import read_config
from datasets import Dataset, DatasetDict


def txt_file_to_list(file_path):
    with open(file_path, "r") as file:
        return file.read().splitlines()


def build_context_pronoun_list(templates, professions, pronoun_choices, prompt_beginning, space_symbol):
    """
    For each template + profession, build a row with:
      - 'context'
      - 'pronoun_list'
    Returns two parallel lists (contexts, pronoun_lists).
    """
    contexts = []
    pronoun_lists = []

    for template in templates:
        # 1. Find the {pronoun_...} placeholder
        pronoun_idx = template.find("{pronoun_")
        if pronoun_idx == -1:
            # If no pronoun placeholder found, skip or handle differently
            continue

        pronoun_start = pronoun_idx + 1
        pronoun_end = pronoun_idx + 10
        pronoun_key = template[pronoun_start:pronoun_end]  # e.g. 'pronoun_m'

        # 2. Remove that placeholder from the template for direct formatting
        #    This is just one approach; you might need more logic if you have more placeholders.
        template_stripped = template[:pronoun_idx].strip()

        # 3. Build context & pronoun_list for each profession
        for profession in professions:
            # Build the text
            prof_lower = profession.lower()
            context_text = prompt_beginning.format(profession=prof_lower) + template_stripped.format(
                profession=prof_lower
            )

            # Lookup the pronoun list
            pronoun_list = pronoun_choices[pronoun_key]

            # Capitalize if the context ends with a period
            if context_text.endswith("."):
                pronoun_list = [p.capitalize() for p in pronoun_list]

            # Prepend space symbol
            pronoun_list = [space_symbol + p for p in pronoun_list]

            contexts.append(context_text)
            pronoun_lists.append(pronoun_list)

    return contexts, pronoun_lists


def prepare_dataset(
    dataset_config_path,
    template_path,
    train_share=0.8,
    validation_share=0.1,
    space_symbol="Ġ",
    print_dataset_info=False,
):
    """
    Reads config & templates, splits them, then builds a DatasetDict
    with train/validation/test. Each split contains columns:
      - "context"
      - "pronoun_list"
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
