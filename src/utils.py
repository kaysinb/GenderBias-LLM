import yaml
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import concurrent.futures
from collections import defaultdict
from openai import RateLimitError
from typing import Any, Dict, List, Optional, Union, Tuple
from transformers import PreTrainedTokenizerBase
from openai import OpenAI


def read_config(path: str) -> Dict[str, Any]:
    """
    Reads a YAML configuration file from the specified path.
    """
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


class LocalLLMGenerator:
    """
    A generator class for creating stories using a local language model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        prompt_story_generation: str,
        max_length: int = 200,
        max_batch_size: int = 20,
    ) -> None:
        """
        Initializes the LocalLLMGenerator with the provided model, tokenizer, and prompt.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_story_generation = prompt_story_generation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length
        self.max_batch_size = max_batch_size

    def generate_story(self, profession: str, n_samples: int) -> List[str]:
        """
        Generates a specified number of stories for a given profession.
        """
        prompt_story_generation_filled = self.prompt_story_generation.format(profession=profession)
        inputs = self.tokenizer(prompt_story_generation_filled, return_tensors="pt").to(self.device)
        all_outputs = []

        for i in range(0, n_samples, self.max_batch_size):
            batch_size = min(self.max_batch_size, n_samples - i)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=self.max_length, num_return_sequences=batch_size)

            prompt_len = len(prompt_story_generation_filled)
            story_response = [
                self.tokenizer.decode(outputs[i], skip_special_tokens=True)[prompt_len:] for i in range(batch_size)
            ]
            all_outputs.extend(story_response)

        return all_outputs


class BiasEvaluator:
    """
    A class to evaluate gender bias in generated stories using OpenAI's GPT models.
    """

    def __init__(
        self,
        openai_client: OpenAI,
        gpt_model_to_check_gender: str,
        prompt_gender_detection: str,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Initializes the BiasEvaluator with the provided OpenAI client and configurations.
        """
        self.openai_client = openai_client
        self.gpt_model_to_check_gender = gpt_model_to_check_gender
        self.prompt_gender_detection = prompt_gender_detection
        self.save_path = save_path

    def check_gender(self, profession: str, story_response: str) -> Union[str, None]:
        """
        Checks the gender representation in a given story response.
        """
        prompt_filled = self.prompt_gender_detection.format(profession=profession, story_response=story_response)
        chat_completion = self.openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_filled,
                }
            ],
            model=self.gpt_model_to_check_gender,
        )
        return chat_completion.choices[0].message.content

    def process_profession(
        self, profession: str, n_samples: int, generate_story_function: Any
    ) -> Tuple[str, List[str]]:
        """
        Processes a single profession by generating stories and evaluating their gender bias.
        """
        file_path = None
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            file_path = os.path.join(self.save_path, f"{profession}.csv")

        if file_path is not None and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            story_responses = df["story_response"].tolist()
            print(f"Loaded {len(story_responses)} stories for {profession}")
        else:
            print(f"Generating {n_samples} stories for {profession}")
            story_responses = generate_story_function(profession, n_samples)
            if file_path is not None:
                df_dict = {
                    "story_response": story_responses,
                    "gender": [None] * len(story_responses),
                }
                df = pd.DataFrame(df_dict)
                df.to_csv(file_path, index=False)

        print(f"Evaluating {profession}")
        fails_counter = 0
        counter = 0
        partial_results = []
        while counter < len(story_responses):
            try:
                story_response = story_responses[counter]
                gender_response = self.check_gender(profession, story_response)
                if gender_response is not None:
                    gender_response = gender_response.strip().lower()
                    partial_results.append(gender_response)
                    counter += 1
                    fails_counter = 0
                else:
                    fails_counter += 1
                    if fails_counter > 5:
                        print(f"Failed to get gender response for {profession}. Skipping...")
                        partial_results.append("ns")
                        counter += 1
                        fails_counter = 0
            except RateLimitError:
                print(f"Rate limit error for {profession}. Retrying...")
                time.sleep(5)

        if file_path is not None:
            df = pd.read_csv(file_path)
            df["gender"] = partial_results
            df.to_csv(file_path, index=False)
        return profession, partial_results


def concurrent_bias_evaluation(
    professions: List[str],
    n_samples: int,
    process_profession_function: Any,
    generate_story_function: Any,
    max_workers: int = 6,
) -> Dict[str, List[str]]:
    """
    Evaluates gender bias concurrently across multiple professions.
    """
    result_dict = defaultdict(list)
    max_workers = min(len(professions), max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit one task per profession
        futures = [
            executor.submit(process_profession_function, profession, n_samples, generate_story_function)
            for profession in professions
        ]

        # Gather results as they are completed
        for future in concurrent.futures.as_completed(futures):
            profession, partial_results = future.result()
            result_dict[profession].extend(partial_results)

    return result_dict


def plot_gender_distribution(
    profession_distribution_dict: Dict[str, List[str]], path_to_save: Optional[str] = None, show_plot: bool = True
) -> None:
    """
    Plots the gender distribution of professions as a stacked bar chart.
    """
    professions = sorted(list(profession_distribution_dict.keys()))
    male_counts = [profession_distribution_dict[prof].count("male") for prof in professions]
    female_counts = [profession_distribution_dict[prof].count("female") for prof in professions]
    neutral_counts = [profession_distribution_dict[prof].count("neutral") for prof in professions]

    n_samples = [i + j + k for i, j, k in zip(male_counts, female_counts, neutral_counts)]
    male_percent = [(count / n_sample) * 100 for count, n_sample in zip(male_counts, n_samples)]
    female_percent = [(count / n_sample) * 100 for count, n_sample in zip(female_counts, n_samples)]
    neutral_percent = [(count / n_sample) * 100 for count, n_sample in zip(neutral_counts, n_samples)]

    x = np.arange(len(professions))
    width = 0.6
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.bar(x, male_percent, width, label="male")
    ax.bar(x, female_percent, width, bottom=male_percent, label="female")
    ax.bar(x, neutral_percent, width, bottom=np.array(male_percent) + np.array(female_percent), label="neutral")

    fontsize = 12
    for i in range(len(professions)):
        if male_percent[i] > 0:
            ax.text(
                x[i],
                male_percent[i] / 2,
                f"{male_percent[i]:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=fontsize,
            )
        if female_percent[i] > 0:
            ax.text(
                x[i],
                male_percent[i] + female_percent[i] / 2,
                f"{female_percent[i]:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=fontsize,
            )
        if neutral_percent[i] > 0:
            ax.text(
                x[i],
                male_percent[i] + female_percent[i] + neutral_percent[i] / 2,
                f"{neutral_percent[i]:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=fontsize,
            )

    ax.set_xlabel("Profession")
    ax.set_ylabel("Share (%)")
    ax.set_title("Gender Distribution by Profession")
    ax.set_xticks(x)
    ax.set_xticklabels(professions, rotation=0, ha="center")
    ax.set_ylim(0, 105)
    ax.legend()

    plt.tight_layout()

    if path_to_save:
        df_dict = {
            "profession": professions,
            "male_count": male_counts,
            "female_count": female_counts,
            "neutral_count": neutral_counts,
            "male_percent": male_percent,
            "female_percent": female_percent,
            "neutral_percent": neutral_percent,
        }
        df = pd.DataFrame(df_dict)
        df.to_csv(path_to_save + ".csv", index=False)

        plt.savefig(path_to_save + ".png", dpi=300)
    if show_plot:
        plt.show()
