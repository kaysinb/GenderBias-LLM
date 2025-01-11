import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import concurrent.futures
from collections import defaultdict


def read_config(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


class LocalLLMGenerator:
    def __init__(self, model, tokenizer, prompt_story_generation, max_length=200):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_story_generation = prompt_story_generation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

    def generate_story(self, profession):
        prompt_story_generation_filled = self.prompt_story_generation.format(profession=profession)
        inputs = self.tokenizer(prompt_story_generation_filled, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=self.max_length, num_return_sequences=1)

        prompt_len = len(prompt_story_generation_filled)
        story_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[prompt_len:]
        return story_response


class BiasEvaluator:
    def __init__(self, openai_client, gpt_model_to_check_gender, prompt_gender_detection):
        self.openai_client = openai_client
        self.gpt_model_to_check_gender = gpt_model_to_check_gender
        self.prompt_gender_detection = prompt_gender_detection

    def check_gender(self, profession, story_response):
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

    def process_profession(self, profession, n_samples, generate_story_function):
        print(f"Evaluating {profession}")
        partial_results = []
        max_iterations = int(n_samples * 1.5)

        for _ in range(max_iterations):
            story_response = generate_story_function(profession)
            gender_response = self.check_gender(profession, story_response)

            if gender_response in ["man", "woman", "dta"]:
                partial_results.append(gender_response)

            if len(partial_results) >= n_samples:
                break

        return profession, partial_results


def concurrent_bias_evaluation(professions, n_samples, process_profession_function, generate_story_function):
    result_dict = defaultdict(list)
    max_workers = min(len(professions), 6)
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


def plot_gender_distribution(profession_distribution_dict, path_to_save=None):
    """
    Plot the gender distribution of professions.
    """
    professions = sorted(list(profession_distribution_dict.keys()))
    men_counts = [profession_distribution_dict[prof].count("man") for prof in professions]
    women_counts = [profession_distribution_dict[prof].count("woman") for prof in professions]
    dta_counts = [profession_distribution_dict[prof].count("dta") for prof in professions]

    # Calculate percentages
    n_samples = max([len(profession_distribution_dict[prof]) for prof in professions])
    men_percent = [(count / n_samples) * 100 for count in men_counts]
    women_percent = [(count / n_samples) * 100 for count in women_counts]
    dta_percent = [(count / n_samples) * 100 for count in dta_counts]

    # Plotting
    x = np.arange(len(professions))
    width = 0.6  # Increased width for smaller distance between bars

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.bar(x, men_percent, width, label="man")
    ax.bar(x, women_percent, width, bottom=men_percent, label="woman")
    ax.bar(x, dta_percent, width, bottom=np.array(men_percent) + np.array(women_percent), label="dta")

    fontsize = 12

    # Adding percentage labels on top of each segment
    for i in range(len(professions)):
        if men_percent[i] > 0:
            ax.text(
                x[i],
                men_percent[i] / 2,
                f"{men_percent[i]:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=fontsize,
            )
        if women_percent[i] > 0:
            ax.text(
                x[i],
                men_percent[i] + women_percent[i] / 2,
                f"{women_percent[i]:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=fontsize,
            )
        if dta_percent[i] > 0:
            ax.text(
                x[i],
                men_percent[i] + women_percent[i] + dta_percent[i] / 2,
                f"{dta_percent[i]:.1f}%",
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
        plt.savefig(path_to_save, dpi=300)
    else:
        plt.show()
