import torch
import numpy as np
from typing import Any, Dict, Iterable
from transformers import PreTrainedTokenizerBase


def evaluate_copa(
    model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, dataset: Iterable[Dict[str, Any]]
) -> float:
    """
    Evaluates the model's accuracy on the COPA (Choice of Plausible Alternatives) dataset
    by comparing the model's predictions against the correct labels.
    """
    correct = 0
    total = 0

    for sample in dataset:
        premise = sample["premise"]
        question = sample["question"]
        choice1 = sample["choice1"]
        choice2 = sample["choice2"]
        correct_label = sample["label"]

        scores = []
        for choice in [choice1, choice2]:
            prompt = f"Premise: {premise}\n" f"Question: What is the most likely {question}?\n" f"Choice: {choice}\n"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()

            scores.append(-loss)

        pred_label = int(np.argmax(scores))
        if pred_label == correct_label:
            correct += 1
        total += 1

    return correct / total


def compute_choice_score(
    model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, prompt: str, choice_text: str
) -> float:
    """
    Calculates the negative log-likelihood (as a score) of a given choice text based on a prompt. This score indicates
    how likely the model considers the choice text as a continuation of the prompt.
    """
    device = next(model.parameters()).device

    full_text = prompt + " " + choice_text

    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss.item()

    return -loss


def evaluate_piqa(
    model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, dataset: Iterable[Dict[str, Any]]
) -> float:
    """
    Assesses the model's performance on the PIQA (Physical Interaction: Question Answering) dataset
    by determining the accuracy of the model's selected solutions.
    """
    correct = 0
    total = 0

    for example in dataset:
        goal = example["goal"]
        sol1 = example["sol1"]
        sol2 = example["sol2"]
        label = example["label"]

        prompt = f"Question: {goal}\nAnswer:"

        score_sol1 = compute_choice_score(model, tokenizer, prompt, sol1)
        score_sol2 = compute_choice_score(model, tokenizer, prompt, sol2)

        pred_label = 0 if score_sol1 > score_sol2 else 1

        if pred_label == label:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy


def evaluate_lambada_next_token_accuracy(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Iterable[Dict[str, Any]],
) -> float:
    """
    Measures the model's ability to predict the next token in the LAMBADA dataset by comparing
    the model's predicted token against the actual next token in the text.
    """
    device = next(model.parameters()).device

    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        text = example["text"].strip()
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue

        context_ids = tokens[:-1]
        target_id = tokens[-1]

        context_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        target_id = torch.tensor([target_id], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(context_ids)
            logits_last = outputs.logits[:, -1, :]
            pred_id = torch.argmax(logits_last, dim=-1)

        if pred_id.item() == target_id.item():
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy
