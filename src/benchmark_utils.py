import torch
import numpy as np


def evaluate_copa(model, tokenizer, dataset):
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


def compute_choice_score(model, tokenizer, prompt, choice_text):
    """
    Computes negative log-likelihood of `choice_text` given the `prompt`.
    We'll return *log-prob* (the higher, the more likely).
    """
    device = next(model.parameters()).device

    # Combine the prompt and choice
    full_text = prompt + " " + choice_text

    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # We'll use the model's causal LM head to get the total loss
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,  # computing cross-entropy over the entire sequence
        )
        loss = outputs.loss.item()  # average cross-entropy over all tokens

    # Return negative loss as "score"
    # A higher score => lower cross-entropy => better fit
    return -loss


def evaluate_piqa(model, tokenizer, dataset):
    correct = 0
    total = 0

    for example in dataset:
        goal = example["goal"]
        sol1 = example["sol1"]
        sol2 = example["sol2"]
        label = example["label"]  # 0 or 1

        # Construct a simple prompt. For instance:
        prompt = f"Question: {goal}\nAnswer:"

        # Score each solution
        score_sol1 = compute_choice_score(model, tokenizer, prompt, sol1)
        score_sol2 = compute_choice_score(model, tokenizer, prompt, sol2)

        # Predict choice: whichever has higher log-prob
        pred_label = 0 if score_sol1 > score_sol2 else 1

        if pred_label == label:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy


def evaluate_lambada_next_token_accuracy(model, tokenizer, dataset, max_eval_samples=None):
    """
    For each example, we:
      1. Tokenize the entire text.
      2. Separate the last token as the 'target'.
      3. Feed the preceding tokens (context) into the model.
      4. Let the model predict the next token (top-1).
      5. Check if it matches the actual last token.
    Returns accuracy (#correct / #total).
    """
    device = next(model.parameters()).device

    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        text = example["text"].strip()
        # Convert text to token IDs
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            # If the text is too short (only 1 token), skip
            continue

        context_ids = tokens[:-1]  # all but last token
        target_id = tokens[-1]  # last token

        # Convert to tensors
        context_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        target_id = torch.tensor([target_id], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(context_ids)
            # outputs.logits shape: (batch, seq_len, vocab_size)
            # We want the last hidden state from the final token in context
            logits_last = outputs.logits[:, -1, :]  # shape: (batch=1, vocab_size)
            pred_id = torch.argmax(logits_last, dim=-1)  # top-1 token index

        if pred_id.item() == target_id.item():
            correct += 1
        total += 1

        # Optionally limit number of evaluated samples
        if max_eval_samples is not None and (i + 1) >= max_eval_samples:
            break

    accuracy = correct / total if total > 0 else 0.0
    return accuracy
