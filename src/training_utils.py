import torch
import torch.nn as nn
import numpy as np
from transformers import Trainer
from collections import defaultdict


class GenderLossTrainer(Trainer):
    """
    A custom trainer that uses `gender_loss` in the standard compute_loss flow.
    """

    def __init__(self, *args, **kwargs):
        self.p_total_logs = []
        self.standard_loss_logs = []
        self.gender_loss_logs = []
        self.gender_ds_extra_id = kwargs.pop("gender_ds_extra_id", -777)
        self.lambda_gender = kwargs.pop("lambda_gender", 1.0)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, **kwargs):
        """
        Overrides the default compute_loss to incorporate your custom `gender_loss`.
        Accepts arbitrary keyword arguments to maintain compatibility.
        """
        inputs_standard = defaultdict(list)
        inputs_gender = defaultdict(list)
        for inp, lb, msk in zip(inputs["input_ids"], inputs["labels"], inputs["attention_mask"]):
            if len(lb) > 3 and lb[3] == self.gender_ds_extra_id:
                inputs_gender["input_ids"].append(inp)
                inputs_gender["labels"].append(lb)
                inputs_gender["attention_mask"].append(msk)
            else:
                inputs_standard["input_ids"].append(inp)
                inputs_standard["labels"].append(lb)
                inputs_standard["attention_mask"].append(msk)

        loss_standard = 0
        if len(inputs_standard["input_ids"]) > 0:
            stacked_standard = {key: torch.stack(value) for key, value in inputs_standard.items()}
            loss_standard = super().compute_loss(model, stacked_standard, return_outputs=False, **kwargs)
            if isinstance(loss_standard, tuple):
                loss_standard = loss_standard[0]
            self.standard_loss_logs.append(loss_standard.item())

        loss_gender = 0
        if len(inputs_gender["input_ids"]) > 0:
            stacked_gender = {key: torch.stack(value) for key, value in inputs_gender.items()}
            loss_gender = self.custom_compute_loss(model, stacked_gender, return_outputs=False, **kwargs)
            if isinstance(loss_gender, tuple):
                loss_gender = loss_gender[0]
            self.gender_loss_logs.append(loss_gender.item())

        loss = loss_standard + self.lambda_gender * loss_gender

        if self.should_log():
            self.log({"p_total": np.mean(self.p_total_logs).item()})
            self.p_total_logs = []
            self.log({"standard_loss": np.mean(self.standard_loss_logs).item()})
            self.standard_loss_logs = []
            self.log({"gender_loss": np.mean(self.gender_loss_logs).item()})
            self.gender_loss_logs = []

        return loss

    def custom_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # Save past state if it exists (used by Trainer for some models)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Instead of the HF label_smoother, call your custom function:
        loss = self.gender_loss(model=model, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def gender_loss(self, model, input_ids, attention_mask, labels):
        """
        Compute the difference between logP("he/his...") and logP("she/her...") on the next token.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        labels = torch.tensor(labels)

        # Each element lengths[i] indicates the position in the sequence
        # where we want to compare pronouns
        next_token_logits = logits[torch.arange(logits.size(0)), labels[:, 2] - 1, :]  # [batch_size, vocab_size]

        probs = nn.functional.softmax(next_token_logits, dim=-1)
        # labels: shape [batch_size, 2] => [male_id, female_id]
        ids_male = labels[:, 0]  # e.g. token ID for "he", [batch_size]
        ids_female = labels[:, 1]  # e.g. token ID for "she", [batch_size]

        p_male = probs[torch.arange(probs.size(0)), ids_male]  # [batch_size]
        p_female = probs[torch.arange(probs.size(0)), ids_female]  # [batch_size]

        diff_loss = ((p_male - p_female) / (p_male + p_female + 1e-6)) ** 2
        self.p_total_logs.append((p_male + p_female).mean().item())

        return diff_loss.mean()

    def should_log(self):
        """
        Determines whether logging should occur based on the current step and logging_steps.
        """
        # Access the global step from the Trainer's state
        global_step = self.state.global_step
        return self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0
