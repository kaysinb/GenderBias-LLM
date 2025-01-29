# GenderBias-LLM

This repository was created as part of the final project for the BlueDot Impact AI Alignment course, focusing on the investigation and mitigation of gender bias in Large Language Models. The results of this project are presented in the [accompanying article](https://sites.google.com/view/gender-bias-llm) and encompass all code, datasets, and evaluation metrics.

Below is an overview of the resources included in this repository:

## Pre-Generated Datasets
- **Template-Based Dataset**: A collection of profession-oriented sentences systematically paired with male and female pronouns.
- **Story-Based Dataset**: Synthetically generated short stories covering various professions, each presented in both male-oriented and female-oriented versions.

## Fine-Tuning Scripts
- Implements a specialized gender-aware loss function designed to balance male and female pronoun probabilities.
- Provides configurations for adapting either MLP or attention layers, based on insights into where bias typically arises.

## Evaluation Tools
- Scripts for measuring pronoun probability disparities, capturing metrics such as the sum of male/female probabilities and the absolute difference between them.
- Methods for classifying short stories into male-, female-, or neutral-oriented narratives, highlighting biases in open-ended text generation.