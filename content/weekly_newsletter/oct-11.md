---
title: "Weekly Newsletter - Oct 11"
date: 2024-10-11T11:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["first"]
author: "Prakyath Kantharaju"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Desc Text."
canonicalURL: "https://prakyathk.com/weekly_newsletter/oct-11/"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/prakyathkantharaju/personal_blog/content/weekly_newsletter/oct-11.md"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---
# Weekly Newsletter - October 11, 2024

## AI Breakthroughs Take Center Stage in Nobel Prizes

This year's Nobel Prizes have put a spotlight on AI's transformative impact across scientific disciplines. The Physics prize, awarded to Geoffrey Hinton and John Hopfield, celebrates their groundbreaking work on artificial neural networks – the very foundation of today's deep learning revolution.

Meanwhile, the Chemistry prize went to Demis Hassabis, John Jumper, and David Baker for their development of AlphaFold. This AI system has revolutionized protein structure prediction, opening new frontiers in drug discovery and biotechnology.

![Nobel Prizes Physics](/oct-11/GZW9tFIW0AAMOPB.jpeg)
![Nobel Prizes Chemistry](/oct-11/GZcHPXJWoAAzboh.jpeg)



## ARIA: A New Open-Source AI Powerhouse

The AI community is buzzing about ARIA, a newly released open-source multimodal model. With its Apache 2.0 license and impressive capabilities, ARIA is poised to shake up the field:

- 25.3 billion parameters (3.9 billion active)
- 64K token context window
- Handles text, images, audio, and video
- Innovative Mixture-of-Experts (MoE) architecture
- Early benchmarks show it outperforming several established models

![ARIA](/oct-11/2024-10-11_10-40.png)

## OpenAI's o1 - Meta prompt release

[Link to the Playground meta prompt guide](https://platform.openai.com/docs/guides/prompt-generation)

OpenAI Playground's New Generate Button: Streamlining AI Development
The Playground has introduced an exciting new feature: the Generate button. This tool is designed to simplify the process of creating prompts, functions, and schemas. Here's how it works:

Prompts: Uses meta-prompts incorporating best practices to generate or improve prompts.
Schemas: Employs meta-schemas to produce valid JSON and function syntax.

The Generate button uses two main approaches:

1. **Meta-prompts** for prompt generation and improvement
2. **Meta-schemas** for producing valid JSON and function syntax

While currently relying on these methods, there are plans to potentially integrate more advanced techniques like DSPy and "Gradient Descent" in the future.

**Key features:**

- Generates prompts and schemas from task descriptions
- Uses specific meta-prompts for different output types (e.g., audio)


## Fine-tuning LLMs: Dynamic Reasoning with DOTS

Researchers continue to push the boundaries of LLM fine-tuning, with a recent breakthrough in enhancing reasoning capabilities. A new paper introduces DOTS (Dynamic Optimal reasoning Trajectories Search), an innovative approach to fine-tuning LLMs for improved reasoning:

Key features of DOTS:
- Tailors reasoning strategies to specific questions and the LLM's capabilities
- Defines atomic reasoning action modules that can be combined into various trajectories
- Searches for optimal action trajectories through iterative exploration and evaluation
- Trains LLMs to plan reasoning trajectories for unseen questions

The DOTS method offers two learning paradigms:
1. Fine-tuning an external LLM as a planner to guide the task-solving LLM
2. Directly fine-tuning the task-solving LLM with internalized reasoning action planning

Results from experiments across eight reasoning tasks show that DOTS consistently outperforms static reasoning techniques and vanilla instruction tuning. Notably, this approach enables LLMs to adjust their computation based on problem complexity, allocating deeper thinking to more challenging problems.

This development addresses longstanding challenges in LLM fine-tuning, such as:
- Overcoming the limitations of static, predefined reasoning actions
- Adapting to the specific characteristics of each question
- Optimizing performance for the inherent capabilities of different LLMs

As the field continues to evolve, approaches like DOTS promise to significantly enhance the reasoning capabilities of large language models, opening new possibilities for AI applications across various domains.
![DOTS](/oct-11/2024-10-11_10-51.png)

## Quantization: Making AI More Accessible

The push for efficient AI is driving innovative quantization techniques:

- BitNet (Microsoft): Uses 1-bit weights and quantized activations [link to the paper](https://arxiv.org/abs/2310.11453)
- AdderLM: Replaces floating-point multiplication with integer addition [link to the paper](https://arxiv.org/pdf/2410.00907)

These methods aim to reduce computational resources while maintaining performance, potentially bringing AI to more resource-constrained devices.

## Tools and Frameworks: Empowering Developers

New tools are streamlining AI development workflows:

- Aider v0.59.0: Enhances shell-style auto-complete and YAML config, [link to the repo](https://github.com/aider-ai/aider)
- OpenRouter: Improves LLM routing and API management [link to the website](https://openrouter.ai/)

## Recent Papers of Interest

1. "Benchmarking Agentic Workflow Generation" [link to the paper](https://arxiv.org/abs/2410.07869)
   - Introduces WorFBench and WorFEval for evaluating LLM workflow generation
   - Reveals gaps between sequence and graph planning capabilities in LLMs

2. "Towards Self-Improvement of LLMs via MCTS" [link to the paper](https://arxiv.org/abs/2410.06508)
   - Proposes AlphaLLM-CPL for more effective MCTS behavior distillation
   - Shows promising results in improving LLM reasoning capabilities

3. "Named Clinical Entity Recognition Benchmark" [link to the paper](https://arxiv.org/abs/2410.05046)
   - Establishes a standardized platform for assessing language models in healthcare NLP tasks
   - Utilizes OMOP Common Data Model for consistency across datasets











