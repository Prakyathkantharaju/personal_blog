---
title: "AI Weekly Newsletter (New open models) - Sep 27"
date: 2020-09-15T11:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["weekly_newsletter"]
author: "Me"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Highlights: OpenAI experiences significant leadership changes, Meta releases Llama 3 models with impressive benchmarks, Google unveils updated Gemini models with performance improvements and price reductions, AlphaChip transforms computer chip design using AI"
canonicalURL: "https://canonical.url/to/page"
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
    URL: "https://github.com/<path_to_repo>/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---




# AI Weekly Newsletter (New open models)

## Highlights

- OpenAI experiences significant leadership changes
- Meta releases Llama 3 models with impressive benchmarks
- Google unveils updated Gemini models with performance improvements and price reductions
- AlphaChip transforms computer chip design using AI

## News

### OpenAI Leadership Changes

- Top executives leaving OpenAI:
  - CTO Mira Murati
  - Chief Research Officer Bob McGrew
  - Research Leader Barret Zoph

### Llama 3.2 Release

- Meta releases new Llama 3.2 models:
  - 11B model comparable/slightly better than Claude Haiku
  - 90B model comparable/slightly better than GPT-4o-mini
  - New 128k-context 1B and 3B models competing with Gemma 2 and Phi 3.5
  - Tight on-device collaborations with Qualcomm, Mediatek, and Arm
  - [MMMU Benchmark Leaderboard](https://mmmu-benchmark.github.io/#leaderboard)

![Llama 3.2 Benchmark 1](/sept_27/Pasted%20image%2020240927101929.png)

![Llama 3.2 Benchmark 2](/sept_27/Pasted%20image%2020240927101947.png)

### Molmo Models by AI2

- Open-source multimodal models outperforming Llama 3.2 in vision tasks
- 7B and 72B model sizes (plus 7B MoE with 1B active params)
- Benchmarks above GPT-4V, Flash, etc.
- Human preference of 72B on par with top API models
- [Molmo Blog](https://molmo.allenai.org/blog)
- [Molmo Models on Hugging Face](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19)
- [Molmo Paper](https://arxiv.org/pdf/2409.17146)
### Molmo comparison
![Molmo Comparison](/sept_27/Pasted%20image%2020240927101904.png)

### PixMo Dataset

- High-quality dataset for captioning and supervised fine-tuning
- Created without using VLMs to generate data
- Includes dense captioning and supervised fine-tuning data
- Novel data collection methodology using spoken descriptions

### Google's Gemini 1.5 Updates

- Release of Gemini-1.5-Pro-002 and Gemini-1.5-Flash-002
- Significant improvements:
  - >50% reduced price on 1.5 Pro for prompts <128K
  - 2x higher rate limits on 1.5 Flash and ~3x higher on 1.5 Pro
  - 2x faster output and 3x lower latency
  - Updated default filter settings
- Performance improvements across various benchmarks
- [Google AI Studio](https://aistudio.google.com/app/prompts/new_chat?model=gemini-1.5-pro-002)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs/models/gemini)
This is how cost vs intelligent stands:
![Gemini Cost vs Intelligence](/sept_27/Pasted%20image%2020240927094928.png)
### AlphaChip

- Google's AI method for designing chip layouts
- Used in the last three generations of Google's Tensor Processing Unit (TPU)
- Generates superhuman chip layouts in hours instead of weeks or months
- [Nature Addendum](https://www.nature.com/articles/s41586-024-08032-5)
- [Pre-trained Checkpoint](https://github.com/google-research/circuit_training/?tab=readme-ov-file#PreTrainedModelCheckpoint)
- [DeepMind Blog Post](https://deepmind.google/discover/blog/how-alphachip-transformed-computer-chip-design/)

## Research

### Elo Uncovered: Robustness and Best Practices in Language Model Evaluation

- Study on the suitability of the Elo rating system for evaluating Large Language Models
- Explores reliability and transitivity axioms in LLM evaluation
- Findings offer guidelines for enhancing the reliability of LLM evaluation methods
- [Paper Link](https://arxiv.org/abs/2311.17295)

### Enhancing Structured-Data Retrieval with GraphRAG

- Introduces Structured-GraphRAG framework for improving information retrieval across structured datasets
- Utilizes multiple knowledge graphs to capture complex relationships between entities
- Case study on soccer data demonstrates improved query processing efficiency and reduced response times
- [Paper Link](https://arxiv.org/abs/2409.17580)

### HelloBench: Evaluating Long Text Generation Capabilities of Large Language Models

- Comprehensive benchmark for evaluating LLMs' performance in generating long text
- Categorizes tasks based on Bloom's Taxonomy
- Proposes HelloEval, a human-aligned evaluation method
- Experiments across 30 mainstream LLMs reveal limitations in long text generation capabilities
- [GitHub Repository](https://github.com/Quehry/HelloBench)

## Libraries

### crawl4ai

- GitHub repository for easy web scraping with LLM-friendly output formats
- Features:
  - Supports crawling multiple URLs simultaneously
  - Extracts media tags, links, and metadata
  - Custom hooks for authentication, headers, and page modifications
- [GitHub Repository](https://github.com/unclecode/crawl4ai)

## Good Reads

- [LLAMA vs GPT Comparison](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb)
- [Reverse Engineering OpenAI's O(1) Performance](https://www.interconnects.ai/p/reverse-engineering-openai-o1)

