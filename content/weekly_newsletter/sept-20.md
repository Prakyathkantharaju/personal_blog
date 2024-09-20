---
title: "Weekly Newsletter: Sep 20th"
date: 2020-09-15T11:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["AI - Newsletter"]
author: "Me"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Weekly update on AI/ML - Sept 13th - Sept 20th"
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
    URL: "https://github.com/prakyathkantharaju/personal_blog/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

# AI Newsletter - Latest Developments in Models, Research, and More

## Highlight: Qwen2.5 Family - A Comprehensive Release of AI Models

The Qwen team has unveiled their largest release ever, featuring a wide range of models for various applications.

### Key Features:

- Qwen2.5: Models ranging from 0.5B to 72B parameters
- Qwen2.5-Coder: Specialized models for coding tasks (1.5B, 7B, 32B)
- Qwen2.5-Math: Models optimized for mathematical reasoning (1.5B, 7B, 72B)
- Qwen2-VL-72B: Open-sourced multimodal model
- Over 100 model variants, including quantized versions (GPTQ, AWQ, GGUF)
- Competitive performance against proprietary models
- Apache 2.0 license for most open-source models

The Qwen2.5-72B-Instruct model demonstrates competitive performance against proprietary models and outperforms most open-source models in various benchmark evaluations.

## Models

### 1. Qwen2.5 Family

- 14B and 32B models outperform predecessor Qwen2-72B-Instruct
- Compact 3B model achieves 68 on MMLU, surpassing Qwen1.5-14B
- Qwen2.5-Coder shows competitive performance against larger code LLMs
- Qwen2.5-Math supports both English and Chinese, with improved reasoning capabilities

### 2. Mistral AI's Pixtral 12B

- Natively multimodal model with 400M parameter vision encoder
- Supports multiple images in 128k token context window
- Achieves 52.5% on MMMU reasoning benchmark
- Excels in instruction following, chart understanding, and image-to-code generation

### 3. NVIDIA's NVLM 1.0

- Frontier-class multimodal LLMs rivaling proprietary models
- Novel architecture enhancing training efficiency and reasoning
- 1-D tile-tagging design for high-resolution image processing
- Improved text-only performance after multimodal training

## Research

### 1. GRIN: GRadient-INformed MoE

New approach to Mixture-of-Experts (MoE) training, incorporating sparse gradient estimation for expert routing. Developed a top-2 16×3.8B MoE model that outperforms a 7B dense model and matches a 14B dense model.

### 2. Preference Tuning Survey

Comprehensive overview of recent advancements in preference tuning and human feedback integration across language, speech, and vision tasks.

### 3. Promptriever

First retrieval model able to be prompted like a language model, achieving strong performance on standard retrieval tasks and following instructions. Curated a new 500k instance-level instruction training set from MS MARCO.

## Libraries

New high-performance AI inference stack built for production, utilizing Zig, OpenXLA, MLIR, and Bazel.

## Good Reads

Talk by Hamel Husain and Emil Sedgh on improving LLM apps beyond MVP:

1. Systematic approach to consistently improve AI
2. Avoiding common traps
3. Resources for further learning