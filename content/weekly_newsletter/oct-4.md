---
title: "Weekly Newsletter - Oct 4"
date: 2020-09-15T11:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["weekly-newsletter", "oct-4"]
author: "Prakyath Kantharaju"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Weekly Newsletter - Oct 4"
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: false
ShowBreadCrumbs: false
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: false
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/Prakyathkantharaju/personal_blog/tree/main/content/weekly_newsletter"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---








---
# Highlights
- **OpenAI introduces Canvas**: A new interface for enhanced collaboration in writing and coding projects, rolling out globally.
- **Meta Movie Gen**: A media foundation model for HD video generation with audio, showcasing state-of-the-art advancements.
- **Google Hires OpenAI's Sora Founder**: Google DeepMind recruits lead scientist behind Sora.
- **NVIDIA's NVLM 1.0**: New multimodal models excel at vision-language tasks, rivaling top proprietary models.
- **New Tools**: Torch's ao for quantization and MinerU for comprehensive data extraction.
- **Learning Resources**: Cuda Mode meetup and Torch conference videos covering the latest in LLMs and multimodal systems.

---

# OpenAI News

## Canvas: A New Collaboration Interface 
OpenAI introduces Canvas, an advanced interface for working with ChatGPT on projects that require more than a simple chat. Canvas supports detailed editing, context-based suggestions, and offers tools for writing and coding enhancements. Initially available to ChatGPT Plus, Team, Enterprise, and Edu users, Canvas will eventually be accessible to all users. It helps improve project collaboration by enabling inline editing, feedback, and version control. Canvas launches automatically in applicable scenarios or via the prompt "use canvas."
[Read more here](https://openai.com/index/introducing-canvas)
![Canvas](/oct-4/Canvas_Hero.webp)

Canvas includes various shortcuts:
- **Writing**: Adjust content length, reading level, grammar, add polish, or suggest edits.
- **Coding**: Inline code reviews, debugging support, comments, bug fixes, and language conversion.

Canvas is trained to trigger contextually for different tasks, distinguishing between writing and coding needs. Model evaluations show an 83% accuracy rate for correct canvas usage, outperforming earlier baselines. As Canvas develops, the focus remains on improving collaboration and user interface efficiency.

## Funding Update 
OpenAI secured $6.6 billion in funding at a $157 billion valuation, boosting its frontier AI research, compute capacity, and tool development. This expansion supports OpenAI’s mission to make advanced intelligence widely accessible while working with global partners to shape AI's positive future impact.
[Read more here](https://openai.com/index/scale-the-benefits-of-ai/)

---

# Meta: Movie Gen
Meta has introduced Movie Gen, a set of foundational models capable of generating 1080p HD videos with audio synchronization. The models cover a wide range of tasks, including text-to-video synthesis, video editing, and video personalization. The largest model, a 30-billion parameter transformer, supports a 73,000 video token context for 16-second videos at 16 fps. The paper outlines innovative architectural design, scaling methods, and training protocols that advance media generation capabilities. [Read more here](https://ai.meta.com/static-resource/movie-gen-research-paper).

![Movie Gen](/oct-4/videoframe_2246.png)


---

# Google Hires OpenAI's Sora Founder
Google DeepMind has recruited the lead scientist behind OpenAI's yet-to-be-released Sora video generation model, known for his work on InstructPix2Pix. In a tweet, Demis Hassabis emphasized the goal of building a "world model." [Tweet link](https://x.com/demishassabis/status/1841984103312208037).


![Sora](/oct-4/Pasted%20image%2020241004093735.png)
<!-- ![Sora](/oct-4/Pasted%20image%2020241004094511.png) -->
---

# NVIDIA's New Vision-Language Models: NVLM 1.0
NVIDIA introduces NVLM 1.0, a family of multimodal large language models that perform at the frontier of vision-language tasks, rivaling models like GPT-4o. NVLM 1.0 uses a hybrid architecture that blends the strengths of decoder-only models and cross-attention-based approaches, resulting in state-of-the-art reasoning capabilities across multiple domains, including OCR and multimodal math. By integrating both text and multimodal datasets, NVLM enhances text-only performance while offering production-grade multimodality. Model weights and code will be released for open research. [More details here](https://nvlm-project.github.io/).



### VLM architectures tested in the paper
Please refer to the paper for more details on the performance of each architecture, [paper link here](https://nvlm-project.github.io/).
![NVLM 3](/oct-4/Pasted%20image%2020241004094511.png)

### Benchmarking

![NVLM Benchmark](/oct-4/Pasted%20image%2020241004094615.png)

---

# Fun Tools and Libraries

- **Torch's ao Library**: Facilitates model quantization with built-in support for Torch Compile and FSDP2. [GitHub link](https://github.com/pytorch/ao).
- **MinerU**: Comprehensive, open-source data extraction tool for PDFs, webpages, and e-books. [GitHub link](https://github.com/opendatalab/MinerU).

---

# Videos and Learning Resources
- **Cuda Mode Meetup Keynote**: Andrej Karpathy on LLM.c, vLLM, and more. [Watch here](https://youtu.be/FH5wiwOyPX4?si=yj3u44fns72mpig2).
- **Torch Conference Keynote**: Covers the evolution of LLM architectures. [Watch here](https://youtu.be/frkAt-gZVjc?si=EwGbD3xHvNdlKGtf).
