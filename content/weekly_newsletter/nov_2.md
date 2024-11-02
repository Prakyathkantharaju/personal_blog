---
title: "Nov 2 Weekly Newsletter"
date: 2024-11-02T11:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["weekly_newsletter"]
author: "Prakyath Kantharaju"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Weekly Newsletter for Nov 2"
canonicalURL: "https://prakyath.com/weekly_newsletter/nov_2"
disableHLJS: true # to disable highlightjs
disableShare: false
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
    URL: "https://github.com/Prakyathkantharaju/prakyath.com/content/weekly_newsletter/nov_2.md"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

# AI Newsletter - Nov 2

# highlights:

- Smaller models are better.
- Layer skip decoding is better than beam search.
- Tokenformer is a better architecture for scaling language models.
- Robots can sense the touch using a new sensors developed by meta.
- OpenAI and Google are trading punches with new releases within minutes of each other.

## SmolLM 2: Smaller Models, Bigger Impact
Hugging Face has unveiled SmolLM v2, a groundbreaking release pushing the boundaries of small language models. The collection features three models (135M, 360M, and 1.7B parameters) that outperform larger competitors like Qwen and Llama 3.2 across various benchmarks. Released under the Apache 2 license, these models are specifically designed for edge computing and in-browser applications.

link to the model: https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9 

![smollm2](/nov-3/smollm2.png)

Here is an overview of the small models and how this models fit in.
![small_exosystem](/nov-3/small_model_ecosystem.png)


## Layer Skip: Meta's Breakthrough in LLM Acceleration
Meta has introduced a revolutionary decoding method called Layer Skip that significantly improves LLM performance. Key highlights:
- Achieves up to 2.16x speedup for summarization tasks
- Delivers 1.82x acceleration for coding tasks
- Provides 2.0x improvement in semantic parsing
- Operates by executing select layers and using subsequent ones for verification
- Released with inference code and fine-tuned checkpoints for Llama 3, Llama 2, and Code Llama
link to the paper: https://arxiv.org/abs/2404.16710 


![layer_skip](/nov-3/layerskip.png)

link for more information: https://arxiv.org/abs/2404.16710 


## Knowledge Graphs: A New Approach to Combat Hallucinations
Recent research explores using knowledge graphs for LLM training, offering fascinating insights into hallucination reduction:
- Larger models combined with extended training periods show reduced hallucination rates
- Achieving a ≤5% hallucination rate demands significantly more computing power than previously estimated
- Interesting discovery: as models grow larger, their hallucinations become more difficult to detect
- Provides clearer boundaries and control over knowledge incorporation during training
link for mode information: https://x.com/savvyRL/status/1844073150025515343 


![knowledge_graphs](/nov-3/training_model_with_knowledge_graphs.jpeg)

## Tokenformer: Revolutionizing Language Model Architecture
A new architecture has emerged to tackle the scaling challenges of traditional transformers:
- Introduces token-parameter attention layer to replace linear projections
- Enables progressive scaling from 124M to 1.4B parameters
- Eliminates need for complete retraining when scaling up
- Achieves performance comparable to traditional transformers
- Available as open-source with complete code and model access

link to the paper: https://arxiv.org/abs/2410.23168
![tokenformer](/nov-3/tokenformer.png)

## Meta's Touch-Sensitive Robotics Breakthrough
Meta FAIR has announced significant advances in robotics technology:
- New developments in touch perception and dexterity
- Partnerships with GelSight Inc and Wonik Robotics
- Focus on commercializing tactile sensing innovations
- Commitment to fostering an open ecosystem for AI development

link for more information: https://ai.meta.com/blog/fair-robotics-open-source/?utm_source=twitter&utm_medium=organic_social&utm_content=video&utm_campaign=fair

![meta_touch_robotics](/nov-3/meta-touch.jpg)


## Search Integration Face-off: Google vs OpenAI
### Google's Grounding with Search
Google has launched search grounding for Gemini models, offering:
- Reduced hallucinations through factual grounding
- Real-time information access
- Enhanced trustworthiness with supporting links
- Richer information through Google Search integration

### OpenAI's ChatGPT Web Search
OpenAI has enhanced ChatGPT with improved web search capabilities:
- Faster, more timely answers
- Direct links to relevant web sources
- Improved search accuracy and relevance



<!-- 
## Anthropic's Insight: The Capability-Bias Trade-off
Anthropic's latest research reveals an interesting relationship between language model capabilities and bias:
- Discovery of a feature that significantly reduces bias across nine social dimensions
- Identification of a "sweet spot" in the trade-off between bias reduction and model capabilities
- Slight capability decrease noted when implementing bias reduction features

## SmolLM 2 from huggingface

Super proud to release SmolLM v2 pushing the state-of-the-art performances of LLMs under 2B parameters with three sizes: 135M, 360M and 1.7B parameters.

These models have been several months in the making with many lessons learned that we’ll share soon… all under Apache 2 licence.

The future of on-the-edge/in-browser models is exciting!


These models beat out the other models such as Qwen and llama 3.2 in the varaious benchmarks. 

Here is an overview of how these model fit in the landscape of the small model ecosystem.


## A new decoding method from meta  - Layer skip


Layer Skip accelerates LLMs by executing a subset of its layers and utilizing subsequent layers for verification and correction.

We’re releasing the inference code and fine-tuned checkpoints for Layer Skip, including Llama 3, Llama 2, and Code Llama. These models have been optimized with the Layer Skip training recipe, significantly improving the accuracy of early layer exits. Additionally, we're sharing Layer Skip's inference implementation, which can boost model performance by up to 1.7x.

What sets these Layer Skip checkpoints apart is their robustness to exiting at earlier layers and to skipping intermediate layers, as well as the uniformity of activations across layers. These unique features pave the way for innovative research in optimization and interpretability. We’re excited to see how the research community leverages these tools to push the boundaries of what's possible with AI.

We present LayerSkip, an end-to-end solution to speed-up inference of large language models (LLMs). First, during training we apply layer dropout, with low dropout rates for earlier layers and higher dropout rates for later layers, and an early exit loss where all transformer layers share the same exit. Second, during inference, we show that this training recipe increases the accuracy of early exit at earlier layers, without adding any auxiliary layers or modules to the model. Third, we present a novel self-speculative decoding solution where we exit at early layers and verify and correct with remaining layers of the model. Our proposed self-speculative decoding approach has less memory footprint than other speculative decoding approaches and benefits from shared compute and activations of the draft and verification stages. We run experiments on different Llama model sizes on different types of training: pretraining from scratch, continual pretraining, finetuning on specific data domain, and finetuning on specific task. We implement our inference solution and show speedups of up to 2.16x on summarization for CNN/DM documents, 1.82x on coding, and 2.0x on TOPv2 semantic parsing task. We open source our code and checkpoints at th
paper link: https://arxiv.org/abs/2404.16710

## New conference paper on training models on the knowledge graphs

First, why? As old-fashioned ML researchers we were just very frustrated how **it's never clear what knowledge content is consumed in LM training**. The training data of regular LMs are huge, messy, inaccessible, ambiguous, & do not have a clear boundary of knowledge separation.

This new method of training models helps eliviate this problem incorporating the knowledge graphs, there are some interesting results and observation noted in the paper. About the need of multi epoch training and how the testing and training data was split and can be used to improve the model.

While many capabilities of language models (LMs) improve with increased training budget, the influence of scale on hallucinations is not yet fully understood. Hallucinations come in many forms, and there is no universally accepted definition. We thus focus on studying only those hallucinations where a correct answer appears verbatim in the training set. To fully control the training data content, we construct a knowledge graph (KG)-based dataset, and use it to train a set of increasingly large LMs. We find that for a fixed dataset, larger and longer-trained LMs hallucinate less. However, hallucinating on ≤5% of the training data requires an order of magnitude larger model, and thus an order of magnitude more compute, than Hoffmann et al. (2022) reported was optimal. Given this costliness, we study how hallucination detectors depend on scale. While we see detector size improves performance on fixed LM's outputs, we find an inverse relationship between the scale of the LM and the detectability of its hallucinations.

paper link: https://arxiv.org/abs/2408.07852

## A new and improved architecture for scaling language models - tokenformer.

Transformers have become the predominant architecture in foundation models due
to their excellent performance across various domains. However, the substantial
cost of scaling these models remains a significant concern. This problem arises
primarily from their dependence on a fixed number of parameters within linear
projections. When architectural modifications (e.g., channel dimensions) are introduced, the entire model typically requires retraining from scratch. As model sizes
continue growing, this strategy results in increasingly high computational costs
and becomes unsustainable. To overcome this problem, we introduce Tokenformer,
a natively scalable architecture that leverages the attention mechanism not only
for computations among input tokens but also for interactions between tokens
and model parameters, thereby enhancing architectural flexibility. By treating
model parameters as tokens, we replace all the linear projections in Transformers
with our token-parameter attention layer, where input tokens act as queries and
model parameters as keys and values. This reformulation allows for progressive
and efficient scaling without necessitating retraining from scratch. Our model
scales from 124M to 1.4B parameters by incrementally adding new key-value
parameter pairs, achieving performance comparable to Transformers trained from
scratch while greatly reducing training costs. Code and models are available at
https://github.com/Haiyang-W/TokenFormer.


## Robots can sense the touch using a new sensors developed by meta.


Meta FAIR is publicly releasing several new research artifacts that advance robotics and support our goal of reaching advanced machine intelligence (AMI).
The work we’re sharing today includes advancements in touch perception, dexterity, and human-robot interaction, all critical ingredients on the path towards achieving AMI.
We’re also announcing strategic partnerships with GelSight Inc and Wonik Robotics to develop and commercialize tactile sensing innovations that enable easy access for the research community and help foster an open ecosystem for AI.


## A better language model with search grounding - Google.

Today, we are rolling out Grounding with Google Search in Google AI Studio and the Gemini API, enabling developers to get more accurate and fresh responses from the Gemini models aided by Google Search. In addition to more accurate responses, the model returns grounding sources (in-line supporting links) and Search Suggestions that point users to the search results corresponding to the grounded response.

Developers should enable Grounding with Google Search for queries and applications which could benefit from any of the following:

Reduced hallucinations: Grounding helps ensure that AI applications provide users with more factual information.
More up-to-date information: With grounding, models can access real-time information, making AI applications relevant and applicable to a wider range of scenarios.
Enhanced trustworthiness and traffic to publishers: By providing supporting links, grounding brings transparency to AI applications, making them more trustworthy and encouraging users to click on the underlying sources to find out more.
Richer information: By drawing information from Google Search to enhance the model response, grounding is able to provide richer color on many queries.

## A better search with llm from - openai.

ChatGPT can now search the web in a much better way than before so you get fast, timely answers with links to relevant web sources.


## New study from antropic shows that langauge model bias an capbility might be related to each other and there is a sweet spot for this tradeoff.

Finally, we discovered a feature that significantly reduces bias scores across nine social dimensions within the sweet spot. This did come with a slight capability drop, which highlights potential trade-offs in feature steering.
 -->
