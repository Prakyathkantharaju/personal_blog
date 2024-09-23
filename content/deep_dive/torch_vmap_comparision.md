---
title: "Deep Dive: PyTorch vmap vs JAX vmap - Part 1"
date: 2020-09-15T11:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["pytorch", "vmap", "JAX"]
author: "Me"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Comparing the memory and speed of PyTorch vmap and JAX vmap for dot product and attention calculation "
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
    URL: "https://github.com/prakyathkantharaju/personal-blog/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

# Deep Dive part 1: PyTorch vmap vs JAX vmap

## Introduction:
Hello everyone, I am starting a new series of blog posts called `deep dives`, where I will do a deep and detailed analysis of machine learning frameworks and algorithms. Here I will break down new APIs and features of frameworks such as PyTorch, JAX, and tinygrad. In addition to that, I will also write about new algorithms in RLHF and other areas of research.

The first topic I will be writing about is vmap - vectorized mapping of callable functions across arrays/tensors. It returns a new function which can be called across arrays/tensors. It is a very powerful feature and most used in the XLA-based frameworks such as JAX. However, I recently discovered (slides link) that PyTorch has a similar feature called vmap through its `func` API layer.

In this blog post, I will do both memory and speed comparisons between the two framework implementations for a simple dot product operation and the attention calculation.

The reason why I chose to compare memory and speed for the dot product and attention operation is because I want to compare a simple operation which will require two kernel operations and a complex operation where we need multiple kernel calls such as attention. Another key reason I am writing this blog is because of the memory comparison between the two frameworks. I understand that capturing memory allocation in Python is not perfect and specifically for multi-threaded applications like ML frameworks is even harder; famous memory applications such as `memory_profiler` and `line_profiler` cannot be used due to this reason. However, I think it is still useful to get a general idea of the memory allocation of the two frameworks. To curcumvent this issue, I have captured the process ID and calculated the memory usage using `psutils`. If you believe that you have a better approach, please feel free to DM me or open a PR to the code repo given below.

Here is the repo comparing the two frameworks: [github link](https://github.com/Prakyathkantharaju/benchmark_torch_vmap_jax)

## Experiment information
Each benchmarking will be running for 1000 iterations. During each function call, I am recording the time and memory usage. As mentioned earlier, I am using `psutils` to capture the memory usage and will be using `time.perf_counter` to capture the time usage. I am also counting the GPU memory using the `pynvml` library, but more on the GPU performance comparison later.

Note: I am using warm start for the jitted JAX function comparison since the first function call for JAX is always slower due to compilation. If you feel this is unfair, please feel free to remove the warmup flag in the comparison function and run the code again to compare the results.

## CPU

For the dot product operation, I am comparing the PyTorch dot product with a for loop as suggested in the slides and the vmap version. For the JAX version, I am comparing the vmap version and the jit version of the dot product. For the attention operation, I have simplified and compared the vmap version and the jit + vmap version of the attention operation. In the future, I will also compare other implementations of JAX and PyTorch and complex operations such as model loading and switching between CPU and GPU etc.

The variables are declared before the dot product or attention calculation since in this blog I am concerned about the vmap comparison and not the variable declaration and initialization, etc.

Here is the time comparison for the dot product operation on the left and the attention operation on the right.
![dot product time comparison](../images/framework_comparison.png)

Here is the memory comparison for the dot product operation on the left and the attention operation on the right.
![dot product memory comparison](../images/framework_memory_comparison.png)

## GPU
WIP......

## Discussion and Conclusion
WIP.....