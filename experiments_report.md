---
title: Benchmarking GPU Performance in Vulnerability Severity Model Training
description: Benchmarking GPU Performance in Vulnerability Severity Model Training
author: CIRCL Team
date: 2025-11-27
tags: ["ai", "benchmark", "vulnerability-classification", "VulnTrain"]
toc: truec
numbersections: true
geometry: margin=1in
---

\newpage

# Preface

This document summarizes the benchmarking, training configuration, and performance results obtained while generating the **Vulnerability Severity Classification** model across different GPU architectures.

The  **VLAI Vulnerability Severity Classification** model developed at CIRCL is regularly updated and shared on Hugging Face. It has been presented in:

> Bonhomme, C., & Dulaunoy, A. (2025). *VLAI: A RoBERTa-Based Model for Automated Vulnerability Severity Classification* (Version 1.4.0) [Computer software].  
> https://doi.org/10.48550/arXiv.2507.03607[^1]


[^1]: [https://arxiv.org/abs/2507.03607](https://arxiv.org/abs/2507.03607)

All materials used to produce this technical report—including Matplotlib scripts, datasets, and other resources—are available in the Git repository:
[https://github.com/vulnerability-lookup/gpu-vuln-bench](https://github.com/vulnerability-lookup/gpu-vuln-bench)

---

# Environments used for benchmarking

## GPU Architectures

The benchmarks in the following sections were performed on the GPU architectures listed below.


| Environment | CPU Cores                            | GPU(s)                 | RAM       |
|-------------|--------------------------------------|------------------------|-----------|
| A           | 64  (AMD EPYC 9124 16-Core Processor)| 2 × NVIDIA L40S        | 251.5 GB  |
| B           | 224 (Intel Xeon Platinum 8480+)      | 2 × NVIDIA H100 NVL    | 2,014 GB  |
| C           | 224 (Intel Xeon Platinum 8480+)      | 4 × NVIDIA L40S        | 2,014 GB  |


## Framework Versions

The environment used for training:

- **Python:** 3.12.3
- **Transformers:** 4.57.1
- **PyTorch:** 2.9.1+cu128
- **Datasets:** 4.4.1
- **Tokenizers:** 0.22.1


# Dataset

The dataset used for training and evaluation is available on Hugging Face:

[https://huggingface.co/datasets/CIRCL/vulnerability-scores](https://huggingface.co/datasets/CIRCL/vulnerability-scores)

at the commit ``cbb05f48e20e2186a80284de138cafee56b6544c``[^2].

This is the updated version of the dataset referenced in ``arXiv.2507.03607``.

Dataset statistics:

- Number of rows: 657,024
- Downloaded size: 162 MB
- Auto-converted Parquet size: 162 MB

This dataset is periodically updated with data collected with [Vulnerability-Lookup](https://vulnerability.circl.lu).

[^2]: [https://huggingface.co/datasets/CIRCL/vulnerability-scores/tree/cbb05f48e20e2186a80284de138cafee56b6544c](https://huggingface.co/datasets/CIRCL/vulnerability-scores/tree/cbb05f48e20e2186a80284de138cafee56b6544c)


# Model training

## Resulting models

The main model is available on Hugging Face[^3].  
It is a fine-tuned version of [RoBERTa-base](https://huggingface.co/roberta-base) trained on the [CIRCL/vulnerability-scores](https://huggingface.co/datasets/CIRCL/vulnerability-scores) dataset.  
Intermediate models are also available on Hugging Face and are versioned for reproducibility.


[^3]: [https://huggingface.co/CIRCL/vulnerability-severity-classification-roberta-base](https://huggingface.co/CIRCL/vulnerability-severity-classification-roberta-base)

The code fo the trainer is available in the VulnTrain project[^4].

[^4]: [https://github.com/vulnerability-lookup/VulnTrain](https://github.com/vulnerability-lookup/VulnTrain)


## Training Hyperparameters

The following hyperparameters were used during training:

- **Learning rate:** `3e-05`
- **Per device Batch Size:** 8
- **Seed:** `42`
- **Optimizer:** `ADAMW_TORCH_FUSED`
- **Scheduler:** `linear`
- **Epochs:** `5`

For a RoBERTa model, the default batch size per device we chose is **8**.

RoBERTa-base is a medium-sized Transformer model (approx. 125 million parameters). A batch size of 8 per device is a standard, conservative choice that is un likely to cause Out-of-Memory (OOM) errors on most modern GPUs (like NVIDIA V100, A100, or even modern consumer cards like the RTX 3080/4080) for typical sequence lengths (e.g., 128 or 256 tokens).

$3 \times 10^{-5}$ is a standard and safe learning rate for fine-tuning RoBERTa, with the optimizer using its default settings.


### A quick note on epochs and batches

A **batch** is a subset of the training data processed together in **one forward and backward pass**, producing gradients that update the model weights.  
The batch size is the number of samples in that batch.  

An **epoch** is one full pass over the entire training dataset.  
Since the dataset is divided into batches, an epoch consists of multiple steps, where each step processes one batch and updates the model weights.  

The **effective batch size** (batch size × number of GPUs) influences training dynamics:  

- Larger effective batches produce more stable gradients, require fewer optimization steps per epoch, and often converge faster.
- Smaller batches introduce noise in the gradients, which can help escape poor local minima and improve generalization, but each epoch takes longer.
- The impact on generalization also depends on using an appropriate learning rate.

RoBERTa often benefits from slightly larger batches. For example, using a batch of 32 samples per step can reduce gradient noise and stabilize learning, leading to quicker convergence.


![# GPUs / Batch Size - Illustration 1](img/gpu-batch-size-example-1.png)

![# GPUs / Batch Size - Illustration 2](img/gpu-batch-size-example-2.png)

In our case, the training set contains 657,024 samples. The visualizations use a simplified view to illustrate the concepts more clearly for learning purposes.

The diagrams above illustrate how batch size, number of GPUs, and dataset size affect training steps per epoch:

1. **Batch size per GPU (`per_device_batch`)**: Number of samples each GPU processes in a single step (forward + backward pass).
2. **Effective batch size (`per_device_batch × #GPUs`)**: Total samples processed per step across all GPUs.
3. **Step**: One forward + backward pass of the effective batch.
4. **Epoch**: One full pass over the dataset.



## Training results

| Environment| Final Loss | Final Accuracy | Epochs to Converge | Batch Size | Steps per Epoch    |
| ---------- | ---------- | -------------- | ------------------ | ---------- | ------------------ |
| A          | 0.2537     | 0.8232         | 5                  | 16         | **29470**          |
| B          | 0.2801     | 0.8230         | 5                  | 16         | **29470**          |
| C          | 0.3793     | 0.8173         | 5                  | 32         | **14735**          |

Results in terms of **loss** and **accuracy** are very similar, regardless of the system used.  
Each experiment produced slightly different rankings, but the differences are minimal.


### Environment A

| Training Loss | Epoch |  Step  | Validation Loss | Accuracy | steps_per_epoch | samples_per_epoch |
|--------------:|:-----:|-------:|----------------:|---------:|----------------:|-----------------:|
| 0.4999        | 1.0   | **29470**  | 0.6657          | 0.7290   | 29470.0         | 471520.0         |
| 0.5279        | 2.0   | 58940  | 0.5911          | 0.7685   | 29470.0         | 471520.0         |
| 0.4775        | 3.0   | 88410  | 0.5392          | 0.7961   | 29470.0         | 471520.0         |
| 0.3753        | 4.0   | 117880 | 0.5125          | 0.8122   | 29470.0         | 471520.0         |
| 0.2537        | 5.0   | 147350 | 0.5169          | 0.8232   | 29470.0         | 471520.0         |

https://huggingface.co/CIRCL/vulnerability-severity-classification-roberta-base-expA

### Environment B

| Training Loss | Epoch |  Step  | Validation Loss | Accuracy | steps_per_epoch | samples_per_epoch |
|--------------:|:-----:|-------:|----------------:|---------:|----------------:|-----------------:|
| 0.5379        | 1.0   | **29470**  | 0.6573          | 0.7358   | 29470.0         | 471520.0         |
| 0.5714        | 2.0   | 58940  | 0.5810          | 0.7710   | 29470.0         | 471520.0         |
| 0.4636        | 3.0   | 88410  | 0.5412          | 0.7918   | 29470.0         | 471520.0         |
| 0.4738        | 4.0   | 117880 | 0.5098          | 0.8131   | 29470.0         | 471520.0         |
| 0.2801        | 5.0   | 147350 | 0.5175          | 0.8230   | 29470.0         | 471520.0         |

https://huggingface.co/CIRCL/vulnerability-severity-classification-roberta-base-expB

### Environment C

| Training Loss | Epoch |  Step  | Validation Loss | Accuracy | steps_per_epoch | samples_per_epoch |
|--------------:|:-----:|-------:|----------------:|---------:|----------------:|-----------------:|
| 0.6270        | 1.0   | **14735**  | 0.6594          | 0.7298   | 14735.0         | 471520.0         |
| 0.5675        | 2.0   | 29470  | 0.5780          | 0.7693   | 14735.0         | 471520.0         |
| 0.4690        | 3.0   | 44205  | 0.5363          | 0.7930   | 14735.0         | 471520.0         |
| 0.4373        | 4.0   | 58940  | 0.5069          | 0.8107   | 14735.0         | 471520.0         |
| 0.3793        | 5.0   | 73675  | 0.5071          | 0.8173   | 14735.0         | 471520.0         |

https://huggingface.co/CIRCL/vulnerability-severity-classification-roberta-base-expC


Note that $147350 / 2 = 73675$.


### Comparisons

![Cumulative Samples vs Steps](img/cumulative-samples-steps.png)

A common rule of thumb is the **linear scaling rule**: when the effective batch size is doubled, the learning rate is also doubled.  
This behavior is confirmed in all of our experiments.


![Validation Accuracy per Epoch](img/accuracy-per-epoch.png)

The chart shows the validation accuracy per epoch for the various experiments with the environments A, B, and C.  
All experiments exhibit very similar accuracy trends.  
Experiment C reaches higher accuracy more quickly in the early epochs, reflecting faster convergence per epoch due to a larger effective batch size (more GPUs × batch per device).  
By the final epoch, all experiments achieve comparable accuracy (~0.82), indicating consistent model performance across the different setups.


## Key Observations

- **More GPUs → larger effective batch → fewer steps per epoch**
  - Example:
    - 4 GPUs × 256 samples → 1024 samples/step → fewer steps to process the full dataset
    - 2 GPUs × 256 samples → 612 samples/step → more steps to process the same dataset
- **Larger batch size per device → fewer steps per epoch**, but each step processes more data.
- **Epoch duration** is proportional to number of steps × time per step, so increasing GPUs or batch size reduces total training time per epoch.

This visual makes it easier to understand why Exp C (4 GPUs, batch size 8 per device → effective batch 32) completes fewer steps per epoch and thus runs faster per epoch than Exp A/B (2 GPUs, effective batch 16), even though the dataset and model are identical.


# Benchmark Comparisons


## Duration

![Duration](img/duration_comparison.png)

## Energy

![Energy breakdown comparison](img/energy_breakdown_comparison.png)

![Energy consumption comparison](img/energy_consumption_comparison.png)

![CPU/GPU/RAM Energy breakdown](img/radar_energy_comparison.png)

## Emissions

![Emissions comparison](img/emissions_comparison.png)

## GPU Power

![GPU power](img/gpu_power_comparison.png)

## Energy vs. Duration

![Energy vs duration](img/scatter_energy_vs_duration.png)

## GPU Power vs. Duration

![GPU power vs duration](img/scatter_gpu_power_vs_duration.png)

## GPU Power vs. Energy

![GPU power vs energy](img/scatter_gpu_power_vs_energy.png)




# Resources

## Related to CodeCarbon's RAM Energy Calculation

CodeCarbon primarily calculates the energy used by **RAM** through a **power consumption model** based on estimations, rather than direct hardware measurement, unless specific system features are available.

The power estimation for a "large server" is approximately 40W (using 8x128GB DIMMs with high efficiency scaling).

Reference: [https://mlco2.github.io/codecarbon/methodology.html#ram](https://mlco2.github.io/codecarbon/methodology.html#ram)


### Estimation Methodology

The default method relies on a fixed power consumption value per installed RAM module (DIMM):

1.  **Fixed Power per DIMM:** A standardized, average power consumption value is assigned to each RAM module.
    * For **x86 Systems** (most standard laptops/desktops), this is typically set at **5 Watts** per DIMM.
    * For **ARM Systems** (e.g., Raspberry Pi), a lower base power, like **1.5W** per DIMM, or a constant of **3W**, is used.
2.  **Counting RAM Modules:** CodeCarbon attempts to determine the **number of installed RAM modules (DIMMs)** on the system by querying the operating system.
3.  **Total Power Calculation:** The estimated total RAM power is calculated by multiplying these two values:
    $$\text{RAM Power (Watts)} = \text{Fixed Power per DIMM} \times \text{Number of RAM Slots Used}$$
4.  **Scaling (for Servers):** For systems with many DIMMs (e.g., servers with 8+ slots), a scaling factor is applied to reduce the power assigned to each additional DIMM, acknowledging that power consumption doesn't increase strictly linearly in large configurations.


### Energy Calculation

Once the estimated **RAM Power** (in Watts) is determined, the **Energy Consumed** (in kilowatt-hours, or kWh) is calculated based on the duration of the code execution:

$$\text{Energy (kWh)} = \frac{\text{Power (Watts)} \times \text{Time (hours)}}{1000}$$


### Direct Measurement Alternative

On Linux systems, CodeCarbon offers a more accurate method with the **Intel Running Average Power Limit (RAPL)** interface.

* If the `rapl_include_dram` parameter is set to `True`, CodeCarbon will attempt to use the **direct power measurement** for the DRAM (memory subsystem) provided by RAPL, overriding the fixed power estimation model. This method offers the most precise consumption data when available.

Reference: [https://mlco2.github.io/codecarbon/parameters.html](https://mlco2.github.io/codecarbon/parameters.html)


## Related to CodeCarbon's GPU Energy Calculation

The energy consumption is tracked using ``nvidia-ml-py``library.

Reference: [https://mlco2.github.io/codecarbon/methodology.html#gpu](https://mlco2.github.io/codecarbon/methodology.html#gpu)


## Environmental Considerations

Our server room is hosted in LuxConnect’s data centers, which are powered entirely by renewable energy ([https://www.luxconnect.lu/infrastructure](https://www.luxconnect.lu/infrastructure)).


# Feedback

Feel free to share your feedback at [info@circl.lu](mailto:info@circl.lu).



# Funding

![EU Funding](europe.png)

[AIPITCH](https://www.science.nask.pl/en/research-areas/projects/12456) aims to create advanced artificial intelligence-based tools supporting key operational services in cyber defense.
These include technologies for early threat detection, automatic malware classification, and improvement of analytical processes through the integration of Large Language Models (LLM).
The project has the potential to set new standards in the cybersecurity industry.

The project leader is NASK National Research Institute. The international consortium includes:

- CIRCL (Computer Incident Response Center Luxembourg), Luxembourg
- The Shadowserver Foundation, Netherlands
- NCBJ (National Centre for Nuclear Research), Poland
- ABI LAB (Centre of Research and Innovation for Banks), Italy

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Cybersecurity Competence Centre.
Neither the European Union nor the European Cybersecurity Competence Centre can be held responsible for them.

