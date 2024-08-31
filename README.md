# The Fabrication of Reality and Fantasy: Scene Generation with LLM-Assisted Prompt Interpretation
This is the repository that contains source code for the [Reality-and-Fantasy website](https://leo81005.github.io/Reality-and-Fantasy/).

![Intro Image](https://leo81005.github.io/Reality-and-Fantasy/./static/images/intro.png)

# Acknowledgement
This work builds upon the codebase of [LLM-grounded Diffusion (LMD)](https://github.com/TonyLianLong/LLM-groundedDiffusion). Strongly recommend referring to the original LMD work for further insights and context.

# Realistic-Fantasy Network (RFNet)
## Installation
```
pip install -r requirements.txt
```

## Stage 1: LLM-Driven Detail Synthesis
```
python prompt_batch.py --prompt-type demo --model gpt-4 --always-save --template_version v0.1
```

## Stage 2: Comprehensive Image Synthesis
```
python generate.py --prompt-type demo --model gpt-4 --save-suffix "gpt-4" --repeats 4 --frozen_step_ratio 0.5 --regenerate 1 --force_run_ind 0 --run-model lmd --use-sdv2 --no-scale-boxes-default --template_version v0.1 --sdxl
```

## Citation
```
@article{yao2024fabricationrealityfantasyscene,
    title = {The Fabrication of Reality and Fantasy: Scene Generation with LLM-Assisted Prompt Interpretation}, 
    author = {Yi Yao and Chan-Feng Hsu and Jhe-Hao Lin and Hongxia Xie and Terence Lin and Yi-Ning Huang and Hong-Han Shuai and Wen-Huang Cheng},
    year = {2024},
    eprint = {2407.12579},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url = {https://arxiv.org/abs/2407.12579}
}
```