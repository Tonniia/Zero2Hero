# Zero-to-Hero: Zero-Shot Initialization Empowering Reference-Based Video Appearance Editing

<a href="https://arxiv.org/abs/2505.23134"><img src='https://img.shields.io/badge/arXiv-2501.11325-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'></a>&nbsp;
<a href="https://zero2hero-project.github.io/"><img src='https://img.shields.io/badge/Project-Page-Green' alt='GitHub'></a>&nbsp;

# Abstract
Appearance editing in video according to user needs is a pivotal task in video editing. Existing text-guided methods often lead to ambiguities regarding user intentions and restrict fine-grained control over editing specific aspects of objects. To overcome these limitations, this paper introduces a novel approach named \emph{Zero2Hero}, which focuses on reference-based video editing that disentangles the editing process into two distinct problems. It achieves this by first editing an anchor frame to satisfy user requirements as a reference image and then consistently propagating its appearance across other frames. 
In our Zero-Stage, we leverage correspondence within the original frames to guide the attention mechanism, which is more robust than previously proposed optical flow or temporal modules in memory-friendly video generative models, especially when dealing with objects exhibiting large motions. It provides a good starting point with accurate and temporally consistent appearance transfer. However, intervention in the attention mechanism can degrade image quality, leading to over-saturated colors and unknown blurring issues. Starting from Zero-Stage, our Hero-Stage can efficiently learn a conditional generative model to restore frames with degradation through a proxy task using the anchor frame and the reference image.
To evaluate consistency more accurately, we construct a group of videos with multiple appearances using Blender, which supports fine-grained evaluation. Our method outperforms the best-performing baseline with a PSNR improvement of 2.6 dB.

![teasor](assets/teasor.jpg)



# Method

Our method contains two stages.
## Pre-preparation
- Download SDv1.5 or SD2.1(defualt) for Zero-Stage appearance transfer. Put under `./_pretrained_model/stable-diffusion-2-1-base`
- Download FLUX.1-dev for Hero-Stage conditional generative model fune-tuning. LoRA rank r=32 under resolution of 512 requires 80GB. 
    - (TODO: find more light-weight DiT-based diffusion model as base model)
- `pip install -r requirements.txt`


## Zero-Stage: zero-shot initialization
- select an anchor frame, and edit it use WebUI, ComfyUI with ControlNet to get reference. We have prepared several reference for *car-turn*.
- prepare data folder at `./_input/_data/car_turn`. `content` are target frames, `style` are reference. 
    - We have prepared several reference for *car-turn* with anchor frame .appearance_name selected in ['rainbow', 'watercolor1', 'red'] . The name of reference: `frame_{anchor_id}_{appearance_name}.png`
- check zero-shot settings at `./_input/_json/zero.json`. ensure `anchor_frame` and `appearance` is corresponding to you selected. 
- run:

```
python ./Zero_Stage/run_zero.py
```
<!-- - We have provided intermediate result, saved in `./_result/zero_stage/1.0_1.0_k=1_100` -->

- Zero-Stage with Adaptive-k strategy have been accepted by IJCAI 2025 AI, the Arts and Creativity track ([paper](assets/tongtong_IJCAI25.pdf)). We further explore few-shot training without hand-craft design, with an addtional Hero-Stage:

## Hero-Stage
- add Hero-Stage training data jsons at `./_input/_json/hero_stage/`
    - We have prepared json of *car-turn*, named as `{appearance_name}_2cond_1pairs.jsonl`. 2cond represent conditional Mode 2 with explicit injection of target frame. 1pairs represent we use ONE data pair b, in Table 1 of paper.
- add corresponding eval jsons
    - named as `{appearance_name}_2cond_eval.jsonl`
- edit training bash script at `./Hero_Stage/train/train_spatial_2cond.sh`:
```
export OUTPUT_DIR=...
export TRAIN_DATA="{training data json}"
...
--eval_json_path "{eval data json}"
```
- "./_result/hero_stage/Mode2/car_turn_rainbow"
- during training, evaluation logs will be saved under folder `OUTPUT_DIR`, under subfolder:
    - `./ckpt`: saved LoRAs
    - `./configs`: backup code and jsons 
    - `./output`: 
        - `__eval_log__.txt`: eval PSNR
        - `lora_{step}_enhance_{i}.jpg`: inference result from `eval_json_path` at that step.
    

<!-- # Experiments -->
<!-- <video width="1280" height="960" controls>
    <source src="assets/baselines.mp4" type="video/mp4"> -->


