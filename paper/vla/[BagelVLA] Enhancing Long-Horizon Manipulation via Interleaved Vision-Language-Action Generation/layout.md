# Page 1
BagelVLA: Enhancing Long-Horizon Manipulation via
Interleaved Vision-Language-Action Generation

Yucheng Hu ∗ 1 , † , Jianke Zhang ∗ 1 , † , Yuanfei Luo ∗ 2 , Yanjiang Guo 1 , Xiaoyu Chen 1 ,
Xinshu Sun 1 , Kun Feng 1 , Qingzhou Lu 1 , Sheng Chen 2 , Yangang Zhang 2 , Wei Li 2 , § ,
Jianyu Chen 1 , §

1 Tsinghua University , 2 ByteDance Seed

∗ Equal contribution , † Joint project lead , § Corresponding Author

Abstract

Equipping embodied agents with the ability to reason about tasks, foresee physical outcomes,
and generate precise actions is essential for general-purpose manipulation. While recent Vision-
Language-Action (VLA) models have leveraged pre-trained foundation models, they typically focus
on either linguistic planning or visual forecasting in isolation. These methods rarely integrate
both capabilities simultaneously to guide action generation, leading to suboptimal performance in
complex, long-horizon manipulation tasks. To bridge this gap, we propose BagelVLA, a unified
model that integrates linguistic planning, visual forecasting, and action generation within a single
framework. Initialized from a pretrained unified understanding and generative model, BagelVLA
is trained to interleave textual reasoning and visual prediction directly into the action execution
loop. To efficiently couple these modalities, we introduce Residual Flow Guidance (RFG), which
initializes from current observation and leverages single-step denoising to extract predictive visual
features, guiding action generation with minimal latency. Extensive experiments demonstrate
that BagelVLA outperforms existing baselines by a significant margin on multiple simulated and
real-world benchmarks, particularly in tasks requiring multi-stage reasoning.

Date: February 12, 2026

Correspondence: liwei.85@bytedance.com, jianyuchen@tsinghua.edu.cn

Project Page: https://cladernyjorn.github.io/BagelVLA.github.io

1
Introduction

The pursuit of generalist robots capable of performing complex manipulation tasks in unstructured envi-
ronments remains a central goal in robotics. A robust embodied agent must possess three fundamental
capabilities: understanding what to do based on instructions, predicting what will happen next, and executing
the necessary motions. While recent Vision-Language-Action (VLA) models [ 4 , 28 , 40 , 44 ] have made progress
by incorporating vision language models (VLMs) [ 2 , 18 , 23 ] or visual generation models [ 12 , 17 , 24 , 37 ], they
often treat these capabilities as separate modules. Some methods focus on high-level planning [ 13 , 18 ] but lack
visual forecasting, while others focus on visual prediction [ 5 , 12 , 17 ] but struggle with the logical reasoning
required for complex tasks [ 27 ]. A unified framework that seamlessly integrates reasoning, prediction, and
control remains a key challenge.

1

arXiv:2602.09849v2  [cs.RO]  11 Feb 2026

# Page 2
Figure 1 Overview of our framework. We present BagelVLA, a unified model that integrates linguistic planning,
visual forecasting, and action generation within a single framework. We construct a massive hybrid dataset combining
general multimodal data with large-scale robotic datasets. Robotic datasets with sub-tasks and keyframes are annotated
to transfer the foundation model’s general reasoning and visual generation abilities to embodied settings.

Meanwhile, the field of multimodal learning has witnessed the emergence of unified understanding and
generation models [ 10 , 38 , 39 , 47 , 48 ]. Architectures like Bagel [ 10 ] employ a single transformer backbone to
jointly process and generate text and images, exhibiting emergent abilities in multimodal reasoning. These
models provide an appealing prior for embodied agents: the model can “think” about the next step in text
and “imagine” the outcome in pixels. However, such general-purpose models are not designed for embodied
domain reasoning and continuous real-time control.

To make unified multimodal priors actionable for long-horizon manipulation, we propose BagelVLA, a unified
VLA framework that integrates linguistic planning, visual forecasting, and action generation. Rather than
treating these as isolated modules, BagelVLA interleaves them within a unified transformer architecture.
The model first generates a textual plan to decompose the instruction (e.g., identifying the next object to
manipulate), then predicts the future visual state, and finally generates the action. This design combines
the logical reasoning of language models with the predictive power of visual generation, providing rich visual
dynamics aligned with instruction to guide low-level control for long-horizon tasks.

Realizing this interleaved behavior requires a suitable training architecture and data, for which we design
a two-stage training strategy to inject embodied multi-modal planning capabilities into the model. In the
first stage, we construct a massive hybrid dataset combining general multimodal data [ 26 , 31 , 45 ] with
large-scale robotic datasets [ 1 , 19 , 21 , 46 ]. Robotic datasets from diverse embodiments are annotated to
transfer the model’s general reasoning and visual predictive abilities to embodied settings. In the second stage,
we introduce the action expert and fine-tune the full model to couple language, predicted visual dynamics,
and control. This progressive approach ensures the model retains its high-level reasoning capabilities while
acquiring precise low-level control policies. To address the high latency in combining visual generation, we
introduce Residual Flow Guidance (RFG). Instead of generating future frames from scratch, RFG conditions
on the current observation as a strong structural prior and performs a single-step denoising to predict the
residual change toward the next keyframe. This mechanism allows the model to extract predictive visual

2

# Page 3
features efficiently, guiding action generation without the computational cost of full image synthesis [ 11 , 24 ],
which substantially reduces the foresight cost.

We validate BagelVLA through extensive experiments in both simulation and real-world environments. Results
show that explicitly coupling linguistic planning with visual forecasting significantly improves performance
over baselines, particularly in long-horizon tasks. In real-world scenarios, BagelVLA demonstrates strong
robustness, successfully generalizing to unseen instructions and diverse object arrangements where baseline
methods often fail. Our contributions are as follows:

• We propose BagelVLA, which integrates linguistic planning, visual forecasting, and action generation
into a single architecture. By explicitly modeling the transition from language to visual dynamics, our
approach enhances reasoning and control in long-horizon tasks.

By exploring various schemes for learning action representations from interleaved planning, we introduce
Residual Flow Guidance (RFG), which uses the current observation as a structural prior and applies
single-step denoising to capture future visual dynamics with minimal latency.

• BagelVLA substantially outperforms existing baselines in simulation benchmarks and demonstrates
strong generalization to diverse instructions and environments in real-world experiments.

2
Related Works

2.0.1
Vision-Language-Action Models

Vision-Language-Action (VLA) models aim to enhance policy generalization to linguistic instructions and
visual scenes by integrating vision-language models (VLMs) with action prediction. For example, methods like
RT-2 [ 3 ] and OpenVLA [ 22 ] employ discrete action tokens compatible with VLMs, allowing direct mapping
from vision-language representations to executable actions, though this can limit expressiveness in continuous
control. In contrast, approaches such as Octo [ 41 ], 3D Diffuser Actor [ 20 ], and π 0 [ 2 ] utilize continuous
action representations via diffusion models to capture multimodal distributions, better handling fine-grained
manipulations. However, these methods—whether discrete or continuous, overlook the alignment gap between
VLM pre-training and VLA fine-tuning, resulting in degraded vision-language capabilities during adaptation.
To mitigate this gap, other approaches [ 15 , 17 , 24 , 37 , 51 , 53 ] introduce visual prediction tasks as a bridge to
map vision-language signals to action signals. For instance, VPP [ 17 ] proposes a video prediction policy that
conditions robot actions on future visual representations derived from video diffusion models. Cosmos Policy
[ 24 ] directly fine-tunes a large pretrained video model to serve as a robot policy. Although pre-training with
pixel prediction can be easily aligned with the robot observations, the absence of a dedicated VLM backbone
often leads to poor instruction-following performance, particularly in tasks requiring complex reasoning.

2.0.2
Unified understanding and generation models

In multimodal learning, recent efforts [ 10 , 38 , 39 , 47 ] have developed unified architectures for joint under-
standing and generation across modalities. For example, Bagel [ 10 ] uses a single transformer to process and
generate text and images, trained on interleaved datasets for emergent reasoning. Chameleon [ 39 ] employs a
token-based framework for mixed-modal input/output, supporting tasks like question answering and image
generation. LMFusion [ 38 ] integrates language and vision in a fused transformer, focusing on efficient cross-
modal alignment, while Show-o [ 47 ] emphasizes unified multimodal understanding and generation, including
text-conditioned image generation and editing for enhanced scene comprehension. These models, trained
on diverse datasets including generation, QA, and editing, demonstrate strong capabilities in multimodal
reasoning that can extend to embodied agents. Inspired by these, several VLA works [ 8 , 34 , 35 , 52 ] have
introduced action experts to transfer their capabilities to embodied scenarios. However, the lack of explicit
embodied vision-language interleaved reasoning means these approaches only retain a subset of the original
model’s capabilities, failing to implement step-by-step multimodal chain-of-thought reasoning. This deficiency
is deemed critical for complex long-horizon tasks. In contrast, our proposed methods successfully incorporate
the multi-modal reasoning capability into robotic manipulation via a complete data processing pipeline and a
progressive training paradigm.

3

# Page 4
3
Methodology

3.1
Preliminaries: Interleaved planning for Robot Control

For classic language-conditioned manipulation settings, a policy is typically learned from a demonstration
dataset D = { L i , τ i } N
i =1 , where each trajectory τ i = { ( v 1 , l 1 , a 1 ) , . . . , ( v T , l T , a T ) } consists of observations v t
(images and proprioception), stage-specific language descriptions l t , and action chunks a t . Conventional VLA
models simplify this by conditioning purely on the global instruction L , learning a direct mapping policy
p θ ( a t | v t , L ) . However, this formulation is insufficient for long-horizon tasks where a global instruction (e.g.,
stacking blocks in a specified order (red → yellow → blue → green)) implicitly entails a sequence of distinct stages.
We address this by modeling the problem as Interleaved Planning . Instead of a black-box mapping, we require
the model to explicitly reason through the causal chain of the task.

Formally, given the global instruction L and current observation v t , BagelVLA models the joint distribution
of the current subtask l t , the future outcome (keyframe) v t + k , and the action a t . This joint distribution
p θ ( a t , v t + k , l t | v t , L ) is factorized based on the logical dependency of manipulation:

1. Linguistic Planning: The model first identifies the immediate textual objective l t from the global
instruction. We consider task decomposition to be the primary semantic capability of VLM-based
architectures.

2. Visual Forecasting: Conditioned on this subtask, the model acts as a world model to predict the physical
outcome v t + k .

3. Action Generation: Finally, the action a t is generated, grounded in both the textual plan and the visual
forecast.

Consequently, our objective is formulated as the maximization of the following factorized likelihood:

J = − ( L l + L v + L a )

= max
θ
E D log p θ ( l t | v t , L ) · p θ ( v t + k | v t , L, l t ) · p θ ( a t | v t , L, l t , v t + k )

3.2
Model Architecture

To address the interleaved planning problem defined in Sec. 3.1 , we propose BagelVLA , a unified framework
for understanding, prediction, and action generation. As illustrated in Fig. 2 , BagelVLA is designed to process
data across three modalities simultaneously. To leverage pre-existing large-scale multimodal data, we employ a
Mixture of Transformers (MoT) architecture to orchestrate experts managing different modalities: specifically,
an LLM expert, a generation expert, and an action expert, all connected via self-attention mechanisms.

We initialize the LLM and generation experts using Bagel [ 10 ], a unified MoT model for understanding and
generation. On top of this foundation, we incorporate a smaller transformer to serve as the action expert.
Distinct from prior MoT-based VLA architectures [ 8 , 35 , 52 ], BagelVLA benefits from robust pre-training
initialization for both its language and vision components and employs a novel dual flow-matching mechanism
(detailed in Sec. 3.3 ). Detailed model settings are described in Appendix A .

3.2.1
Understanding Expert & Generation Expert

The understanding and generation experts adopt the architecture of Qwen2.5-LLM-7B [ 49 ]. Following Bagel’s
configuration, we utilize two distinct visual encoders responsible for visual-language understanding and goal
image prediction, respectively. Each input observation view v t is encoded by the SigLIP2 [ 42 ] and concatenated
with the text instructions L (and l t ) to serve as input for the LLM Expert.

We also utilize the VAE from FLUX [ 25 ] to encode images. For linguistic planning , the understanding expert
attends to ViT features when autoregressively generating the subtask l t . We optimize textual-planning
task using an autoregressive Cross-Entropy (CE) loss: L l = − log p θ ( l t | v t , L ) . For visual forecasting , the
generation expert, while denoising the keyframe image, attends to all input views’ VAE and ViT features,

4

# Page 5
Figure 2 Illustration of the BagelVLA framework. BagelVLA utilizes a Mixture-of-Transformers (MoT) architecture,
comprising three independent transformers specialized for linguistic, visual, and action modalities. To tackle long-
horizon tasks and semantic generalization, we formulate language-conditioned action learning as a long-sequence
interleaved planning problem. As shown, we structure these modalities into a unified sequence, enabling the model to
generate predictions across all three modalities based on the interleaved context. To support this architecture, we have
designed specific mechanisms to facilitate interaction among multiple flow-matching experts and to enhance inference
efficiency.

and relevant textual information. It generates keyframe by iteratively denoising input VAE noise using Flow
Matching [ 29 , 33 ], denoted as: L v = − log p θ ( v t + k | v t , L, l t ) .

3.2.2
Action Expert

We employ an independent transformer connected via the MoT framework as the action expert, which is
responsible for processing proprioceptive and action modalities. The action expert shares a similar architecture
with the Qwen2.5 LLM; however, we reduce the intermediate size of the MLP to 1/5th of the original, resulting
in 2B parameters. This compact size facilitates higher execution frequency during inference through KV-cache
and asynchronous action generation.

For action planning , we employ Flow Matching to learn action chunks, denoted as L a = − log p θ ( a t | v t , L, l t , v t + k ) .
During the denoising process, the action sequence can attend to the VAE and ViT features of the input views,
the global instruction L , the generated subtask l t , and also the proprioceptive state input to the action expert.
Notably, the action expert attends to the intermediate latent states of the image currently being generated.
This involves handling the asymmetric information interaction between the dual Flow Matching modules of
the generation and action experts. We detail the various schemes we explored in Sec. 3.3 and ablate these
methods in Sec. 4.3 .

3.3
Conditioning Schemes in Dual Flow-Matching

In this section, we detail the computation of L v and L a within a unified interleaved input sequence, ensuring
consistency with the inference context. As illustrated in Fig. 3 , we propose three interaction mechanisms for
the Flow Matching (FM) of keyframe prediction and action generation.

5

# Page 6
Figure 3 Illustration of different types of dual denoising schemes. (a) Complete Denoise: Image prediction and
action generation are performed separately, requiring a total of N 1 + N 2 denoising steps. (b) Joint Denoise: Image
prediction and action generation are performed simultaneously, denoising together for N steps. (c) Single-Step Denoise:
Action generation is conditioned directly on the context from the first denoising step of the image prediction. Further
implementation details, including the construction of the concatenated sequence and the attention mask are provided
in Appendix B .

Scheme 1: Complete Denoise
As shown in Fig. 3 (a), Complete Denoise prioritizes the full denoising of
the keyframe image by the generation expert. The generated image is then fed back as context for action
generation. During training, to ensure the action expert observes a fully denoised image, we append the
ground truth keyframe subsequent to the denoising sequence. The loss functions are defined as follows:

L v = E

|| v v,θ ( L, v t , l t , τ, v τ
t + k ) − ( v 1
t + k − v 0
t + k ) || 2
2

,
v τ
t + k = (1 − τ ) v 0
t + k + τv 1
t + k ,
v 1
t + k = v t + k
L a 1 = E

|| v a,θ ( L, v t , l t , v τ =1
t + k , τ, a τ
t ) − ( a 1
t − a 0
t ) || 2
2

,
a τ
t = (1 − τ ) a 0
t + τa 1
t ,
a 1
t = a t
(1)

where v denotes the velocity predicted by the model for the corresponding modality. L and l t represent the
global instruction and sub-task plan, v t is the current input observation, v t + k is the target keyframe and a t is
the action chunk. τ denotes the denoising timestep (where τ = 0 represents initial noise and τ = 1 represents
the ground truth target).

This approach effectively combines a World Model (WM) with an Inverse Dynamics Model (IDM) [ 12 ]. While
theoretically sound for leveraging the WM, it suffers from high inference latency (total denoising steps N 1 + N 2 )
and the potential accumulation of visual errors. To mitigate these issues, we propose alternative schemes.

Scheme 2: Joint Denoise
As shown in Fig. 3 (b), we synchronize the denoising processes of the keyframe
and the action. Here, the action generation attends to the noisy image currently undergoing denoising. The
computation for the action FM loss in Eq. 1 is modified as:

L a 2 = E

|| v a,θ ( L, v t , l t , v τ
t + k , τ, a τ
t ) − ( a 1
t − a 0
t ) || 2
2


During training, we append the action denoising block directly after the keyframe denoising sequence, allowing
the action component to attend to the intermediate noisy keyframes. During inference, the model generates
both keyframes and actions within N steps, significantly reducing latency.

Scheme 3: Single-step Denoise
To further minimize the computational cost of action inference imposed
by image denoising, we propose single-step denoise. In this scheme, action generation attends only to the
KV-cache from the initial denoising step of the keyframe. This implies the model generates actions while
conditioning on the initial noise as the keyframe input:

L a 3 = E

|| v a,θ ( L, v t , l t , v τ =0
t + k , τ, a τ
t ) − ( a 1
t − a 0
t ) || 2
2


6

# Page 7
Furthermore, based on Scheme 3, we introduce a variant of Single-step Denoise where we inject current frame
information into the initial image noise to provide stronger priors for both keyframe and action generation:

Naive Single-step Denoise :
v τ =0
t + k ∼N (0 , I )
(2)

Residual Flow Guidance (RFG) : v τ =0
t + k ∼N ( v t , I )
(3)

More details about implementing the above methods can be found in Appendix B . We provide an ablation
study of these methods in Sec. 4.3 . Based on the results, we select the Single-step Denoise (RFG) as our
default setting for BagelVLA. Notably, we observe that RFG, which incorporates the initial frame v t prior,
significantly reduces the required denoising steps as shown in Fig. 5 . We hypothesize that this allows the
WM to focus on modeling robot manipulation changes rather than reconstructing static background details.
Further quantitative comparisons are available in Sec. 4.3.1 .

3.4
Data Engine

To construct a large-scale pretraining dataset for subtask planning and keyframe prediction in embodied
scenarios, we leverage diverse sources of manipulation demonstrations and apply tailored processing pipelines
to four major data categories in Fig. 1 according to their characteristics. Details of all data annotations and
components are provided in Appendix C .

• Robotic Data The robot data comprises self-collected expert demonstrations and publicly available data
from diverse embodiments. For proprietary data, we manually annotate and segment videos to obtain l t ,
ensuring high-quality planning and keyframe prediction. For public datasets lacking fine-grained labels,
we utilize Seed-1.5-VL-thinking [ 14 ] to synthesize l t and identify temporal boundaries (start and end
frames). These samples are then filtered to retain high-quality instances. These two components are
used exclusively for pretraining to transfer the model’s fundamental planning and prediction capabilities
to the embodied domain.

• General Data General Data includes egocentric human videos and large-scale image–text VQA data.
For the former, we similarly employ Seed-1.5-VL-thinking to generate language annotations; however,
due to the complexity of human-centered scenes, we do not annotate subtasks and instead predict only
the final frame of each operation.

These two data sources are mainly used to preserve the base model’s original understanding and generation
capabilities.

3.5
Training and Inference Strategy

BagelVLA requires the simultaneous alignment of three distinct planning tasks: linguistic planning, visual
forecasting, and action generation. To achieve this, we divide the training process into two stages, maximizing
the utilization of general multimodal data and large-scale embodied data without action labels. Detailed data
recipes and implementation details can be found in Appendix C and D .

Stage 1: Pretraining - Finetuning Linguistic Planning and Learning Visual Dynamics
In this stage, we
exclusively finetune the understanding and generation experts to acquire capabilities in textual planning
and keyframe prediction. To preserve the model’s general linguistic proficiency, we co-train with general
Question-Answering (QA) data. Specifically, the pretraining data comprises:

• General VQA (Language Co-training): 2.98M QA pairs.

• Human-hand Data (Visual Dynamics): 310k episodes.

• Open-source Robot Data (Language Planning & Visual Dynamics): 146k episodes.

• Open-source Robot Data (Visual Dynamics): 297k episodes.

• Self-collected Real Robot Data (Language Planning & Visual Dynamics): 75k episodes.

7

# Page 8
Stage 2: Finetuning - Learning Action Planning
In this stage, we introduce downstream robot data containing
action labels for finetuning. We finetune the entire model on all three planning tasks simultaneously to obtain
an interleaved planning model that performs robustly in specific scenarios. For the four scenarios used in our
experiments, we employ the following finetuning strategies:

• Calvin (Visual & Action Planning): ABC split dataset.

• Robotwin (Linguistic, Visual & Action Planning): 50 tasks with 50 episodes each, totaling 2.5k episodes.

• ALOHA Basic Tasks (Visual & Action Planning): 3k episodes.

• ALOHA Long-horizon Tasks (Linguistic, Visual & Action Planning): 1.5k episodes.

Inference Strategy During inference, the model generates textual plans, keyframes, and actions in an
interleaved manner. At each denoising step, only a single expert is activated (7B model for text and keyframe
or 2B model for action generation). The single-step denoise scheme further enhances execution frequency.
Specifically, we concatenate the current frame, instruction context, and a pure noise image to compute the
KV pairs of the understanding and generation experts, which then condition the action generation. This
mechanism enables BagelVLA to infer at a speed of 1.2 seconds per chunk on a single RTX 5090 GPU (yielding
a real-world action frequency of 40Hz with a chunk size of 48).

We also introduce Asynchronous Execution [ 9 , 50 ] to further boost inference speed. During training, we
randomly replace the current frame with a preceding image. This allows us to reduce the updating frequency
of the KV contexts of understanding and generation experts during inference, updating only the proprioceptive
inputs to output new action chunks. Under this setting, our policy can achieve an execution frequency of
72Hz.

4
Experiment

We conduct extensive experiments to evaluate the interleaved planning capabilities of BagelVLA across a
diverse range of manipulation tasks. These experiments encompass two simulation environments, Calvin [ 36 ]
and Robotwin [ 7 ], as well as a basic tasks suite containing 9 skills of 30 tasks, and a long-horizon task suite
performed on the AgileX dual-arm robot system.

4.1
Evaluation in Simulation Environment

We benchmark BagelVLA against π 0 [ 2 ], RDT [ 32 ] and two VLA models that incorporate future prediction
capabilities, UP-VLA [ 51 ] and VPP [ 17 ], in the Calvin and Robotwin environments.

In the Calvin environment, models are trained on the ABC split and evaluated in the D environment. For
Robotwin, we utilize a training dataset consisting of 50 clean demonstrations for each of the 50 tasks. All
models are then tested in both Clean and Randomized settings using unseen instructions. To verify the
efficacy of interleaved planning, we conduct experiments with BagelVLA trained and tested both with and
without interleaved planning. Further details regarding simulation experiments can be found in Appendix E .

Table 1 Results on Calvin and Robotwin2.0 Benchmarks. Since Calvin consists exclusively of single-step tasks, we
did not evaluate BagelVLA’s performance under the textual planning setting in this domain. Detailed results can be
found in Table 8 and 7 .

Model

Calvin
Robotwin

ABC-D
Clean
Randomized

π 0
3.648
46.42
16.34
RDT
-
34.50
13.72
UP-VLA
4.078
52.92
15.16
VPP
4.329
-
-
w/o Textual-planning
-
54.00
19.20
w/o Keyframe-forecasting
3.345
56.72
15.92
BagelVLA
4.405
75.26
20.87

8
