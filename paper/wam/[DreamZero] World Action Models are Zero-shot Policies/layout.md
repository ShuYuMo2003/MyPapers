# Page 1
2026-2-19

World Action Models are Zero-shot Policies

Seonghyeon Ye †
Yunhao Ge *
Kaiyuan Zheng *
Shenyuan Gao *
Sihyun Yu *
George Kurian *

Suneel Indupuru *
You Liang Tan *
Chuning Zhu
Jiannan Xiang
Ayaan Malik
Kyungmin Lee

William Liang
Nadun Ranawaka
Jiasheng Gu
Yinzhen Xu
Guanzhi Wang
Fengyuan Hu

Avnish Narayan
Johan Bjorck
Jing Wang
Gwanghyun Kim
Dantong Niu
Ruijie Zheng
Yuqi Xie

Jimmy Wu
Qi Wang
Ryan Julian
Danfei Xu
Yilun Du
Yevgen Chebotar
Scott Reed
Jan Kautz

Yuke Zhu †
Linxi “Jim” Fan †
Joel Jang †

NVIDIA
† Project Leads
* Core Contributors
https://dreamzero0.github.io

Figure 1: Overview . By jointly predicting video and action, World Action Models (WAMs) inherit world physics
priors that enable 1) effective learning from diverse, non-repetitive data, 2) open-world generalization, 3)
cross-embodiment learning from video-only data, and 4) few-shot adaptation to new robots.

© 2026 NVIDIA. All rights reserved.

arXiv:2602.15922v1  [cs.RO]  17 Feb 2026

# Page 2
World Action Models are Zero-shot Policies

Abstract

State-of-the-art Vision-Language-Action (VLA) models excel at semantic generalization but struggle
to generalize to unseen physical motions in novel environments. We introduce DreamZero , a World
Action Model (WAM) built upon a pretrained video diffusion backbone. Unlike VLAs, WAMs learn physical
dynamics by predicting future world states and actions, using video as a dense representation of how
the world evolves. By jointly modeling video and action, DreamZero learns diverse skills effectively
from heterogeneous robot data without relying on repetitive demonstrations. This results in over 2 ×
improvement in generalization to new tasks and environments compared to state-of-the-art VLAs in real-
robot experiments. Crucially, through model and system optimizations, we enable a 14B autoregressive
video diffusion model to perform real-time closed-loop control at 7Hz. Finally, we demonstrate two
forms of cross-embodiment transfer: video-only demonstrations from other robots or humans yield a
relative improvement of over 42% on unseen task performance with just 10–20 minutes of data. More
surprisingly, DreamZero enables few-shot embodiment adaptation, transferring to a new embodiment
with only 30 minutes of play data while retaining zero-shot generalization.

1. Introduction

Recent robotic foundation models, termed Vision-Language Action models (VLAs), extend pretrained Vision-
Language Models (VLMs) to predict motor actions ( Bjorck et al. , 2025 ; Black et al. , 2024 ; Brohan et al. , 2023 ;
Gemini Robotics Team , 2025 ; Kim et al. , 2024 ). While VLAs successfully inherit linguistic priors to generalize
across diverse language instructions, especially manipulating diverse objects ( Brohan et al. , 2023 ), their
generalization to novel environments and, more critically, to new motions or skills remains limited ( Guruprasad
et al. , 2025 ; Zhou et al. , 2025 ). For example, VLAs can successfully execute “move coke can to Taylor Swift”
( Brohan et al. , 2023 ) by leveraging the web knowledge acquired during VLM pretraining to identify the target
location, and connecting it to the learned move skill from the robot data. However, they fail at a task like
“untie the shoelace” if that specific skill was not present in the robot training data. Although VLM priors encode
what to do at a semantic level, they lack representations of how actions should be executed with precise spatial
awareness, aligned with geometry, dynamics, and motor control ( Chen et al. , 2024 ; Feng et al. , 2025 ). As a
result, VLAs often struggle to adapt to new environments or generalize to novel tasks beyond the distribution
of expert demonstrations, without explicitly collecting large-scale task- and environment-specific action data.

In this paper, we present DreamZero , a 14B robot foundation model built upon a pretrained image-to-
video diffusion backbone ( Team Wan , 2025 ). We term this architecture a World Action Model (WAM) —a
foundation model designed to predict both actions and visual future states in an aligned manner. Initialized
from video diffusion models trained on web-scale video data, WAMs leverage rich spatiotemporal priors to jointly
generate future frames and actions conditioned on language instructions and observations. This shifts action
learning from dense state–action imitation to inverse dynamics—aligning motor commands with predicted
visual futures. Consequently, we observe that this enables (1) effective learning from robot data that are
heterogeneous trajectories collected during the execution of useful behaviors in real-world settings, rather
than relying solely on carefully repeated demonstrations (2) zero-shot generalization to new tasks in new
environments, and (3) efficient cross-embodiment transfer.

This approach yields three core advancements that distinguish DreamZero from prior work, including other
WAMs ( Kim et al. , 2026 ; Liang et al. , 2025 ; Pai et al. , 2025 ). First, DreamZero unlocks new generalization
capabilities beyond traditional VLAs and previous WAMs—across environments, across tasks, and across
embodiments (Figure 2 and Figure 3 ). Compared to the state-of-the-art pretrained VLAs, we observe more
than a 2 × improvement in average task progress on environment and task generalization benchmarks. Second,
DreamZero demonstrates that generalist policies can be learned effectively from diverse, heterogeneous
data, breaking away from the conventional wisdom that generalist robot policies require multiple repeated

2

# Page 3
World Action Models are Zero-shot Policies

Figure 2: Joint Video and Action Prediction . DreamZero jointly generates video and action. We observe
that the predicted actions closely align with the generated video. The examples are from totally unseen tasks.

demonstrations per task. Although other WAMs show that priors learned from videos prediction improves
sample efficiency for action learning compared to VLAs ( Liao et al. , 2025 ; Pai et al. , 2025 ), most works still
focus on repeated demonstrations. Moreover, the environment generalization of DreamZero is retained even
after task-specific post-training, outperforming state-of-the-art VLAs by 10% on average task progress. Lastly,
we demonstrate two forms of cross-embodiment transfer. First, video-only demonstrations from another robot
(YAM) or humans yield a relative improvement of over 42% on unseen task performance for the target robot
(AgiBot G1) with just 10–20 minutes of data. Second, and more surprisingly, we show that DreamZero
enables few-shot embodiment adaptation : a model pretrained on AgiBot G1 adapts to an entirely new robot
(YAM) with only 30 minutes of play data, retaining zero-shot generalization. To the best of our knowledge,
this sets a new benchmark for data-efficient embodiment adaptation.

DreamZero is a 14B autoregressive diffusion transformer trained with a teacher-forcing chunk-wise video
denoising objective. Our architectural analysis reveals that larger pretrained video diffusion models produce
higher-quality video predictions, which directly translates to superior downstream action execution—indicating
that policy performance is fundamentally tied to video generation quality. We further find that diverse
distribution of the training data is essential for generalization, outperforming multi-task repetitive data with
the same amount of hours. Furthermore, we observe that autoregressive architectures lead to smoother robot
motions and higher modality alignment between predicted videos and executed actions.

To address the computational overhead inherent to video diffusion models, we introduce a suite of optimizations
spanning three categories: (1) algorithmic improvements, including decoupled video and action denoising
schedules ( DreamZero -Flash); (2) system-level parallelism and caching strategies; and (3) low-level opti-
mizations such as quantization, and CUDA kernel tuning. Collectively, these techniques achieve a 38× inference
speedup without degrading performance, enabling DreamZero to generate action chunks at approximately
7Hz for smooth, real-time robotic control.

Our main contributions are:

• We introduce DreamZero , a 14B WAM that jointly predicts video and actions, enabling effective learning
from diverse, non-repetitive robot data.

• We demonstrate over 2 × improvement in zero-shot generalization to unseen verbs and motions compared
to state-of-the-art VLAs, while retaining generalization across objects and environments.

• We present model and system optimizations achieving 38 × inference speedup , enabling real-time closed-
loop control at 7Hz .

• We demonstrate cross-embodiment transfer: video-only data from humans (12 minutes) or other robots (20
minutes) yields a relative improvement of over 42% on unseen tasks, and introduce few-shot embodiment
adaptation — DreamZero pretrained on AgiBot G1 adapts to an entirely new robot (YAM) with only 30
minutes of play data, enabling zero-shot generalization.

• We open-source our model weights, inference code, and code to run publicly available real-world (RoboArena)

3

# Page 4
World Action Models are Zero-shot Policies

Figure 3: Free-form Evaluation. DreamZero performs a diverse range of tasks when conditioned on natural
language instructions, including object manipulation, tool use, and human-robot interaction.

and simulation benchmarks (PolaRiS and Genie Sim 3.0) 1 at https://github.com/dreamzero0/dreamzero .

2. Related Work

2.1. Vision Language Action Models

Utilizing Foundation Models for Robotics. Developing foundation models ( Bommasani et al. , 2021 ) for
physical artificial intelligence has emerged as a significant research frontier. One line of work involves using
existing, pre-trained foundation models as “black-box” reasoners to handle high-level task planning. These works
usually involve modular systems, where the foundation models generate sequences of instructions, visual traces,
or affordances that are subsequently executed by specialized, low-level robotic policies or controllers ( Brohan
et al. , 2023 ; Driess et al. , 2023 ; Huang et al. , 2023 ; Kumar et al. , 2026 ; Singh et al. , 2023 ). While this
modularity simplifies complex planning and enables stronger generalization ( Kaelbling and Lozano-Pérez ;
Lee et al. , 2025 ; Li et al. , 2025 ) and efficiency ( Dreczkowski et al. , 2025 ), it is contingent upon having a
pre-existing library of low-level skills and a robust interface to bridge the gap between abstract reasoning and
physical execution. Additionally, these decoupled systems face the risk of compounding errors across modules.

VLAs. On the other hand, end-to-end models such as Vision-Language-Action models (VLAs) ( Bjorck et al. ,
2025 ; Black et al. , 2024 ; Brohan et al. , 2022 , 2023 ; Bu et al. , 2025 ; Gemini Robotics Team , 2025 ; Kim et al. ,
2024 ; Physical Intelligence , 2025 ; Yang et al. , 2025 ; Ye et al. , 2025 ; Zheng et al. , 2025 ), have gained popularity
by moving away from a rigid hierarchy of planning and control, combining language-conditioned semantics
and low-level robot actions within the same model. VLAs are often initialized from large vision-language (VLM)
models pre-trained on web-scale datasets. While pushing the frontier on visual-semantic knowledge transfer,
these models are pre-trained on static image-text datasets, which limits their ability to inherit spatiotemporal
priors required to transfer knowledge to new physical skills.

Generalization in VLAs. Generalization in VLAs has been mostly demonstrated on object and semantic
level ( Brohan et al. , 2023 ; Gao et al. , 2025 ) w hile generalization to completely new skills and environments has

1 Despite only being trained on ∼ 500 hours of real-world data, DreamZero shows non-trivial performance on Genie Sim 3.0, which is
a simulation benchmark comprised of 100 different tasks without being explicitly trained on the 10k hours of simulation training data.

4

# Page 5
World Action Models are Zero-shot Policies

remained limited. In particular, existing work utilizing VLAs achieves environment generalization by collecting
human teleoperation data across hundreds of diverse environments for specific tasks ( Physical Intelligence ,
2025 ). Furthermore, while current VLAs attempt to achieve task generalization by covering a large library
of language-conditioned motion primitives ( Gemini Robotics Team , 2025 ), this approach is fundamentally
constrained by the impracticality of capturing the vast amount of possible physical interactions and motions
with a fixed set of episode-level language-conditioned tasks. In contrast, video-based world models learn from
every consecutive frame pair in the data, while also leveraging large-scale video pretraining to understand
physical dynamics.

2.2. Video Model-based Robot Policies

Video Generation in Robotics. Prior works show that video generation models can be used to synthesize
robot trajectories and extract executable actions at test-time through various approaches: inverse-dynamics
models ( Du et al. , 2023 ; Zhou et al. , 2024 ), optical flow as dense correspondence ( Ko et al. , 2024 ), or
trajectory prediction as high-level planning ( Du et al. , 2024 ; Yang et al. , 2024 ). Other works generate human
videos—either with 3D tracking ( Liang et al. , 2024 ) or for novel scenes and motions ( Bharadhwaj et al. , 2024 ;
Chen et al. , 2025 )—and train policies using point tracking objectives. Most recently, ( Jang et al. , 2025 ; Luo
et al. , 2025 ) demonstrated that video generation models can produce synthetic robot data for unseen behaviors
in novel environments, leveraging the strong generalization capabilities of these models.

Joint Video and Action Generation. Another line of work couples video and action generation for end-to-end
learning. These methods demonstrate that incorporating a world modeling objective alongside action prediction
improves multi-task performance, sample efficiency, and generalization to novel scenes and objects. Previous
work ( Cheang et al. , 2024 ; Li et al. , 2025 ; Won et al. , 2025 ; Wu et al. , 2024 ; Zhao et al. , 2025 ; Zheng et al. ,
2025 ; Zhu et al. , 2025 ) learns to do joint world modeling and action prediction from scratch or from VLAs,
while more recent work ( Hu et al. , 2024 ; Kim et al. , 2026 ; Liang et al. , 2025 ; Liao et al. , 2025 ; Pai et al. , 2025 )
leverages pretrained video diffusion models to inherit rich visual dynamics priors. We refer to these models
collectively as World Action Models (WAMs) since they leverage world modeling capability (predicting the future
state) for action prediction. We use the term World Action Models (WAMs) rather than Video Action Models
(VAMs) to reflect that video is just one possible world modeling objective—future WAMs may align actions with
other predictive modalities such as tactile sensing, force feedback, or learned latent representations. In contrast
to prior WAMs, DreamZero systematically explores data diversity and scale to expose the full generalization
potential of WAMs, adopts an autoregressive architecture better suited for long-horizon world–action modeling,
achieves state-of-the-art generalization across both novel tasks and environments, and achieves state-of-the-art
cross-embodiment tranfer, both learning from different embodiments (video only) and few-shot adaptation to
a new embodiment.

Why WAMs. WAMs built upon video diffusion backbones inherit rich spatiotemporal priors from web-scale
data, capturing the best of both paradigms: the seamless gradient flow of end-to-end VLAs and dense world
modeling supervision for planning. Unlike latent world models ( Assran et al. , 2025 ; Hafner et al. , 2019 ,
2020 , 2023 ), which learn dynamics from scratch in compact latent spaces, WAMs leverage pretrained video
representations that already encode physical dynamics from internet-scale data. Central to this approach
is learning the joint distribution of video and action— DreamZero simultaneously learns both modalities,
with video prediction serving as an implicit visual planner that guides action generation. This formulation
not only means that improving robotic capabilities reduces to improving video generation, but also enables
three capabilities that elude current VLAs: zero-shot generalization to novel tasks, effective learning from
heterogeneous robot data, and extremely efficient cross-embodiment transfer from videos. We provide further
discussion about the differences between WAMs and alternative world model architectures (e.g., latent-space,
3D point cloud) in Appendix A .

5

# Page 6
World Action Models are Zero-shot Policies

Figure 4: Model Architecture of DreamZero . The model takes three inputs: visual context (encoded via a
VAE), language instructions (via a text encoder), and proprioceptive state (via a state encoder). These are
processed by an autoregressive DiT backbone using flow matching, which jointly predicts future video frames
and actions through separate decoders. During training (left), for each chunk, the model denoises noisy video
and action latents conditioned on clean video context. During inference (right), predictions are executed
asynchronously in the real world, and ground-truth observations are fed back into the KV cache to prevent
error accumulation.

3. DreamZero

Pretrained video diffusion models offer rich spatiotemporal priors from web-scale data, making them attractive
backbones for robot policies. However, converting these models into effective World Action Models (WAMs)
presents three key challenges: (1) Video-action alignment : jointly predicting video and actions requires tight
coupling between visual futures and motor commands, yet naively combining separate video and action heads
can lead to misalignment; (2) Architectural design : it remains unclear whether bidirectional or autoregressive
architectures are better suited for WAMs, with implications in modality alignment, error accumulation, and
inference efficiency; and (3) Real-time inference : video diffusion models require iterative denoising across
high-dimensional latent spaces, making them prohibitively slow for closed-loop control.

DreamZero addresses these challenges through three design choices. First, we train a single end-to-end
model that jointly denoises video and action with a shared objective, ensuring deep integration between
modalities. Second, we adopt an autoregressive architecture and exploit the closed-loop setting: after each
action chunk is executed, we replace predicted frames with ground-truth observations in the KV cache,
eliminating compounding errors while enabling efficient inference via KV caching and preserving native frame
rates for precise modality alignment (See right side of Figure 4 ). Third, we introduce a suite of system-,
implementation-, and model-level optimizations that achieve a 38 × inference speedup, enabling real-time
control at 7Hz. We detail the model architecture in Section 3.1 and real-time execution in Section 3.2 .

3.1. Model Architecture

Problem Formulation. DreamZero jointly predicts video o 𝑙 : 𝑙 + 𝐻 and actions a 𝑙 : 𝑙 + 𝐻 conditioned on language
instruction c , proprioceptive state q 𝑙 and visual observation including the current and the past history o 0: 𝑙
where 𝐻> 0 is a fixed horizon and 𝑙 is a random index sampled from a trajectory. Note that joint prediction of
video and action is a decomposition of (1) autoregressive video prediction and (2) action prediction from an
inverse-dynamics model (IDM):

𝜋 0 ( o 𝑙 : 𝑙 + 𝐻 , a 𝑙 : 𝑙 + 𝐻 | o 0: 𝑙 , c , q 𝑙 )
⏟
⏞
DreamZero

= 𝜋 0 ( o 𝑙 : 𝑙 + 𝐻 | o 0: 𝑙 , c , q 𝑙 )
⏟
⏞
video prediction

𝜋 0 ( a 𝑙 : 𝑙 + 𝐻 | o 0: 𝑙 + 𝐻 , q 𝑙 )
⏟
⏞
IDM

(1)

6

# Page 7
World Action Models are Zero-shot Policies

Instead of using two separate models (video prediction model and inverse dynamics model) to model the
decomposed objective ( Li et al. , 2026 ; Pai et al. , 2025 ), we train a single model end-to-end with joint prediction
objective. We believe that this end-to-end design enables better video-action alignment through a deep
integration between the two modalities. Since pretrained video models are already optimized on the video
prediction objective on diverse web-scale video data, DreamZero only needs to additionally learn to predict
videos for the robot embodiment videos and extract corresponding actions from the generated videos. We
further hypothesize that this encourages better generalization than the conventional practice of training VLA
from VLM , as our approach explicitly learns temporal dynamics from video frames used both as conditioning
inputs and prediction targets.

Model Architecture. The model architecture is shown in Figure 4 . To retain the generalization capability of
video models, we introduce minimal additional parameters: state encoders, action encoders, and decoders.
For robot training data that contains multiple views, we concatenate all views into a single frame instead of
making architectural changes to the backbone model.

In particular, DreamZero is trained to predict video frames and corresponding actions autoregressively.
Autoregressive generation possesses the following advantages: (1) it enables faster inference speed by utilizing
KV-cache, (2) the policy model can leverage the visual observation history as guidance for the next generation,
and (3) it avoids the modality alignment challenges (video, action, and language alignment) inherent to
bidirectional models. Concretely, bidirectional diffusion typically requires processing fixed-length sequences,
which often necessitates video subsampling that distorts native FPS, potentially harming video-action alignment.
On the other hand, autoregressive generation leverages KV caching to support arbitrarily long contexts within
a single forward pass. This preserves the native frame rate, ensuring precise alignment between video frames
and robot actions. Some details illustration of this difference is provided in Appendix B .

We introduce autoregressive modeling only for the video modality to avoid error propagation coming from
closed-loop action prediction. DreamZero is trained to predict video frames in a chunk manner; each chunk
has a fixed number of latent frames 𝐾 to match the action horizon. Chunk-wise generation enables training on
variable length of videos, similar to how LLMs are trained on variable length of language tokens. We provide
more details on the QKV attention masking strategy for the different modalities in Appendix C .

Training Objective. Similar to recent video diffusion models and VLAs, we employ flow-matching ( Albergo
et al. , 2023 ; Lipman et al. , 2022 ; Liu et al. , 2022 ) as the training objective ( Ali et al. , 2025 ; Team Wan , 2025 ;
Teng et al. , 2025 ). Unlike recent WAMs ( Kim et al. , 2026 ; Li et al. , 2025 ; Liao et al. , 2025 ; Zhu et al. , 2025 ),
DreamZero shares the denoising timestep between video and action modality for faster convergence at the
beginning of training. Also, we apply teacher forcing ( Gao et al. , 2024 ; Jin et al. , 2024 ) as a training objective;
the model is trained to denoise the noisy current chunk conditioned on the clean previous chunks.

Formally, given a chunk index 𝑘> 0 and the denoising timestep 𝑡 𝑘 ∈ [0 , 1] , we denote the corresponding noisy
video latent vector for original video o 𝑘 as z 𝑘
𝑡 𝑘 and noisy normalized actions as a 𝑘
𝑡 𝑘 . All frames within the same
chunk share the same timestep 𝑡 𝑘 , while different chunks are assigned independent timesteps. Our model
denoises z 𝑘
𝑡 𝑘 and a 𝑘
𝑡 𝑘 , defined as linear interpolations between clean vectors and random Gaussian noises:

z 𝑘
𝑡 𝑘 = 𝑡 𝑘 z 𝑘
1 + (1 − 𝑡 𝑘 ) z 𝑘
0 ,
a 𝑘
𝑡 𝑘 = 𝑡 𝑘 a 𝑘
1 + (1 − 𝑡 𝑘 ) a 𝑘
0 ,
(2)

where z 𝑘
0 ∼𝒩 ( 0 , I ) , a 𝑘
0 ∼𝒩 ( 0 , I ) , and z 𝑘
1 and a 𝑘
1 are a clean video latent vector and a normalized action,
respectively. Thus, the clean context from the previous chunks can be denoted as 𝒞 𝑘 = { ( z 𝑗
1 , a 𝑗
1 ) } 𝑘 − 1
𝑗 =1 .

We train the model u 𝜃 to predict the joint velocity for both modalities using the following flow-matching
objective:

ℒ ( 𝜃 ) = E z , a , { 𝑡 𝑘 }

[︃
1
𝐾

𝐾
∑︁

𝑘 =1
𝑤 ( 𝑡 𝑘 )
⃦⃦ u 𝜃 ([ z 𝑘
𝑡 𝑘 , a 𝑘
𝑡 𝑘 ]; 𝒞 𝑘 , c , q 𝑘 , 𝑡 𝑘 ) − v 𝑘 ⃦⃦ 2
]︃

,
(3)

7

# Page 8
World Action Models are Zero-shot Policies

where 𝑤 ( 𝑡 𝑘 ) > 0 is a predefined weight function for 𝑡 𝑘 , c is the text condition, q 𝑘 is the proprioceptive states of
𝑘 -th chunk, and the velocity v 𝑘 := [ z 𝑘
1 , a 𝑘
1 ] − [ z 𝑘
0 , a 𝑘
0 ] . To enable efficient training, we perform trajectory-level
updates and apply attention masking (e.g., see Figure 14 for details) so that the current noisy chunk can attend
to clean context of previous chunks. We provide the pseudo-code in Algorithm 1 .

Model Inference. As shown in Figure 4 , during inference, DreamZero jointly denoises video and action
chunks, leveraging KV caching for efficiency ( Huang et al. , 2025 ; Teng et al. , 2025 ; Yin et al. , 2025 ). Unlike
pure video generation, our closed-loop setting allows ground-truth observations to replace generated frames in
the KV cache after each action execution (see Figure 14 ). This eliminates the compounding error problem
inherent to autoregressive video generation—a key advantage unique to WAMs. Moreover, as a stateful policy,
DreamZero can leverage visual history for tasks requiring memory. 2 . We provide the pseudo-code of inference
in Algorithm 2

3.2. Real-time Execution of DreamZero

Diffusion-based WAMs inherit powerful generalization from video foundation models, but their iterative
denoising process creates a fundamental tension with reactive robotic control. We address two questions: (1)
What prevents WAMs from being reactive policies? (2) How do we resolve this for real-time control?

3.2.1. The Reactivity Gap

Reactive policies must respond to environmental changes within tens of milliseconds. A naive implementation
of DreamZero on a single GPU requires approximately 5.7 seconds per action chunk due to three bottlenecks:
(1) iterative denoising across 16 diffusion steps required for smooth actions, (2) the computational cost of a
14B parameter DiT backbone, and (3) sequential execution that blocks robot motion during inference. This
latency makes closed-loop control infeasible. 3

3.2.2. Asynchronous Closed-Loop Execution

Our first step towards resolving this is through asynchronous execution that decouples inference from action
execution. Rather than waiting for each inference to complete, the motion controller continuously executes the
most recent action chunk while inference runs concurrently on the latest observation. This structure transforms
the latency constraint from “inference must complete before the robot moves” to “inference must complete
before the current action chunk expires.” In our experiments, we deploy policies at an action horizon of 48
steps at 30Hz control frequency (1.6 seconds per chunk) for bimanual manipulation robots. Hence, we target
inference latency below approximately 200ms to ensure sufficient overlap for smooth, reactive control.

3.2.3. System-level Optimizations

Given the asynchronous execution structure, we optimize inference throughput through parallelism and caching.

• CFG Parallelism. Classifier-free guidance ( Ho and Salimans , 2022 ) requires two forward passes (condi-
tional and unconditional). We distribute these across two GPUs, reducing per-step latency by 47%.
• DiT Caching. We exploit the directional consistency of velocity predictions during flow matching. When
cosine similarity between successive velocities exceeds a threshold, we reuse cached velocities, reducing
effective DiT steps from 16 to 4 with minimal quality loss on action prediction.

3.2.4. Implementation-level Optimizations

We further reduce latency through compiler and kernel enhancements.

2 In this work, we do not explicitly evaluate or post-train DreamZero on tasks that can only succeed with memory. We leave this for
future work.
3 One might expect that generating only actions (not video) would accelerate inference, but at 14B scale we empirically found out
that the speed gain is minimal—the number of diffusion steps and the number of DiT blocks dominate latency. Moreover, because video
and action are jointly trained for strong cross-modal alignment, naively reducing action denoising steps degrades quality. This motivates
DreamZero-Flash.

8
