# Page 1
LDA-1B: Scaling L atent D ynamics A ction Model
via Universal Embodied Data Ingestion

Jiangran Lyu ∗ 1 , 2 , Kai Liu ∗ 2 , 3 , 4 , Xuheng Zhang ∗ 1 , 2 , Haoran Liao 2 , 6 , Yusen Feng 1 , 2 , Wenxuan Zhu 1 , Tingrui Shen 1 ,
Jiayi Chen 1 , 2 , Jiazhao Zhang 1 , 2 , Yifei Dong 1 , Wenbo Cui 2 , 3 , 4 , Senmao Qi 2 , Shuo Wang 2 , Yixin Zheng 2 , 3 , 4 , Mi Yan 1 , 2 ,
Xuesong Shi 2 , Haoran Li 3 , Dongbin Zhao 3 , Ming-Yu Liu 7 , Zhizheng Zhang 2 , † , Li Yi 5 , † , Yizhou Wang 1 , † , He Wang 1 , 2 , †

1 Peking University
2 Galbot
3 CASIA
4 BAAI
5 Tsinghua University
6 Sun Yat-sen University
7 NVIDIA

Code & Data: https://pku-epic.github.io/LDA ∗ Equal contribution
† Corresponding authors

Fig. 1: We introduce LDA-1B, a 1.6 B-parameter robot foundation model scaled on over 30k hours of heterogeneous embodied
data. LDA-1B unifies policy, dynamics, and visual forecasting in a structured DINO [50] latent space, allowing different data
sources to play complementary roles. Beyond high-quality data alone, noisy data and actionless videos also provide valuable
visual and physical priors for dynamics learning. This universal data ingestion paradigm enables stable scaling with data and
model size, significantly outperforming strong baselines such as π 0 . 5 [26] across diverse manipulation tasks.

Abstract —Recent robot foundation models largely rely on
large-scale behavior cloning, which imitates expert actions but
discards transferable dynamics knowledge embedded in hetero-
geneous embodied data. While the Unified World Model (UWM)
formulation has the potential to leverage such diverse data,
existing instantiations struggle to scale to foundation-level due
to coarse data usage and fragmented datasets. We introduce
LDA-1B, a robot foundation model that scales through universal
embodied data ingestion by jointly learning dynamics, policy,
and visual forecasting, assigning distinct roles to data of varying
quality. To support this regime at scale, we assemble and
standardize EI-30k, an embodied interaction dataset comprising
over 30k hours of human and robot trajectories in a unified
format. Scalable dynamics learning over such heterogeneous data
is enabled by prediction in a structured DINO latent space, which
avoids redundant pixel-space appearance modeling. Complement-
ing this representation, LDA-1B employs a multimodal diffusion
transformer to handle asynchronous vision and action streams,
enabling stable training at the 1B-parameter scale. Experiments
in simulation and the real world show LDA-1B outperforms prior
methods (e.g., π 0 . 5 ) by up to 21%, 48%, and 23% on contact-rich,
dexterous, and long-horizon tasks, respectively. Notably, LDA-1B
enables data-efficient fine-tuning, gaining 10% by leveraging 30%
low-quality trajectories typically harmful and discarded.

I. I NTRODUCTION

Inspired by the success of Large Language Models (LLMs)
and Vision-Language Models (VLMs), the robotics commu-
nity has increasingly pursued general-purpose robot founda-
tion models through large-scale pretraining [7, 44]. Most exist-
ing approaches center on scaling behavior cloning (BC), which
imitates expert actions but fundamentally restricts learning
to high-quality demonstrations. Consequently, a large portion
of heterogeneous embodied data [46] is discarded or only
weakly utilized, despite containing rich physical interaction
dynamics [29].
Unified World Model (UWM) formulation [33, 65] provides
an alternative by jointly optimizing dynamics, policy, and
video generation within a single model, which can leverage
not only expert data. Despite the potential value, existing
UWM instantiations remain far from scaling to foundation-
level. A major limitation lies in coarse data usage: hetero-
geneous embodied data are often treated uniformly, without
differentiating their roles by quality or supervision, which
underutilizes transferable dynamics knowledge. In addition,
the community lacks ready-to-use large-scale datasets that

arXiv:2602.12215v2  [cs.RO]  3 Jun 2026

# Page 2
unify varying-quality data with consistent formats and aligned
action representations. Furthermore, UWM represents future
state in pixel space, entangling dynamics learning with redun-
dant appearance modeling. Subtle variations in illumination,
texture, background clutter, or camera viewpoint can dominate
the training objective, making large-scale training inefficient
and hindering the learning of interaction-relevant dynamics.
To overcome these limitations, we introduce LDA-1B , a
robot foundation model that scales via universal embodied
data ingestion . In this framework, heterogeneous data play
distinct yet complementary roles: actionless human videos
supervise visual forecasting [41, 40, 28], lower-quality trajec-
tories primarily inform dynamics learning, and high-quality
trajectories support both policy and dynamics. To realize
this approach at scale, we assemble EI-30k , a large-scale
embodied interaction dataset with over 30k hours of human
and robot trajectories across real and simulated environments,
standardized in format and aligned in action representation.
Scalable learning on such diverse data is facilitated by a
structured DINO latent space [50, 64, 25], which reduces re-
dundant appearance modeling [41, 28], and a multimodal dif-
fusion transformer that aligns asynchronous visual and action
prediction. By combining this ingestion strategy, dataset, latent
representation, and model architecture, LDA-1B achieves sta-
ble training at the 1B-parameter scale while maximizing data
utilization.
We
evaluate
LDA-1B
on
challenging
RoboCasa-GR1
benchmark and a diverse set of real-world tasks involving both
grippers and high-DoF dexterous hands [61]. LDA-1B consis-
tently outperforms π 0 . 5 , achieving 21% gains on contact-rich
manipulation, benefiting from improved dynamics understand-
ing and 48% gains on dexterous manipulation, benefiting from
effective utilization of human data. Moreover, under a mixed-
quality fine-tuning setting, LDA-1B improves data efficiency
by 10% through leveraging low-quality trajectories that are
detrimental to baseline methods. These results highlight uni-
versal embodied data ingestion and unified latent dynamics
learning as a scalable alternative to behavior-cloning-centric
robot pretraining. In summary, our contributions are threefold:

• We propose LDA-1B, a scalable robot foundation model
that learns generalizable interaction dynamics through
unified latent dynamics pretraining.

• We construct EI-30k, a large-scale embodied interac-
tion dataset covering diverse embodiments, environments,
data qualities, with aligned end effector coordinate sys-
tem.

• We demonstrate that LDA-1B achieves superior gen-
eralization and robustness across a wide range of set-
tings, including simulation and real-world environments,
contact-rich manipulation, dexterous manipulation, and
long-horizon manipulation.

II. R ELATED W ORK

Robot Foundation Models. Recent robot foundation models
predominantly adopt the Behavior Cloning paradigm. As sum-
marized in Table I, representative approaches including π 0 [6],

Model
Data Src. #Data Action Quality
Train.
Param.

π 0 . 5 [26]
Tele.
10k+
High
BC
3B
RDT [35]
Tele.
< 10k
High
BC
1B
GraspVLA [16]
Sim.
20k+
High
BC
2B
InternVLA-M1 [13]
Sim.
< 10k
High
BC
3B
Being-H0 [38]
Hum.
< 10k
Mixed
Aln. + BC
14B
InternVLA-A1 [11]
Het.
10k+
High
VF + BC
3B
GR00T-N1.6 [5]
Het.
< 10k
Mixed
LA + BC
1B
UniVLA [9]
Het.
< 10k
Mixed
LA + BC
7B

LDA-1B
Het.
30k+
Mixed
UWM [65]
1B

TABLE I: Comparison of Representative Robot Founda-
tion Models. This table compares the proposed LDA with
recent robot foundation models in terms of data source,
data quantity, action quality, training paradigm, and the
number of trainable model parameters (excluding frozen
components). Data source abbreviations are as follows:
Tele.=teleoperation, Sim.=simulation, Hum.=human demon-
stration, and Het.=heterogeneous data. Training paradigm ab-
breviations include: BC=behavior cloning, VF=visual fore-
sight,
Aln.=alignment,
LA=latent
action
modeling,
and
UWM=unified world model. Only embodied interaction data
are considered, excluding internet-scale VQA data.

RDT [35], and InternVLA [13] rely heavily on high-quality
teleoperation or simulation data, which fundamentally con-
strains their scalability. Hybrid methods such as Being-H0 [38]
and UniVLA [9] attempt to incorporate heterogeneous data
with mixed quality; however, they largely depend on action
alignment or auxiliary pretrained latent action models, limiting
the effective data scale to around 6k hours of embodied data.
In contrast, LDA-1B breaks this ceiling by adopting a unified
world model formulation, enabling efficient ingestion of up to
30k hours of mixed-quality embodied data.
Unified Video Action Models. Recent works have explored
joint modeling of dynamics and policy for embodied deci-
sion making. Methods such as DyWA [39], FLARE [63],
DreamVLA [59] and the WorldVLA series [12, 27] demon-
strate that co-training next-state prediction and policy learning
can improve generalization in interactive environments. To
enrich dynamics modeling, UWM [65] and UVA [33] fur-
ther propose optimizing multiple objectives jointly, including
video generation, forward and inverse dynamics, and action
prediction. Concurrent with our work, Motus [4] adopts the
UWM paradigm and integrates priors from pretrained VLM
and video generation models. Despite their promising results,
these approaches typically operate directly in pixel space and
do not explicitly consider the roles of data quality, scale, or
heterogeneity during training, which limits their ability to fully
exploit large-scale, mixed-quality interaction data for robust
dynamics learning.
Large-Scale Embodied Interaction Datasets. The progress
in embodied AI relies on large-scale embodied datasets. Many
widely used datasets are collected via teleoperation on real
robots [6, 26, 30, 34] or generated in simulation [16, 13], pro-
viding high-quality action-labeled trajectories. Beyond robot-

# Page 3
Multi-Modal
Diffusion Transformer

DINO Future
Action Chunk

noise
Current DINO+ noise

Linear Projection

Policy

𝒑(𝒂|𝒐, 𝒍)

𝒐

𝒂

Visual Planning

𝒑(𝒐 ! |𝒐, 𝒍)

𝒐

𝒐 )

Time

𝒑(𝒂|𝒐, 𝒐 ! , 𝒍)

𝒐

𝒐 )

𝒂

Inverse Dynamics

𝒑(𝒐 ! |𝒐, 𝒂, 𝒍)

𝒐 )

𝒐

𝒂

Forward Dynamics

𝑻
𝑻

x N

AdaLN
AdaLN
Task & Time

Embedding

Multi-Modal Self-Attention

QKV

+
+

LayerNorm
LayerNorm

Cross
Attention

Cross
Attention
VLM
Tokens
KV

+
+

FFN
FFN

KV

QKV

MM-DiT Block

AdaLN

Conditioning Input

Language
VLM Tokens

Diffusion
Timestep t

Sinusoidal

Encoder
𝑓 *

Task
Selection
Task Embeddings
+

VLM

Obs

OR
OR

register
register

dense/sparse sampling

Linear Projection

Fig. 2: Architecture of LDA. LDA jointly denoises action chunks and future visual latents under multiple co-training objectives,
including policy learning, forward dynamics, inverse dynamics, and visual forecasting. Conditioned on VLM tokens, diffusion
timesteps, and task embeddings, the model adopts a multimodal diffusion transformer architecture, where action and visual
experts are decoupled and interact through a shared self-attention layer.

collected data, recent works explore human-centric embod-
ied datasets, such as egocentric recordings with hand ac-
tions [56, 38, 19]. While these datasets significantly expand
data diversity, many are either not publicly released or provide
limited action supervision, making them difficult to directly
integrate with robot learning pipelines. More broadly, existing
embodied datasets are highly fragmented: some are closed-
source, others are open but vary substantially in data formats,
sensor configurations, action representations, and annotation
quality. This lack of standardization poses a major obstacle to
large-scale data aggregation and unified training. In contrast,
our work introduces EI-30k, a large-scale embodied interac-
tion dataset that unifies diverse data sources including robot
and human trajectories from both real-world and simulated
environments under consistent data formats and aligned action
representations.

III. L ATENT D YNAMICS A CTION M ODEL

A. Preliminary: Unified World Models

Given the current observation o t (typically an RGB image),
UWM [65] jointly models multiple conditional distributions
over future observations o t +1: t + k and action chunk a t +1: t + k ,
enabling unified learning of:

1) Policy: p ( a t +1: t + k | o t )
2) Forward Dynamics: p ( o t +1: t + k | o t , a t +1: t + k )
3) Inverse Dynamics: p ( a t +1: t + k | o t : t + k )
4) Visual Planning: p ( o t +1: t + k | o t )
Concretely, UWM [65] instantiates this framework using a
joint diffusion model that predicts noise for both actions and
future observations:

( ϵ θ
a , ϵ θ
o ) = s θ
 
o, a t a , o ′
t o , t a , t o ′ 
,

where t a and t o are independently sampled diffusion timesteps
for actions and observations, and ˜ a t a , ˜ o t o denote their corre-
sponding noisy inputs. The model is trained with a standard

DDPM [23] objective, jointly denoising future actions and
observations conditioned on o t . We further extend this formu-
lation by introducing language ℓ conditioning through a VLM,
enabling instruction-guided action and observation prediction.

B. Universal Data Ingestion via Multi-task Co-training

We adopt a universal data ingestion regime to jointly train
the unified objectives described above, allowing heterogeneous
embodied data to contribute according to their supervision
quality. Specifically, high-quality robot and human demon-
strations are co-trained with all objectives, supporting both
action policy learning and dynamics modeling. Lower-quality
trajectories, which may contain suboptimal or noisy actions,
are used exclusively for dynamics and visual forecasting,
where accurate action optimality is not required. In addition,
we leverage large-scale human manipulation videos without
action annotations to train the visual forecasting objective,
providing supervision for instruction-conditioned future state
prediction. This role-aware data usage prevents overfitting
to expert-only behaviors and enables scalable learning of
transferable dynamics and action representations.
To implement differentiated objectives within a single dif-
fusion model, we introduce four learnable task embeddings
and two learnable register tokens . Each task embedding corre-
sponds to a specific training objective (policy, forward dynam-
ics, inverse dynamics, or visual forecasting) and is added to
the diffusion timestep embedding f t to condition the denoising
process. The learnable register tokens, one for action and one
for visual state, serve as placeholders for modalities that are
absent in a given task. For example, during policy training, the
model receives noisy action tokens along with a visual register
token representing the unobserved future state; in contrast,
visual forecasting uses noisy future visual tokens with an
action register token. This design enables a unified architecture
to flexibly support different input-output structures without

# Page 4
modifying the network topology. Overall, the model predicts
a denoising vector field v θ
a under different task conditions and
is trained using a flow-matching objective:

l θ
action = E ( o t : t + k , a t +1: t + k ,ℓ ) ∼D
τ a ∼U (0 ,T τ )
ϵ a ∼N ( 0 , I )

v θ
a − ( ϵ a − a t +1: t + k )
2

2 ,

l θ
obs = E ( o t : t + k , a t +1: t + k ,ℓ ) ∼D
τ o ∼U (0 ,T τ )
ϵ o ∼N ( 0 , I )

v θ
o − ( ϵ o − o t +1: t + k )
2

2 ,

l θ = l θ
action + l θ
obs .
(1)
During training, action and visual losses are selectively
activated according to the task specification, allowing het-
erogeneous data to contribute under appropriate supervision.
At inference time, the same model can be flexibly invoked
for different objectives by specifying the task embedding and
corresponding inputs.

C. Representation of Predictive Targets

We represent predictive targets, future visual states and
actions, in a unified format to maximize knowledge sharing
across heterogeneous datasets. For visual prediction, we adopt
latent features extracted from a pretrained DINO [50] encoder,
rather than VAE-based pixel-space representations. DINO la-
tents encode high-level semantic and spatial structure while
suppressing background noise and low-level visual variations,
which facilitates learning scene dynamics that generalize
across diverse environments and object configurations.
For actions, we define a unified hand-centric action space
based on end-effector motion, consisting of delta wrist poses
and finger configurations. For parallel-jaw grippers, the finger
state is represented by a single degree-of-freedom gripper
width, while for multi-finger dexterous hands, finger config-
urations are described using keypoints expressed in the wrist
coordinate frame. This design enables consistent action model-
ing across different embodiments and manipulation platforms.
To model temporal dynamics, visual states and actions are
organized as two synchronized temporal streams with different
sampling rates. Visual observations are sampled at 3 Hz, a
lower frequency than actions, 10 Hz. This reduces redundant
computation from highly correlated consecutive frames while
preserving fine-grained action dynamics, allowing the model
to maintain coherent temporal alignment between fast-varying
control signals and slower-evolving visual states.

D. Architecture: MM-DiT

We adopt a Multimodal Diffusion Transformer (MM-DiT)
to jointly denoise action chunks and predict future visual
features within a unified diffusion framework (Fig. 2). The
model operates on heterogeneous tokens while sharing a com-
mon Transformer backbone. Conditioning inputs include the
current observation, language instruction, diffusion timestep,
and task specification. Observations and language are encoded
by a pretrained VLM into conditioning tokens. The diffusion
timestep is encoded using a sinusoidal embedding, and task
information is represented by a learned task embedding. All

conditioning signals are injected into each Transformer block
via adaptive layer normalization (AdaLN [47]).
Actions are organized as fixed-length chunks and corrupted
with Gaussian noise. Future visual features (DINO [50] fea-
tures) are noised in parallel. Both modalities are projected
into token embeddings through modality-specific linear layers
and processed jointly by MM-DiT. Each MM-DiT block
applies multimodal self-attention over concatenated action
and visual tokens, enabling cross-modal interaction. Modality-
specific QKV projections and FFNs are retained to preserve
inductive biases, while attention is shared across modalities.
Language tokens are incorporated via cross-attention to pro-
vide high-level semantic guidance. Finally, modality-specific
output heads predict denoised action sequences and future
visual features.

E. Pre-training and Post-training

Pre-training Configurations. Our model is trained on a
server cluster equipped with 48 NVIDIA H800 GPUs. The
training process contains 400k iterations, resulting in a total
computational cost of 4,608 GPU hours. To preserve the
generalization capability and visual representation quality of
the pre-trained foundation models, we keep the parameters of
the VLM [55] and the DINO [50] encoder frozen throughout
the pre-training process, updating the MM-DiT and action
encoder/decoder. This design ensures that the model can learn
from new data without degrading the core abilities of the base
models in cross-modal understanding and fine-grained visual
feature extraction.
Data-Efficient Fine-tuning. To adapt the model to target
embodiments and tasks for real-world deployment, we in-
troduce a lightweight post-training stage. This stage follows
the same data regime as pretraining and effectively leverages
naturally collected teleoperation data of mixed quality, without
requiring expert-level demonstrations. Compared to prior fine-
tuning pipelines that rely on carefully curated expert datasets,
our method directly utilizes unfiltered teleoperation data, sub-
stantially improving data efficiency and reducing the cost of
data collection and annotation, thereby facilitating practical
deployment.

IV. E MBODIED I NTERACTION D ATASET (EI-30 K )

We introduce the Embodied Interaction Dataset (EI-30k),
a large-scale collection of embodied interaction trajectories
totaling over 30k hours. It consists of 8.03k hours of real-
world robot data, 8.6k hours of simulated robot data, 7.2k
hours of human demonstrations with actions, and 10k hours
of actionless human videos. All subdatasets are annotated
with explicit quality labels, enabling systematic analysis across
different fidelity levels and supporting quality-aware learning.
Data Unification. EI-30k consolidates datasets from hetero-
geneous platforms and tasks, which vary in storage formats,
sensor modalities, and annotations. All data are converted
into the LeRobot format, providing a unified representation of
observations, actions, and language. This standardization fa-
cilitates plug-and-play training, flexible data composition, and

# Page 5
Fig. 3: Aligned End Effector Coordinate Systems. We manually align coordinate frames across diverse robot and human
embodiments to ensure consistency. This shared representation enables joint learning from heterogeneous interaction data.

Fig. 4: Statistics of EI-30k. The dataset contains more than
30k hours of diverse human and robot interaction data (right).
It spans varying episode lengths (left) and a rich set of
manipulation tasks (center).

seamless integration of additional annotations, while greatly
reducing engineering overhead for handling diverse sources.
Aligned Action Representation. To support consistent model-
ing of physical interactions across embodiments, all available
action annotations are expressed as hand-centric motion in a
shared coordinate frame (Fig. 3). For robots, this includes the
6-DoF end-effector pose plus gripper width or dexterous hand
joints. For humans, the 6-DoF wrist pose and full MANO [49]
hand parameters are recorded. Camera extrinsics are retained
to decouple hand motion from egocentric head motion. All
coordinate frames are manually aligned to ensure geometric
consistency across datasets, enabling joint learning from both
human and robot trajectories.
Quality Annotation and Cleaning. EI-30k applies systematic
cleaning and quality-aware annotation. Language annotations
are normalized using a vision-language model to ensure
semantic consistency. Motion segments without meaningful
hand-object interaction are removed, e.g., head-only or idle
segments in egocentric videos. Each trajectory is assigned a
quality label based on action accuracy, and annotation com-
pleteness. Unlike aggressive filtering, low-quality trajectories
are preserved, allowing downstream models to exploit the full
spectrum of data through quality-aware training.

V. E XPERIMENTS

A. Simulation Experiments

Benchmark and Baselines. We evaluate our method on
RoboCasa-GR1 [42], a simulated kitchen benchmark featuring

Model
Vis. Rep. MMDiT
VLM
Success Rate ↑

GR00T-N1.6 [5]
-
-
Cosmos [2, 43]
47.6
StarVLA [14, 57]
-
-
Qwen3-VL [55]
47.8
GR00T-EI10k
-
-
Qwen3-VL
51.3

UWM-0.1B [65]
VAE
✗
-
14.2
UWM-1B
VAE
✗
Qwen3-VL
19.3
UWM(MM-DiT)
VAE
✓
Qwen3-VL
20.0
LDA(DiT)
DINO
✗
Qwen3-VL
48.9
LDA-0.5B
DINO
✓
Qwen3-VL
50.7
LDA-1B
DINO
✓
Qwen3-VL
55.4

TABLE II: Results on RoboCasa-GR1 [42] and impact of state
representation (VAE vs. DINO [50]), model size, and the MM-
DiT architecture on task success rates.

24 tabletop rearrangement and articulated-object manipulation
tasks with the GR-1 humanoid robot and Fourier dexterous
hands. The benchmark provides challenging and realistic
settings that require high-DoF dexterous manipulation from
egocentric RGB observations captured by a head-mounted
camera. Following the GR00T [5] evaluation protocol, we fine-
tune all models using 1,000 trajectories per task and evaluate
each task with 51 trials, reporting average success rates. We
compare LDA against GR00T and its strong variants, as well
as UWM [65], under matched training paradigms and data.
To ensure a fair comparison in terms of model capacity and
pretraining, we reproduce a strong GR00T baseline (denoted as
GR00T-EI10k) with 1B parameters, pretrained on our curated
EI-30k high-quality subset and using Qwen3-VL as the VLM
encoder.
Comparison with Baselines. As shown in Table II, the orig-
inal GR00T-N1.6 [5] with 3B parameters achieves a success
rate of 47.6%. When pretrained on our curated EI-30k dataset,
the reproduced GR00T-EI10k with 1B parameters shows a
clear improvement, reaching 51.3%, highlighting the impact
of high-quality embodied data. Under the same parameter
budget, LDA further improves the success rate to 55.4%.
These results indicate that, beyond data quality and parameter
scaling, jointly learning actions and dynamics within a unified
model provides additional gains when pretrained on mixed-
quality data.
Ablation Study. We further analyze key design choices under
identical training data and optimization settings. UWM [65],
despite jointly predicting actions and dynamics, achieves only
14.2% success due to limited model capacity and the use

# Page 6
Use the tongs to put the cake on the plate

Bimanually remove the lid.

Pull the nail out using the claw hammer.

Sweep the crumbs into the dustpan.

Wipe whiteboard.

Flip the box.
Fig. 5: Real-World Manipulation Demonstrations Across Multiple Robotic Platforms and End-Effectors. Galbot G1
equipped with a Sharpa dexterous hand (top-left), Unitree G1 with a BrainCo dexterous hand (middle and bottom-left), and
Galbot G1 with a two-finger gripper (right).

Put the pepper into the box
pick and handover and place
0

20

40

60

80

Success Rate (%)

40.0

50.0
50.0

70.0

80.0

90.0
Pick & Place

Flip the box
Wipe the board

20.0

52.0

40.0

44.0

60.0

72.0

Contact-rich Manipulation

Watering the flower
Knock the specific block with hammer

40.0

33.3

60.0

40.0

80.0

55.0

Fine Manipulation

Sweep the table
Clean the rubbish

35.0

0.0

53.0

0.0

65.0

35.0

Long-horizon Manipulation

GR00T-N1.6
0.5
Ours
Fig. 6: Success Rate Comparison on Real-World Gripper Manipulation Tasks. All models are few-shot fine-tuned on Galbot
and evaluated on eight tasks spanning Pick & Place, Contact-rich, Fine, and Long-horizon manipulation. LDA consistently
outperforms GR00T-N1.6 [5] and π 0 . 5 [26].

of entangled VAE latent representations. Scaling UWM to
1B parameters or replacing its DiT backbone with our MM-
DiT yields only marginal improvements (19.3% and 20.0%,
respectively), suggesting that architectural constraints funda-
mentally limit its performance. In contrast, replacing pixel-
space VAE latents with DINO [50] representations leads to a
substantial performance gain (20.0% → 55.4%), highlighting
the importance of semantically structured latent spaces for
effective scaling. Finally, removing the proposed MM-DiT
architecture or reducing the model size to 0.5B parameters
results in performance drops of 6.5% and 4.7%, respectively,
confirming the effectiveness of the multi-expert design and its
favorable scaling behavior.

B. Real-world Experiments

To validate the scalability and robustness of LDA-1B, we
conduct extensive real-world experiments focusing on few-
shot adaptation to new embodiments, dexterous manipulation,
and data efficiency under mixed-quality supervision.
Real-World Robot and Task Setup. We evaluate our method
on two humanoid platforms: Galbot G1 and Unitree G1.
Galbot G1 is equipped with either a two-finger gripper or

22-DoF Sharpa dexterous hands, while Unitree G1 uses 10-
DoF BrainCo hands. Across all configurations, the policy
receives only egocentric RGB observations from a head-
mounted camera. We evaluate four categories of manipulation
tasks under the gripper setting, Pick and Place , Contact-
rich Manipulation , Fine Manipulation , and Long-horizon Ma-
nipulation , covering diverse contact dynamics and temporal
horizons. Representative tasks include Beat Block, Flip Box,
Handover, Pick-and-Place (Pepper), Sweep Table, Clean Rub-
bish, Water Flower, and Wipe Board . Dexterous manipulation
further includes tool-use tasks such as pulling a nail with a
hammer and flipping bread with a spatula, which require pre-
cise force control and coordinated finger motion. Qualitative
demonstrations are shown in Fig. 5. For each task, we collect
100 teleoperated trajectories without enforcing expert-level
execution. As a result, the dataset naturally exhibits mixed
quality: approximately 50–80% of trajectories correspond to
expert behavior, while the remainder contain suboptimal ac-
tions such as pauses, retries, or inefficient motion patterns.

Baselines and Fine-tuning Protocol. We compare LDA-1B
against two strong baselines, π 0 . 5 [26] and GR00T [5]. To en-
sure stable and competitive performance, baseline models are

# Page 7
Pick Bottle

Open Macbook

Pull Nail

Pick Bread

Flip Bread

0

20

40

60

80

100

Success Rate (%)

Low DoFs Hand
High DoFs Hand

75

40
40

20

10

20

100

0

10
10

90

100

80

70

90

GR00T-N1.6
0.5
Ours

Fig. 7: Success Rate Comparison on Real-World Dexterous
Manipulation Tasks. We evaluate the real-world performance
of our model against baselines (GR00T-N1.6 and π 0 . 5 ) on 3
low-DoF hand (BrainCo) tasks and 2 high-DoF hand (Sharpa)
tasks. Ours (dark blue) consistently outperforms baselines,
especially on fine dexterous tasks (pulling nails) and high-
DoF tasks.

Method
Pick & Place

Object Background OOD Pos.

π 0 . 5
26.7
20.0
6.7
GR00T
40.0
40.0
20.0
Ours
60.0
60.0
40.0

TABLE III: Robust Generalization under visual and spatial
perturbations. LDA-1B achieves 60.0% success on unseen
objects and backgrounds, and 40.0% under OOD positions,
demonstrating effective focus on task-critical affordances over
visual noise through latent dynamics pretraining.

finetuned exclusively on the filtered expert subset. In contrast,
LDA-1B leverages all collected trajectories and learns directly
from the full mixed-quality distribution via our Universal
Embodied Data Ingestion mechanism.
Results on Gripper Manipulation. We first evaluate few-shot
adaptation by deploying LDA-1B on the Galbot G1, which is
excluded from our EI-30k pretraining dataset. As shown in
Fig. 6, LDA-1B consistently outperforms all baselines across
task categories. On simple pick-and-place tasks, LDA-1B
achieves success rates of 80%–90%, indicating effective few-
shot adaptation to a new robot embodiment. The performance
gap widens substantially in contact-rich and long-horizon
scenarios. For instance, the Clean the Rubbish task requires
coordinated dual-arm manipulation, tool usage (dustpan), and
sequential object transfer into a trash bin, where errors can
easily accumulate over time. In this setting, LDA-1B achieves
a 35% success rate, while both GR00T and π 0 . 5 fail entirely
(0%). This result suggests that latent dynamics modeling en-
ables LDA to better anticipate action-induced state transitions,
maintain temporal consistency, and recover from intermediate
failures in extended manipulation sequences.
Results on Dexterous Manipulation. We further evaluate
LDA-1B on both low-DoF and high-DoF dexterous manip-
ulation tasks, as reported in Fig. 7. On low-DoF tasks such
as Pull Nail , which requires precise motion direction and

Method
Place the pen into the box
Bimanually remove the lid

63 High 63 High + 37 Low 66 High 66 High + 34 Low

π 0 . 5
60
40 (20 ↓ )
50
40 (10 ↓ )
Ours
70
80 (10 ↑ )
50
60 (10 ↑ )

TABLE IV: Data-efficient mixed-quality fine-tuning. LDA-
1B improves success rates by +10% on both tasks when
incorporating low-quality trajectories, while π 0 . 5 degrades
significantly, demonstrating effective utilization of noisy data
for enhanced generalization.

stable contact maintenance between the hammer and the nail,
LDA-1B achieves 80% success, reliably localizing targets
and adjusting sensitive actions, whereas π 0 . 5 largely fails.
On high-DoF tasks such as Flip Bread , which involve high-
dimensional control, continuous contact, and coordinated wrist
motion, LDA-1B attains 90% success, while π 0 . 5 reaches only
10%. These results demonstrate that pretraining on large-scale
human data provides strong latent priors for dexterous control,
enabling precise finger coordination and object reorientation
with limited robot data. In contrast, baseline policies struggle
to generalize as action dimensionality and contact complexity
increase.
Generalization Ability. To evaluate the generalization of our
policy, we test pick and place task under three conditions:
novel objects, unseen backgrounds, and out-of-distribution
(OOD) starting position, shown in Fig. 8. As summarized in
Table III, our model maintains high success rates despite visual
and spatial perturbations. The large-scale latent dynamics
pretraining allows the model to ignore visual distractors (back-
ground changes) while focusing on relevant object affordances,
demonstrating strong generalization relative to baselines.
Data-Efficient Fine-tuning. We analyze the value of mixed-
quality data ingestion during the fine-tuning stage by post-
training on two splits: (1) High-Quality Only (expert data),
and (2) High + Low Quality (all 100 trajectories). As shown
in Table IV, while baseline models degrade when low-quality
data is added, LDA-1B effectively leverages these noisy
trajectories, boosting performance by 10 percentage points,
substantially improving data efficiency and reducing the cost
of data collection and annotation for practical deployment.

C. Analysis of Design Choices for Model Scaling

To analyze the scaling behavior of LDA, we systematically
vary model capacity, data composition, and training objectives.
All models are evaluated on an unseen test set sampled from a
held-out subset of Agibot World [8]. We report the action pre-
diction L1 error as the primary metric, which serves as a stable
and reproducible proxy for real-world performance. Fig. 10
summarizes the results under four training configurations: (i)
Policy Only , (ii) Policy + Visual Forecasting , (iii) Policy with
Forward and Inverse Dynamics , and (iv) the full co-training
framework ( Ours ). These experiments jointly reveal how LDA
scales under heterogeneous supervision and increasing model
capacity.

# Page 8
Training
OOD-Position
Novel Objects
Variant Background

Fig. 8: Generalization evaluation setup on Pick and Place task

Fig. 9: Visualization of latent forward dynamics. Our model generates accurate future visual representations (top) aligned
with ground truth (bottom) across time steps, capturing semantic object structure and motion dynamics.

Fig. 10: Scaling Analysis of LDA, evaluated by action predic-
tion error on an unseen test set. Top: Action prediction error
decreases to 6.6 with 30k hours of training data, demonstrating
effective utilization of diverse data sources. Bottom: LDA con-
sistently outperforms UWM across model sizes (0.1B → 1B)
with increasing training data.

Effectiveness of Universal Data Ingestion. Effectively lever-
aging heterogeneous embodied data requires jointly scaling
both data sources and training objectives. As shown in Fig. 10,
LDA achieves its best performance only when all supervi-
sion signals-policy learning, dynamics modeling, and visual
forecasting-are optimized together. When either the data scale
or the training objectives are reduced, performance degrades
noticeably. Using only action-labeled trajectories with a Policy
Only objective (grey line), increasing the dataset size yields
unstable behavior: while moderate scaling initially reduces

error, incorporating lower-quality data leads to performance
degradation. Similarly, partial co-training variants that exclude
either dynamics or visual forecasting objectives (green and
brown lines) improve robustness but fail to fully exploit the
available data. In contrast, the full co-training framework (blue
line) exhibits consistent improvement as additional hetero-
geneous data is introduced. Notably, even after all action-
labeled trajectories are exhausted, adding 10k actionless videos
continues to reduce prediction error. These results indicate that
LDA can extract useful supervisory signals from low-quality
data and non-action data through latent dynamics and visual
forecasting, rather than treating such data as noise. Overall,
these results demonstrate that Universal Data Ingestion is most
effective when heterogeneous data and co-training objectives
are scaled together, enabling LDA to fully utilize mixed-
quality supervision.

Effectiveness of Latent Representation. Although both LDA
and UWM incorporate dynamics-related supervision, their
scaling behaviors diverge substantially due to differences in
the structure of their latent spaces. As shown in Fig. 10,
UWM quickly saturates as data scale and model capacity
increase, with additional supervision yielding diminishing or
even negative returns. This indicates that simply increasing
data or parameters is insufficient when the latent space cannot
support compositional and causal reasoning. This limitation
stems from UWM’s VAE-derived latent representation, which
entangles appearance, geometry, and dynamics at a low-level
feature granularity. Such entanglement restricts the models
ability to factorize action-induced state transitions and prevents
effective reuse of heterogeneous supervision during scaling.
In contrast, LDA operates in a semantically structured la-
tent space obtained from large-scale visual pretraining. This
representation preserves object-level semantics and spatial
coherence, enabling dynamics learning to scale smoothly with
increased model capacity, richer training objectives, and more
diverse datasets.

Effectiveness of Model Size Scaling. Beyond data scale, LDA
