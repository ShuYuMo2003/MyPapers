# Page 1
2026-2-19

EgoScale: Scaling Dexterous Manipulation with Diverse
Egocentric Human Data

Ruijie Zheng 1 * , Dantong Niu 1 , 2 * , Yuqi Xie 1 * , Jing Wang 1 , Mengda Xu 1 , Yunfan Jiang 1 , Fernando Castañeda 1 ,
Fengyuan Hu 1 , You Liang Tan 1 , Letian Fu 1 , 2 , Trevor Darrell 2 , Furong Huang 3 , Yuke Zhu 1 † , Danfei Xu 1 † , Linxi Fan 1 †

1 NVIDIA
2 University of California, Berkeley
3 University of Maryland
* Equal Contribution
† Project Lead

https://research.nvidia.com/labs/gear/egoscale/

Abstract

Human behavior is among the most scalable sources of data for learning physical intelligence, yet how
to effectively leverage it for dexterous manipulation remains unclear. While prior work demonstrates
human-to-robot transfer in constrained settings, it is unclear whether large-scale human data can
support fine-grained, high-degree-of-freedom dexterous manipulation. We present EgoScale, a human-
to-dexterous-manipulation transfer framework built on large-scale egocentric human data. We train
a Vision–Language–Action (VLA) model on over 20,854 hours of action-labeled egocentric human
video—more than 20× larger than prior efforts—and uncover a log-linear scaling law between human
data scale and validation loss. This validation loss strongly correlates with downstream real-robot
performance, establishing large-scale human data as a predictable supervision source. Beyond scale, we
introduce a simple two-stage transfer recipe: large-scale human pretraining followed by lightweight
aligned human–robot mid-training. This enables strong long-horizon dexterous manipulation and
one-shot task adaptation with minimal robot supervision. Our final policy improves average success
rate by 54% over a no-pretraining baseline using a 22-DoF dexterous robotic hand, and transfers
effectively to robots with lower-DoF hands, indicating that large-scale human motion provides a reusable,
embodiment-agnostic motor prior.

1. Introduction

Human behavior is one of the most scalable sources of data for learning physical intelligence. Humans routinely
perform dexterous manipulation across diverse objects, environments, and task variations at a scale that far
exceeds what can be collected through robot teleoperation. As robotic hardware continues to improve toward
more human-like kinematics and dexterity, a natural question arises: can human data serve as a primary training
signal for dexterous robot manipulation?

Recent work shows that transfer from human data to robots is possible by aligning observations or actions
across embodiments [ 12 , 42 , 25 , 24 , 30 ]. However, existing results remain limited in two respects. First, most
approaches rely on relatively small human datasets, typically on the order of tens to hundreds of hours. Second,
many focus on grippers or low DoF hands, where fine-grained finger articulation is absent. It therefore remains
unclear whether human data can meaningfully support complex, dexterous manipulation at scale.

In this work, we show that human-to-robot transfer for dexterous manipulation is fundamentally a scaling
phenomenon, and present EgoScale, a scalable human-to-dexterous-manipulation transfer framework built on
large-scale egocentric human data. We pretrain on 20,854 hours of egocentric human manipulation data ,
over 20 × larger than the dataset sizes used in prior studies of human–robot policy transfer, and uncover a
clear scaling law : human wrist and hand action prediction validation loss follows a log-linear relationship with
data volume. This enables us to extrapolate: as human data scales, validation loss continues to decrease, and
the learned representations generalize increasingly well. Crucially, the loss strongly correlates with real robot

© 2026 NVIDIA. All rights reserved.

arXiv:2602.16710v1  [cs.RO]  18 Feb 2026

# Page 2
EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data

20,854 Hours Human Videos

Pre-training

Efficient Learning on Highly Dexterous Tasks

Inject
Syringe

Tong
Fruit

One-shot on Unseen Dexterous Tasks

Unscrew

Bottle

Fold
Shirt

Post-training

Human-robot Aligned Play data
50 Hours Human + 4 Hours Robot

Mid-training

Figure 1: EgoScale: Two-stage human-to-robot learning framework. A flow-based Vision-Language-Action
(VLA) policy is first pretrained on 20,854 hours of egocentric human videos using wrist motion and retargeted
dexterous hand actions.
A lightweight mid-training stage with aligned human robot play data (pairs
highlighted with green and gray boundaries) adapts the representation to robot sensing and control. The
resulting policy is post-trained on downstream tasks, enabling efficient learning of dexterous manipulation
and one-shot generalization to unseen skills.

performance on long-horizon, complex manipulation tasks. Together, these results establish large-scale human
data as a scalable and predictable supervision source for learning dexterous manipulation policies.

Beyond scale, we identify a simple yet effective training recipe that enables new generalization capabilities. We
supervise the model using human manipulation behaviors represented as relative wrist motion and retargeted
high-DoF hand joint actions. This aligned action space encourages the model to extract information that
is directly useful for manipulation, rather than learning task-agnostic visual features. After pretraining, we
introduce a small amount of aligned human-robot mid-training data through co-training. The mid-training data
includes humans and robots performing similar manipulation tasks in matched tabletop scenes with comparable
visual viewpoints. This alignment provides supervisions for grounding the pretrained representations in the
robot’s sensing and control space.

Importantly, this mid-training stage gives rise to emergent one-shot and few-shot generalization . With only one or
a few robot demonstrations, the policy can adapt to new dexterous tasks without requiring extensive task-specific
data collection. For instance, using a single robot demonstration, the trained policy achieves up to 88% average
success on shirt folding, even though the mid-training data contains only folding behaviors. Moreover, although
human actions are supervised in a high-DoF dexterous hand space, the learned representations generalize to
substantially different robot embodiments . On the Unitree G1 robot with a tri-finger hand, human-pretrained
policies also achieve over 30% absolute improvement in success rate across both evaluated tasks compared to
baseline without human pretraining.

Taken together, our results show that effective dexterous human-to-robot transfer requires scale, explicit
motion supervision, and a small amount of precise human–robot alignment. Rather than replacing robot
data, large-scale human demonstrations dramatically amplify its effectiveness, pointing toward a future where
humans can be treated as another scalable embodiment in robot learning. We summarize the contributions as
follows:

2

# Page 3
EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data

• Human Data Pretraining Scaling Laws. At the scale of over 20k hours of human data, we uncover a
clear log-linear relationship between data scale and hand-action prediction loss and show that this loss
strongly correlates with real-robot dexterous manipulation performance.
• An Effective Human-to-Robot Transfer Recipe. We combine high-DoF human hand action supervision
with a small amount of aligned human–robot mid-training, enabling strong post-training performance
with minimal robot data.
• Emergent One-shot Transfer and Generalization. Our approach enables one-shot transfer to previously
unseen dexterous task with 22-DoF hands. The pretraining policy also generalizes effectively to robots
with lower-DoF hands, indicating that rich human motion provides a reusable motor prior.

2. Method

We aim to learn representations from large-scale egocentric human video that are directly useful for dexterous
robot control. This setting poses two core challenges. First, human demonstrations are noisy and lack paired
robot actions. Second, human and robot embodiments differ substantially in kinematics and control interfaces.
Our method (Figure 1 ) addresses these challenges through two design choices. We first pretrain on human data
using explicit supervision of wrist motion and hand articulation extracted from egocentric videos, forcing the
model to learn physically grounded action representations. We then introduce a small amount of aligned human-
robot data for mid-training, which grounds these representations in executable robot control without requiring
large-scale paired demonstrations. Together, this two-stage design decouples data scale from embodiment
alignment, enabling effective transfer from large human datasets to dexterous robot manipulation.

2.1. Human Action Representation

Raw Sensor Streams. Each human demonstration consists of egocentric RGB observations captured from a
head-mounted camera, together with estimated camera motion and human hand pose obtained from off-the-
shelf perception pipelines. We convert these raw sensory signals into a unified action representation suitable for
large-scale pretraining and downstream robot execution. Let ℱ 𝑤 denote the world frame and ℱ 𝑡
𝑐 the camera
frame at time 𝑡 . The estimated camera pose is represented as T 𝑡
𝑤 ← 𝑐 ∈ SE (3) . Human hand pose is modeled by
21 keypoints, each represented as a rigid transform H 𝑡
𝑐,𝑖 ∈ SE (3) in the camera frame, where 𝑖 = 1 corresponds
to the wrist. The wrist pose in the world frame is given by W 𝑡
𝑤 = T 𝑡
𝑤 ← 𝑐 H 𝑡
𝑐, 1 .

Wrist-level Arm Motion. To obtain motion commands that are invariant to global camera movement, we
represent arm motion using relative wrist motion between consecutive timesteps. Given timestep 𝑡 in an action
chunk, ∆ W 𝑡 = ( W 0
𝑤 ) − 1 W 𝑡
𝑤 . This relative end-effector formulation removes dependence on absolute camera
pose and captures local arm motion in a physically meaningful manner. The same representation is shared
across human demonstrations and robot executions, serving as the primary arm-level action abstraction for
cross-embodiment learning.

Hand Articulation. For finger-level control, we retarget the 21 human hand keypoints into a dexterous robot
hand joint space using an optimization-based procedure that enforces joint limits and kinematic constraints.
Our default choice is the 22-DoF hand action space of the Sharpa hand [ 29 ], which preserves human finger
articulation during pretraining while aligning with the control interface of our target robot. Although this
representation is defined using a high-DoF hand, we later show that the learned models transfer effectively to
robots with lower-DoF hands.

2.2. Human Data Sources and Processing

Stage I: Large-Scale Egocentric Human Pretraining Data. We pretrain our model on a large-scale mixture of
egocentric human activity datasets totaling 20,854 hours of video. The majority consists of in-the-wild egocen-
tric recordings spanning diverse real-world environments (e.g., household, industrial, retail, and educational
settings), covering 9,869 scenes, 6,015 tasks, and 43,237 objects, and providing broad coverage of long-tailed

3

# Page 4
EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data

manipulation behaviors.

Ego View

Left / Right Wrist View

(a) Egocentric Human Data Collection.

Pretrained

VLM

DiT
Action
Expert

“Iron the T-shirt”

Text
Encoder

Noisy
Joint
Angle
Action
Encoder

Visual
Encoder

Hand

Pose
extraction

Action
Decoder

Noisy

EEF
Pose

Joint
Angle

EEF
Pose

N iterations

Hand
retargeting

(b) EgoScale Model Architecture.

Figure 2: Human Data Collection and Model Architecture. ( Left ) Aligned human-robot mid-training data
are collected using the same sensing setup as the robot. Vive trackers and Manus gloves capture arm and
hand motion, while one head-mounted camera and two wrist-mounted cameras record egocentric and wrist
views, enabling consistent perception–action alignment. ( Right ) A flow-based VLA policy with a VLM backbone
and DiT action expert. Human and robot data are unified through a wrist-level action representation, with
lightweight embodiment-specific adapters for proprioception and hand actions.

All recordings are captured using egocentric RGB cameras at 30 FPS. We apply off-the-shelf SLAM and hand-pose
estimation pipelines to recover camera motion and human hand trajectories. Although these estimates are
noisy due to unconstrained data collection, the scale and diversity of the data provide effective supervision
for learning transferable action representations, which continue to improve downstream performance as data
volume increases.

To complement this large-scale but noisy supervision, we additionally incorporate 829 hours of EgoDex
dataset [ 8 ], collected using Apple Vision Pro with accurate wrist and hand tracking. EgoDex covers 194
tabletop manipulation tasks involving everyday objects and provides higher-precision kinematic signals that
help anchor pretraining while preserving scalability.

Stage II: Aligned Human-Robot Mid-Training Data. To further bridge the embodiment gap between human
demonstrations and robot execution, we introduce a smaller dataset with both human and teleoperated robot
data. We later show that this dataset is critical to anchor the pretrained representations to the robot’s sensing
and action spaces.

This dataset comprises 344 tabletop manipulation tasks, with each task captured in approximately 30 human
trajectories and 5 robot trajectories, totaling about 50 hours of human data and only 4 hours of robot data. As
shown in Figure 2a , human demonstrations are collected using the same camera configuration as the robot,
with matched viewpoints and calibrated intrinsics, ensuring that visual observations are directly comparable
across domains. Human hand motion is captured using the same motion-capture stack as in robot teleoperation:
Vive trackers provide wrist pose (3D position and orientation), while Manus gloves record full in-hand pose as
25 joint transforms. All motion signals are synchronized with the video stream.

Compared to the large-scale but unconstrained data used in Stage I, this dataset is significantly smaller but
explicitly embodiment-aligned. It focuses on tabletop tasks that match the robot’s workspace and kinematics,
enabling abstract human actions learned during pretraining to be grounded in executable robot control.
Together, Stage I and Stage II decouple scale and alignment: Stage I provides diversity and semantic grounding,
while Stage II supplies precise human-robot correspondence for downstream deployment.

4

# Page 5
EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data

2.3. Model Architecture

As shown in Figure 2b , our model follows a flow-based VLA architecture similar to GR00T N1 [ 19 ]. At each
timestep 𝑡 , the model conditions on an observation 𝑜 𝑡 = ( 𝐼 𝑡 , 𝑙 𝑡 ) consisting of an image and a language instruction,
which is encoded into a vision-language embedding 𝜑 𝑡 . The model then predicts a chunk of future actions
using a flow-matching objective.

For robot data, the model conditions on the robot proprioceptive state 𝑞 𝑡 , while human demonstrations
do not provide such signals. In the absence of proprioception, we replace 𝑞 𝑡 with a learnable placeholder
token, enabling a unified model formulation without architectural changes. To accommodate multiple robot
embodiments with different state and hand action spaces, following GR00T N1 [ 19 ], we use lightweight
embodiment-conditioned MLP adapters at the input and output interfaces. Specifically, these adapters encode
embodiment-specific proprioceptive state and decode hand actions, while relative wrist motion prediction, the
vision-language backbone, and the DiT action expert are fully shared. In practice, this mechanism is used only
for a small number of additional embodiments (e.g., G1 with a tri-finger hand).

2.4. Training Recipe

We use a three-stage training pipeline. In Stage I (human pretraining), we train on 20K hours of egocentric
human data for 100K steps with 256 GB200 GPUs using a global batch size of 8,192 and learning rate 5 × 10 − 5 ,
fully unfreezing every parameter of the VLA model to absorb large-scale data. Then in Stage II (aligned
mid-training), we train on the aligned human-robot play dataset for 50K steps with batch size 2,048 and
learning rate 3 × 10 − 5 , freezing the vision-language backbone while only updating the vision encoder and DiT
action expert to anchor representations to robot sensing and control. In Stage III (post-training), we fine-tune
on task-specific robot demonstrations for 10K steps with batch size 512 and learning rate 3 × 10 − 5 . During
post-training, the vision encoder is frozen if mid-training is used and unfrozen otherwise, to accommodate new
embodiments when needed.

2.5. Robot Systems and Control

The real world experiments are conducted on the Galaxea R1Pro humanoid robot with 22-DoF Sharpa dexterous
robot hands. We refer the readers to Appendix B for the system figure.

Dual-arm Wheeled Humanoid System Galaxea R1Pro. We fix the base and torso and focus on bimanual
manipulation, controlling both 7-DoF arms in relative end-effector space where actions specify incremental
position and orientation changes, matching the wrist-pose representation used in human demonstrations for
direct human-robot alignment.

22-DoF Dexterous Hands. We equip the robot with Sharpa Wave hands with 22 degrees of freedom and joint-
space control, where actions directly specify target joint angles, enabling precise articulation and preserving
the fine-grained structure of retargeted human hand motion.

Perception System. We use three RGB cameras: a head-mounted camera that provides an egocentric first-
person view consistent with human videos, and two wrist ones mounted on the inner side of each wrist facing
the palm, capturing close-range hand-object interactions and provide detailed visual feedback essential for
fine-grained dexterous manipulation.

3. Experiment

In this section, we aim to answer the following research questions through our experiments:
RQ1: Does large-scale egocentric human pretraining improve downstream dexterous manipulation performance
compared to training from scratch or embodiment-aligned data alone?
RQ2: How does the scale of human pretraining data affect representation quality and real-robot performance?
RQ3: What role does mid-training play in enabling few-shot adaptation and generalization to novel tasks?

5

# Page 6
EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data

RQ4: Do human-pretrained representations transfer across robot embodiments with substantially different kine-
matics and control interfaces?
RQ5: How does the choice of human action representation during pretraining affect downstream dexterous
manipulation?

3.1. Experiment Setup

Figure 3: Post-Training Evaluation Tasks. Five dexterous manipulation tasks used to evaluate post-training
performance

Tasks. To evaluate policy performance, we design five highly dexterous manipulation tasks shown in Figure 3 .

Each task is provided with 100 teleoperated robot demonstrations, except for Shirt Rolling , a deformable
manipulation task that requires less precise control, for which we use only 20 demonstrations.

(Task I) (Shirt) Shirt Rolling. The robot coordinates both hands to alternately fold and roll a T-shirt into a
cylindrical shape before placing it into a basket.

(Task II) (Card) Card Sorting. The robot uses its fingers to rub and separate a single card from a tightly stacked
deck, and then precisely inserts it into the correct holder based on color.

(Task III) (Tong) Dexterous Tool Use: Tongs for Fruit Transfer. The robot first grasps a pair of tongs from a
toolbox and then uses them to pick up a fruit and place it at a target location.

(Task IV) (Bottle) Unscrewing a Bottle Cap. The robot grasps and continuously rotates a small cap to remove it
from a bottle. We collect demonstrations on four bottles of different sizes, with 25 trajectories per bottle.

6

# Page 7
EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data

Shirt
Card
Tong
Bottle
Syringe
Average
0.0

0.2

0.4

0.6

0.8

1.0

Task Completion Score

0.40

0.14

0.35

0.18
0.16

0.24

0.55
0.54

0.85

0.51

0.23

0.53

0.83

0.74

0.79

0.63

0.53

0.71

0.90

0.87
0.87

0.82

0.70

0.83

(a) Task Completion Score

Shirt
Card
Tong
Bottle
Syringe
Average
0.0

0.2

0.4

0.6

0.8

1.0

Task Success Rate

0.05

0.00

0.05

0.00
0.00
0.02

0.20

0.50

0.60

0.10

0.00

0.28

0.40

0.70

0.45

0.21

0.17

0.38

0.50

0.65
0.65

0.59

0.42

0.56

(b) Task Success Rate

No Pretrain
Midtrain Only
Human Pretrain
Human Pretrain + Midtrain

Figure 4: Main Experimental Results. Comparison of Human Pre-train + Mid-Training, Human Pretraining,
and No Pretraining across five dexterous manipulation tasks under two evaluation metrics.

(Task V) (Syringe) Syringe Liquid Transfer. This is the most challenging task, requiring the robot to pick up
a syringe, draw liquid from tube A, inject it into tube B, and discard the syringe into a trash can. The task
involves long-horizon, multi-step reasoning, precise spatial alignment for fluid extraction and injection, and
dexterous manipulation of the syringe plunger.

Evaluation Metric. To evaluate policy performance, we train each method using two random training seeds.
Then, for each trained policy checkpoint, we evaluate performance over 10 trials, except for Task III, where we
conduct 4 trials per bottle across four bottle instances, resulting in 16 evaluation trials. To ensure consistency
across evaluation runs, we employ an image-overlay–based initialization procedure, in which the robot evaluator
is provided with a visual overlay of the target initial scene configuration to reduce variability in initial conditions.
For each task, we record both the absolute task success rate and fine-grained task completion score.

3.2. Large-Scale Human Pretraining Is Key to Strong Dexterous Manipulation Policy
Performance

To evaluate the impact of large-scale human pretraining and aligned mid-training on policy learning efficiency,
we compare four checkpoints: (1) a model trained from scratch, (2) a model pre-trained only on the midtrained
aligned human–robot play dataset, (3) a model pretrained on large-scale human data, and (4) a human-
pretrained model further mid-trained on aligned human–robot data. For each checkpoint, we report both the
task completion score and the absolute success rate.

Results are summarized in Figure 4 . Across all tasks, human pretraining consistently yields substantial
performance gains over training from scratch, improving average task completion by over 55%. Notably,
large-scale human pretraining, despite being noisy, unconstrained, and not task- or sensor-aligned, already
outperforms the mid-training-only baseline across most tasks. This provides the evidence that scale and diversity
of human demonstrations provide strong inductive biases for dexterous manipulation, even in the absence of
precise embodiment alignment.

Finally, combining human pretraining with a small amount of aligned mid-training yields the best overall
performance, suggesting a complementary effect: large-scale human data supplies general manipulation
structure, while mid-training anchors these representations to executable robot control.

3.3. Policy Performance Scales with Pretraining Data Size

We study how the scale of egocentric human pretraining data affects downstream real-robot manipulation
performance, and analyze how this behavior is reflected in offline human-action prediction metrics. We pretrain
models using 1k, 2k, 4k, 10k, and 20k hours of human data. To isolate the effect of stage II midtraining, we
directly post-train each checkpoint on the downstream task and evaluate their performance.

7

# Page 8
EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data

20
40
60
80
100
Training Steps (×1000)

0.014

0.016

0.018

0.020

0.022

0.024

0.026

0.028

0.030

Human Validation Loss (MSE)

Human Validation Loss vs Training Steps

1k
2k
4k
10k
20k
Human Data Amount (hours, log scale)

0.014

0.016

0.018

0.020

0.022

0.024

0.026

Human Validation Loss (MSE)

L = 0.024
0.003 ln( D )

Scaling Law: Optimal Loss vs Data Amount

1k
2k
4k
10k
20k
Human Data Amount (hours)

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8

Avg. Task Completion Score

0.30

0.45

0.48

0.57

0.71

Policy Performance vs Data Amount

1k hrs
2k hrs
4k hrs
10k hrs
20k hrs

Figure 5: Scaling behavior of human pretraining. Left: Human validation loss versus training steps for models
pretrained with increasing amounts of egocentric human data (1k–20k hours). Larger datasets yield stable,
monotonic improvements, while smaller datasets exhibit early overfitting. Center: Optimal validation loss at
convergence as a function of human data scale, revealing a near-perfect log-linear scaling law ( 𝑅 2 = 0 . 9983 ).
Right: Downstream robot performance after post-training, measured by average task completion score,
improves consistently with increased human data scale. Together, these results demonstrate predictable scaling
of learned action representations and their direct translation to improved dexterous manipulation performance.

As shown in Figure 5 (right), increasing the amount of human pretraining data leads to consistent and sub-
stantial gains in downstream robot performance. Average task completion rises monotonically from 0.30 at
1k hours to 0.71 at 20k hours, with no signs of saturation in the explored regime. These results indicate that
large-scale human data provides increasingly strong priors for dexterous manipulation, even though the human
demonstrations are noisy, unconstrained, and not task-aligned.

To better understand this trend, we examine how pretraining data scale affects the quality of learned action
representations. We evaluate each pretrained model on a held-out human-video validation set consisting of
2,000 egocentric episodes. For evaluation, we randomly sample 20 timesteps per trajectory; at each timestep,
we draw 16 samples from the flow-matching policy, average the predicted action chunks, and compute the
mean squared error against ground-truth wrist and hand actions.

Figure 5 (left) shows human validation loss as a function of training steps for different data scales. Models
trained with smaller datasets (1k–2k hours) initially reduce validation loss but later plateau or degrade,
indicating overfitting to limited behavioral diversity. In contrast, models trained with larger datasets (10k–20k
hours) exhibit stable, monotonic improvement throughout training without signs of overfitting.

Strikingly, when we plot the optimal validation loss achieved at convergence against data scale (Figure 5 ,
center), we observe a remarkably clean log-linear scaling law:

L = 0 . 024 − 0 . 003 · ln( D ) ,
(1)

where 𝐷 denotes the number of hours of human pretraining data. The fitted curve achieves an 𝑅 2 of 0.9983 ,
indicating an almost perfect linear relationship in log space. Crucially, this offline scaling behavior is strongly
predictive of real-robot performance. Human validation loss closely tracks downstream task completion across
data scales, establishing it as a meaningful indicator of embodied control capability rather than a purely offline
metric.

Together, these results show that effective human-to-robot transfer for dexterous manipulation is fundamentally
a scaling phenomenon. Within the explored regime, increasing human data yields predictable reductions in
validation loss and corresponding improvements in robot performance, with no evidence of diminishing returns.

8
