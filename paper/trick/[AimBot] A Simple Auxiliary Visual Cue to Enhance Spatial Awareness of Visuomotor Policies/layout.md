# Page 1
AimBot : A Simple Auxiliary Visual Cue to
Enhance Spatial Awareness of Visuomotor Policies

Yinpei Dai †∗ , Jayjun Lee ‡∗ , Yichi Zhang † , Ziqiao Ma † , Jianing Yang †

Amir Zadeh ♢ , Chuan Li ♢ , Nima Fazeli †‡ ⋆ , Joyce Chai † ⋆

† Computer Science and Engineering Department, University of Michigan
‡ Robotics Department, University of Michigan
♢ Lambda Labs
∗ Equal Contribution ⋆ Equal Advising
{ daiyp, jayjun, nfz, chaijy } @umich.edu

Abstract: In this paper, we propose AimBot , a lightweight visual augmentation
technique that provides explicit spatial cues to improve visuomotor policy learning
in robotic manipulation. AimBot overlays shooting lines and scope reticles onto
multi-view RGB images, offering auxiliary visual guidance that encodes the end-
effector’s state. The overlays are computed from depth images, camera extrinsics,
and the current end-effector pose, explicitly conveying spatial relationships be-
tween the gripper and objects in the scene. AimBot incurs minimal computational
overhead (less than 1 ms) and requires no changes to model architectures, as it
simply replaces original RGB images with augmented counterparts. Despite its
simplicity, our results show that AimBot consistently improves the performance
of various visuomotor policies in both simulation and real-world settings, high-
lighting the benefits of spatially grounded visual feedback. Codes and videos can
be found at https://aimbot-reticle.github.io/

Keywords: Robotic Manipulation, Visuomotor Policy, Imitation Learning

① Get the starting
point from the EE pose

③ Find the stopping
point (invisible from

observed depth)

② Calculate
middle points

based on the

gripper
orientation

Add
Shooting

Line

Visuomotor Policy

Multiview
RGB Images

Task
Goal

Robot

State

Closed-Loop Robot Control

Action
Chunk

∆joint
∆Grip = ⋯
<1ms

Global-view Image

Local-view Image

Add
Reticle

AimBot

Figure 1: Overview of AimBot , a lightweight visual guidance method that adds spatial cues onto
RGB images for visuomotor policy learning. Given the robot’s end-effector pose, camera extrinsic
and depth image, AimBot computes shooting lines and reticle overlay to highlight the spatial rela-
tionship between the gripper and objects of interest. The augmented RGB images are used as input
to visuomotor policies for closed-loop control, enhancing task performance with minimal overhead.

1
Introduction

Robotic manipulation in unstructured environments demands visuomotor policies that can robustly
and accurately predict continuous actions from raw RGB observations. Although recent advances,
such as diffusion-based policies [ 1 , 2 ] and vision-language-action (VLA) models [ 3 , 4 , 5 , 6 , 7 ], have
leveraged large-scale datasets to learn complex behaviors, they often lack explicit spatial grounding
in visual input. As a result, these policies often exhibit limited spatial awareness of their end-effector
(EE) pose and its relationship to surrounding objects [ 8 , 9 , 10 , 11 , 12 , 13 ].

arXiv:2508.08113v1  [cs.RO]  11 Aug 2025

# Page 2
Motivated by the intuitive visual feedback offered by scope reticles in optical sighting systems [ 14 ,
15 , 16 ], we present AimBot , a lightweight and effective visual augmentation technique that enhances
spatial awareness in robotic manipulation with visual targeting cues. AimBot overlays auxiliary
scope reticles and shooting lines onto multi-view RGB images by leveraging depth information,
camera extrinsics, and the robot’s EE pose. These spatial cues project the intended grasp point
and the orientation of the EE onto the image plane, offering an interpretable visualization of the
alignment between the gripper and objects of interest (see Figure 1 ). Beyond highlighting spatial
relationships, these augmentations also intuitively encode the robot’s proprioceptive state, making
the EE position and orientation directly accessible in the visual domain.

AimBot requires no changes to model architecture and introduces negligible computational over-
head (less than 1 ms), while substantially enriching the spatial information available to any visuo-
motor policy. Through extensive experiments in both simulation and real-world environments, we
demonstrate that AimBot consistently improves overall task performance across diverse VLA back-
bones, with particularly strong gains on challenging, long-horizon tasks that demand effective spa-
tial alignment between the EE and objects. We believe this simple yet effective approach provides a
practical and scalable means to enhance learning-based robotic manipulation in 3D environments.

2
Related Works

2.1
Vision-Language-Action Models for Manipulation

Recent advances in pre-trained foundation models [ 17 , 18 , 19 , 20 ] have catalyzed the development of
vision-language-action (VLA) models, which significantly boost visuomotor policy learning in gen-
eralization, scalability, and robustness [ 3 , 4 , 21 ]. OpenVLA [ 4 ] and OpenVLA-OFT [ 22 ] extend the
Llama model [ 17 ] to large-scale real-world robot data, achieving strong performance and effective
action generation. π 0 [ 3 ], π 0 -FAST [ 5 ] leverage the Gemma model [ 23 ] and compression techniques
to develop generalist policies capable of handling complex manipulation tasks. GR00T [ 24 ] utilize a
heterogeneous mixture of real-robot trajectories, human videos, and synthetic datasets for expressive
humanoid control. FuSe [ 25 ] moves beyond data scaling by leveraging multiple sensory modalities,
demonstrating that natural language can universally ground vision, touch, and sound without requir-
ing extensive multi-modal datasets. GO-1 [ 26 ] improves long-horizon reasoning through a VQ-VAE
latent action model trained on web-scale video data. All these work highlight the rapid progress in
robotic manipulation, providing a foundation upon which our work can build.

2.2
Visual Augmentation/Guidance for Manipulation

With the success of visual prompting in VLMs [ 27 , 28 , 29 , 30 , 31 , 32 ], recent approaches have ex-
plored visual intermediaries to enhance generalization in robotic manipulation. RT-Affordance [ 33 ]
predicts affordance plans (i.e., key robot poses) conditioned on the visual input, enabling flexible
learning across diverse supervision sources. RT-Trajectory [ 34 ] extends this idea by conditioning
policies on coarse trajectory sketches, facilitating generalization to novel scenarios. GENIMA [ 35 ]
fine-tunes a diffusion model to overlay joint-action targets onto RGB images, which are then trans-
lated into joint positions. TraceVLA [ 9 ] introduces visual trace prompting that encodes spatiotem-
poral trajectories directly into visual inputs, enabling VLA models to better predict actions. Robo-
Point [ 36 ] leverages a VLM to predict keypoint affordances in terms of points on RGB images.
HAMSTER [ 37 ] introduces a hierarchical approach that separates high-level task planning from
low-level motor control using intermediate 2D path prediction. However, all those methods require
online model inference during deployment, significantly limiting their practicality for real-time con-
trol. In contrast, our work presents a lightweight visual augmentation technique that overlays inter-
pretable 2.5D spatial cues onto RGB images, offering spatial information without incurring infer-
ence costs. Moreover, by embedding EE state information directly into the pixel space, our method
provides a novel and effective EE representation to enhance vision-language-action models.

2

# Page 3
3
Methodology

3.1
Method Overview

Visuomotor policies [ 1 , 38 ] aim to directly map RGB camera observations and proprioceptive states
to robot actions for sensorimotor control. Our goal is to enhance these policies with auxiliary visual
cues, such as shooting lines and crosshair reticles, to improve spatial alignment and task success.
The proposed technique, AimBot , is model-agnostic and requires no modifications to underlying
policy architectures. It operates by augmenting the original multi-view RGB images with visual
guidance, embedding spatial information directly into the pixel space. Fine-tuning is then performed
exclusively on these new images, enabling any visuomotor policy or vision-language-action model
to leverage the enhanced spatial cues.

3.2
AimBot Visual Guidance

Suppose we have the camera extrinsics E ∈ R 4 × 4 and intrinsics K ∈ R 3 × 3 , along with the RGB
image I and depth image D captured at the current timestep from camera c . Given a 3D point
p wld ∈ R 3 in the world frame, we can project it into camera frame with pinhole camera model [ 39 ]:

p cam
1


= E ·

p wld
1


,
where p cam = ( x c , y c , z c ) ⊤ .
(1)

Then, we can continue to project p cam to 2D image coordinates ( u c , v c ) using the intrinsic matrix:
" u c
v c
1

#

∝ K

" x c /z c
y c /z c
1

#

.
(2)

We define a point as visible if the projected pixel lies within the image bounds and the projected
depth is smaller than the observed depth (i.e, not being blocked by any objects). Formally, let
( u c , v c ) be the projected pixel location and z c the projected depth. The point is visible if:

0 ≤ u c < W,
0 ≤ v c < H,
and
z c + ϵ < D [ v c , u c ]
(3)

where H and W are the image height and width, respectively, D [ v c , u c ] is the observed depth at
pixel ( u c , v c ) , and ϵ > 0 is a small threshold. Next, we describe how to determine the starting and
stopping point of a forward line based on visible points and how to add AimBot visual guidance to
RGB images. The detailed algorithm can be found in Appendix A.1 .

Starting Point.
We always set the origin of the gripper frame attached to the end-effector (EE) as
the starting point, which is denoted as p ee
wld in the world frame. The pixel location of the starting
point in the image is denoted as ( u ee
c , v ee
c ) for camera c . The camera can be either fixed or movable.

Stopping Point.
Starting from p ee
wld , we iteratively move forward along the direction vector d ∈
R 3 , derived from the EE’s orientation (e.g., the z -axis of the gripper frame):

p ( i +1)
wld
= p ( i )
wld + δ · d ,
with p (0)
wld = p ee
wld .

where δ > 0 is a small value denoting step length. At each step, we project p ( i )
wld to image coordinates
and check visibility using the same procedure as above. The iteration stops when a certain tolerance
number of invisible p ( i )
wld or the maximum step is reached. Then we choose the last p ( L )
wld as our
stopping point, where L is the total iteration number. We denote the pixel location of the stopping
point in the camera image as ( u sp
c , v sp
c ) . The total projection distance from p ee
wld to p ( L )
wld is δL | d | .

Augment Global-View with Shooting Lines.
In robotic manipulation, global-view observations
are typically captured from static external cameras, such as front-facing or shoulder-mounted cam-
eras. For these views, we overlay a shooting line on the image, extending from the pixel location
corresponding to the starting point ( u ee
c , v ee
c ) to the projected stopping point ( u sp
c , v sp
c ) . This line
serves as an explicit visual indicator of the EE’s position and orientation. In our default implemen-
tation, we also use color to convey gripper state: a green line with a red starting point indicates that
the gripper is open, while a purple line with a blue starting point indicates closed gripper.

3

# Page 4
Figure 2: An example of AimBot -augmented LIBERO observations. We add a shooting line and a
crosshair reticle for the front-view (top row) and wrist-view (bottom row) cameras, respectively.

Augment Local-View with Reticles.
Wrist-mounted cameras are commonly employed to capture
dynamic, close-range egocentric observations that provide detailed local views of the scene. To
convey the EE’s state, we overlay scope reticles onto the wrist-view image. Specifically, we render
a crosshair-style reticle centered at the projected stopping point, indicating the direction in which
the gripper is pointing. Note that, the pixel location ( u sp
wrist , v sp
wrist ) varies depending on the projection
distance δL | d | , which reflects the distance from the EE to the nearest surface. When the projection
distance is large (e.g., the gripper is far from the table), ( u sp
wrist , v sp
wrist ) appears closer to the center of
the wrist image due to perspective effects. Conversely, when the projection distance is small (e.g.,
the gripper is close to an object that obstructs the projection path), ( u sp
wrist , v sp
wrist ) aligns more closely
with the center of the gripper pads in the image.

To encode additional information about spatial proximity, the length of the reticle lines is also mod-
ulated based on the projection distance: the reticle lines are shorter when the distance is large and
longer when the distance is small, providing a visual indication of the estimated distance to the near-
est orthogonal surface (see Appendix A.1 for details). This augmentation provides a visual cue that
helps the policy develop a good understanding of spatial depth and distance in the local environ-
ment. In our default implementation, the crosshair lines are rendered in green , and the center point
is colored red or blue to indicate whether the gripper is currently open or closed, respectively.

4
Experiments

We choose three latest vision-language-action models, π 0 [ 3 ], π 0 -FAST [ 5 ] and OpenVLA-OFT [ 22 ],
as our visuomotor policy backbones, and conduct experiments on both simulated and real environ-
ments to evaluate our approach. For the simulation study, we choose the LIBERO [ 40 ] benchmark
as the testbed. For the real-world study, we design five challenging tasks to test our approach.

4.1
Simulation Experiments

Simulation Setup:
The LIBERO benchmark consists of four task suites (LIBERO-Spatial,
LIBERO-Object, LIBERO-Goal, and LIBERO-Long) designed for studying lifelong learning in
robotic manipulation. LIBERO-Spatial varies scene layouts with the same objects to test spatial
reasoning, LIBERO-Object varies objects within fixed layouts to test object understanding, and
LIBERO-Goal varies task goals while keeping objects and layouts fixed to test task-oriented behav-
ior. LIBERO-Long includes long-horizon tasks with diverse objects, layouts, and goals, and is the
most challenging suite as it demands more accurate gripper-object alignment over extended trajec-
tories. To generate training data, we regenerate all the demonstrations following [ 4 ], and augment
both multi-view RGB images with AimBot at a resolution of 256 × 256 pixels (shown in Figure 2 ).

Implementation Details: Following [ 3 ], we train a single multi-task policy for all tasks rather than
different policies for each task. All models are optimized using AdamW [ 41 ] for 30k steps with a
batch size of 32. For π 0 and π 0 -FAST, we adopt a learning rate of 2e-4 with 1k warm-up steps and
cosine decay. For OpenVLA-OFT, we use a fixed learning rate of 5e-4 and an L1 regression loss
for action training, along with LoRA [ 42 ] optimization (rank 32, alpha 16). The action horizon is
set to 50 steps for π 0 , and 10 steps for the other two, following their default settings. Observations

4

# Page 5
Model
LIBERO
Spatial
LIBERO
Object
LIBERO
Goal
LIBERO
Long
A VERAGE
S UCCESS R ATE

OpenVLA-OFT [ 22 ]
96.2
97.3
93.9
87.5
93.8
OpenVLA-OFT + AimBot
95.2 (–1.0)
99.1 (+1.8)
94.2 (+0.3)
91.2 (+3.7)
95.0 (+1.2)

π 0 -FAST [ 5 ]
96.5
96.8
93.6
81.6
92.1
π 0 -FAST + AimBot
96.9 (+0.4)
96.8 (+0.0)
94.0 (+0.4)
87.1 (+5.5)
93.7 (+1.6)

π 0 [ 3 ]
96.8
98.8
95.8
85.2
94.2
π 0 + AimBot
96.9 (+0.1)
98.4 (–0.4)
97.2 (+1.4)
91.0 (+5.8)
95.9 (+1.7)

Table 1: Performance comparison on the LIBERO simulation benchmark. Green and Red numbers
indicate performance gains and losses, respectively. Each task suite averages over four runs.

Model
Fruits
in Box
Tennis Ball
in Drawer
Bread in
Toaster
Place
Coffee Cup
Egg in
Carton
T OTAL
S UCCESS
OpenVLA-OFT
7/10
6/10
4/10
2/10
2/10
21/50
OpenVLA-OFT + AimBot
9/10
7/10
9/10
8/10
3/10
36/50

π 0 -FAST
10/10
10/10
9/10
7/10
6/10
42/50
π 0 -FAST + AimBot
10/10
10/10
10/10
9/10
8/10
47/50

π 0
7/10
7/10
4/10
5/10
4/10
27/50
π 0 + AimBot
10 / 10
10/10
7/10
8/10
8/10
43/50
Other baseline settings
π 0 + Traces [ 9 ]
8/10
8/10
5/10
2/10
2/10
25/50
π 0 + RoboPoint [ 36 ]
8/10
9/10
4/10
6/10
0/10
27/50
π 0 + Depth Images
7/10
9/10
5/10
7/10
4/10
32/50

Table 2: Performance comparison on five real-world tasks. Each task is evaluated over ten trials.

consist of front-view and wrist-view RGB images, along with proprioceptive states provided by the
simulator. All models operate in a delta Cartesian action space (6 dimensions) comprising changes
in EE position and EE orientation represented in Euler angles. An additional dimension is used to
represent the gripper open/close action. During evaluation, we execute 5 steps per action prediction
for π 0 and π 0 -FAST, and 8 steps for OpenVLA-OFT, enabling closed-loop control.

Experimental Results.
For baselines trained without AimBot , we re-implement all models on the
original LIBERO benchmarks and report the higher results between our re-implementations and their
released numbers. The overall results are summarized in Table 1 . Across all three backbone models,
integrating and finetuning with AimBot consistently improves or matches baseline performance in
the majority of cases, with particularly large gains on the hardest LIBERO-Long tasks. For instance,
π 0 -FAST improves from 81.6 to 87.1 and π 0 improves from 85.2 to 91.0 with AimBot . OpenVLA-
OFT also achieves a notable improvement from 87.5 to 91.2 on LIBERO-Long. This indicates that
the spatially grounded visual guidance provided by AimBot can enhance long-horizon manipulation.
For other task suites, where baseline performances are already relatively high, AimBot leads to
smaller improvements. Nevertheless, the average success rates still increase: +1.2 for OpenVLA-
OFT, +1.6 for π 0 -FAST, and +1.7 for π 0 . Overall, AimBot provides the greatest benefits in more
challenging, long-horizon settings while offering modest gains in easier tasks. To further study the
impact of reticle design choices, we compare several AimBot variants in Appendix A.2 and adopt
the best-performing setting (same setting as shown in Figure 2 ) for real-world experiments.

4.2
Real World Experiments

Real World Setup.
We conduct our experiments using a 7-DoF Franka Emika Panda robot
equipped with a pair of UMI fin-ray fingers [ 43 ] mounted on the default Franka hand gripper. The
robot operates in a tabletop environment with three RGB-D cameras similar to the DROID setting
[ 2 ]: two Intel RealSense D435 cameras each positioned on the left and right shoulder, and one Intel
RealSense D405 camera mounted on the wrist. The shoulder cameras are calibrated prior to exper-
imentation to obtain accurate camera extrinsics, while the extrinsics of the wrist-mounted camera

5

# Page 6
Figure 3: We design five contact-rich, long-horizon real-world tasks for policy evaluation.

are dynamically computed based on the end-effector (EE) pose. For the depth images, we use the
built-in Intel RealSense SDK to preprocess and align the depth images to the RGB images.

Task Design and Data Collection. We designed five challenging tasks for the real world experi-
ment: (1) Fruits in Box , where the robot is tasked to move all fruits inside a basket into a box;
(2) Tennis Ball in Drawer , where the robot needs to open a drawer, place a tennis ball inside,
and close the drawer; (3) Bread in Toaster , where the robot has to grasp a bread and insert into
a narrow toaster slot; (4) Place Coffee Cup , where the robot must grab a coffee cup by its han-
dle, reorient it, and place it onto a coffee machine; and (5) Egg in Carton , where the robot needs
to pick up one or more eggs, place them into an egg carton, and firmly close the carton lid. These
tasks vary in length, complexity, and contact-richness, and require a combination of prehensile (e.g.,
pick-and-place) and non-prehensile (e.g., pulling a drawer handle, closing a carton lid) skills. Fig-
ure 3 illustrates the table settings, and Appendix A.3 details the policy rollout as sequences of RGB
images augmented by AimBot for each task. For the real-world data collection, we teleoperate
the robot using an Oculus Quest 2 device, recording demonstrations at 15 Hz. We collect 80–150
demonstrations per task, resulting in a total of 548 episodes with 166k timesteps of data samples.

Implementation Details.
Following the simulation study, we fine-tune π 0 , π 0 -FAST, and
OpenVLA-OFT to evaluate the effectiveness of AimBot in real-world settings. We compare models
trained on raw RGB images from multiple camera streams with models trained on RGB images
augmented with AimBot . All models are optimized for 50k steps with a batch size of 32. We use
a learning rate of 1e-4 for π 0 and π 0 -FAST, which we found to perform better, and set the action
horizon to 10 steps for all models. Other hyperparameters are kept consistent with the simulation
experiments. Observations include left-shoulder, right-shoulder, and wrist RGB images, along with
the robot’s proprioceptive state. In terms of action space, π 0 and π 0 -FAST predict delta joint angles
(7 dimensions), while OpenVLA-OFT predicts delta Cartesian actions (6 dimensions); all models
additionally output a binary gripper open/close command as an extra dimension. During online ex-
ecution, we found that executing 8 steps per action prediction for π 0 and π 0 -FAST, and 2 steps for
OpenVLA-OFT, yields better performance.

Baselines. To compare AimBot with other visual guidance methods, we evaluate two open-sourced
baselines: RoboPoint [ 36 ] and TraceVLA [ 9 ], by training π 0 on images augmented using their re-
spective strategies. RoboPoint predicts spatial affordances as 2D pixel coordinates on semantically
relevant regions of the image, while TraceVLA visualizes temporal motion history through colored
arrows (i.e., ”traces”), offering cues about past trajectories. For RoboPoint, we query the robopoint-
v1-vicuna-v1.5-13b checkpoint, using the prompt “The task is <task goal> . Find relevant points
on the image to perform the task.” , where <task goal> corresponds to the language description
of each real-world task. We apply their default visualization tool to overlay red crosses at the pre-
dicted affordance points onto the RGB images as the visual augmentation for policy training. For
TraceVLA, we apply their co-tracker model to track objects across three camera views and overlay
the resulting traces onto the images. Example comparisons of the visual augmentations are shown in
Figure 4 . In addition, we compare against π 0 trained with both raw RGB images and depth images
to assess the effect of directly incorporating depth information. Concretely, we convert the depth
image in grayscale RGB and treat it as three more images (a total of six images) to train π 0 .

Experimental Results. Table 2 reports task success numbers (out of 10 trials per task) across five
real-world manipulation tasks using different policy and visual guidance configurations. Overall,
AimBot significantly improves performance across all models and tasks. For example, OpenVLA-

6

# Page 7
RoboPoint
TraceVLA
AimBot (Ours)

Figure 4: Comparison of different visual guid-
ance methods.

Without AimBot
With AimBot
Original RGB

Figure 5:
Visualization of attention weights
trained with and without AimBot .

OFT’s total success increases from 21 to 36 successful trials when augmented with AimBot , with
significant gains in more challenging tasks such as Place Coffee Cup and Bread in Toaster .
Similarly, π 0 improves from 27 to 43 successful trials, while π 0 -FAST achieves the highest overall
performance with 47 successful trials when combined with AimBot , outperforming all other models.

In contrast, alternative visual guidance baselines underperform: π 0 trained with RoboPoint and
TraceVLA reaches only 27 and 25 successful trials, respectively, similar to the π 0 trained with-
out any guidance. Unlike these methods, AimBot provides direct spatial targeting by encoding
the end-effector (EE) state as shooting lines and reticles, offering richer and clearer spatial cues
without occluding objects or omitting gripper state information. Additionally, both TraceVLA and
RoboPoint require online model inference, introducing significant computational overhead that lim-
its their practicality for real-time robot control. On average, TraceVLA requires approximately 0.3
seconds to process a single image, while RoboPoint takes over 5 seconds. In contrast, AimBot is
highly efficient, requiring less than 1 ms per image, making it suitable for real-world deployment.

While using raw depth images as additional visual input yields a modest improvement over RGB-
only inputs (increasing success from 27 to 32), it still falls short of AimBot ’s performance (43),
likely due to noise and inconsistency in real-world depth sensing. Despite extensive preprocessing,
real-world depth data remains highly noisy and unreliable. In contrast, AimBot is inherently robust
and less redundant, as it only compares the projected point depth with the camera depth to determine
visibility for visual cue generation, without relying on complete or very accurate depth information.

4.3
Further Analysis

To better understand how AimBot ’s visual guidance influences the model’s internal representations,
spatial reasoning, and generalization to unseen scenarios, we investigate the following questions:

Does AimBot enhance the robot’s spatial awareness for object alignment?
To address this, we
first examine how training with AimBot shapes the model’s attention toward task-relevant objects.
Specifically, we feed the same input image into the OpenVLA-OFT language model backbone and
extract the attention weights from Layer 1, Head 11. We then compute the summed attention scores
from the action head to the input RGB image patches and upsample them to the original image
resolution to generate the heatmaps shown in Figure 5 . As illustrated, models trained with AimBot
exhibit more concentrated attention on the task-relevant objects, which facilitates better task under-
standing and execution. In contrast, the baseline model without AimBot shows dispersed attention
across the entire scene, making it harder for the robot to align accurately with target objects. More
visualization examples can be found in Appendix A.4 .

To further investigate AimBot ’s impact on spatial alignment, we conduct a failure and error analy-
sis based on real-world robot experiments. We categorize all failure trials into different misalign-
ment types: grasping position misalignment, grasping orientation misalignment, placing position
misalignment, placing orientation misalignment, and other non-misalignment failures (e.g., getting
stuck, failed non-prehensile skills, or performing actions in a wrong task order). Examples of fail-

7

# Page 8
Misalignment Type
w/o AimBot
w/ AimBot

Grasping Position
22
7
Grasping Orientation
6
0
Placing Position
18
7
Placing Orientation
3
3
Other Failures
11
7

Table 3: Total failure counts across different
misalignment types with and without AimBot .

Models
LIEBRO-Long

π 0 + AimBot
91.0
π 0 + AimBot − proprio.
88.0
π 0
85.2
π 0 − proprio.
83.2
π 0 + AimBot (random)
77.4

Table 4: Ablation of π 0 variants with or with-
out AimBot and proprioceptive states.

ures are illustrated in Appendix A.5 . The aggregated results for three VLA models are summa-
rized in Table 3 . As shown, models trained with AimBot demonstrate a substantial reduction in
misalignment-related failures, particularly in grasping. This indicates that AimBot effectively en-
hances the model’s spatial understanding and improves alignment capability during manipulation.

Does AimBot provide a better representation than proprioceptive state encoding?
To investi-
gate this question, we conduct an ablation study using π 0 on the LIBERO benchmark. The original
π 0 setup includes a proprioceptive state vector encoding the end-effector state. We introduce two ab-
lated variants: (1) π 0 + AimBot − proprio., where AimBot is incorporated while the proprioceptive
input is replaced with a zero vector; and (2) π 0 − proprio., where neither AimBot nor proprioceptive
input is used, with the proprioceptive state similarly zeroed out. Table 4 reports the performance on
the LIBERO-Long task suite (more results are given in Appendix A.6 ). As shown, removing propri-
oceptive input reduces task success to 83.2%, compared to 85.2% when using proprioception alone.
Incorporating AimBot without proprioception improves performance to 88.0%, and combining both
leads to the highest task success of 91.0%, suggesting that AimBot provides a strong alternative
representation and offers complementary benefits to standard proprioceptive state encoding.

To further validate that AimBot effectively encodes useful spatial priors into the visual input, we in-
troduce an additional variant, π 0 + AimBot (random), where the auxiliary visual cues (shooting lines
and reticles) are deliberately randomized by adding noise and misaligned from the true end-effector
pose during both training and testing. As shown in Table 4 , this model achieves a significantly
lower success rate of 77.4%, compared to 91.0% with correctly aligned AimBot . This performance
gap indicates that the spatial visual guidance provided by AimBot is not only interpretable but also
meaningfully enhances the model’s spatial awareness and downstream policy performance.

Can AimBot improve generalization to out-of-distribution (OOD) scenarios?
We evaluate the
ability of AimBot to enhance OOD generalization in real-world tasks by introducing several dis-
tribution shifts at test time, including changes in object and receptacle heights, variations in table
background colors, and lighting conditions that differ significantly from the training environment
(examples can be found in Appendix A.7 ). For each task, we conduct 3 trials, resulting in a total
of 15 evaluation episodes using the π 0 -FAST model. In this experiment, our method achieves 12
successful trials, whereas the baseline model achieves only 7. This suggests that AimBot provides
robust visual cues that remain effective under distribution shifts, allowing the model to leverage reli-
able spatial guidance even in unseen environments and thus improving generalization performance.

5
Conclusion and Discussions

We present AimBot , a lightweight visual augmentation technique that enhances visuomotor pol-
icy learning by embedding spatial cues, such as end-effector position, orientation, and grasp state,
into RGB observations through scope reticles and shooting lines. This augmentation improves task
performance by depicting spatial context directly in pixel space, offering grounded 2.5D visual guid-
ance without requiring any changes to the model architecture. Experiments in both simulated and
real-world environments demonstrate the effectiveness and generalizability of AimBot across vari-
ous vision-language-action backbones. In the future, we plan to enrich AimBot with more semantic
cues, such as object segmentation and affordance labels, and extend its applicability to diverse em-
bodiments, including bi-manual and hand robots, to address more dexterous manipulation tasks.

8
