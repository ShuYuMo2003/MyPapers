# Page 1
arXiv:2506.15666v1  [cs.RO]  18 Jun 2025

Vision in Action: Learning Active Perception from
Human Demonstrations

Haoyu Xiong
Xiaomeng Xu
Jimmy Wu
Yifan Hou
Jeannette Bohg
Shuran Song

Stanford University

https://vision-in-action.github.io

Abstract: We present Vision in Action (ViA), an active perception system for
bimanual robot manipulation. ViA learns task-relevant active perceptual strate-
gies (e.g., searching, tracking, and focusing) directly from human demonstrations.
On the hardware side, ViA employs a simple yet effective 6-DoF robotic neck to
enable flexible, human-like head movements. To capture human active percep-
tion strategies, we design a VR-based teleoperation interface that creates a shared
observation space between the robot and the human operator. To mitigate VR mo-
tion sickness caused by latency in the robot’s physical movements, the interface
uses an intermediate 3D scene representation, enabling real-time view rendering
on the operator side while asynchronously updating the scene with the robot’s lat-
est observations. Together, these design elements enable the learning of robust
visuomotor policies for three complex, multi-stage bimanual manipulation tasks
involving visual occlusions, significantly outperforming baseline systems.

Keywords:
Active Perception, Bimanual Manipulation, Imitation Learning,
Teleoperation Systems

1
Introduction

Target Visible
Target Not Visible

Target Not Visible

Target Visible

Figure 1: Vision in Action (ViA) uses an active head
camera to search for the target object (yellow banana)
inside the bag. The wrist cameras are ineffective in this
visually occluded scenario, as they are constrained by
the arm motions.

Perception is inherently active [ 1 ]. Consider the
task of retrieving a banana from a bag (Fig. 1 ):
one must first scan the environment to locate
the bag, then peek inside to identify the banana,
and finally focus on the object to determine an
appropriate grasp. These deliberate viewpoint
changes serve to increase visual coverage dur-
ing the search, reduce occlusions caused by ob-
stacles ( e.g ., the bag), and focus attention on
action-critical regions ( e.g ., for grasp finding).

Yet, most robotic imitation learning systems [ 2 ,
3 , 4 , 5 , 6 , 7 , 8 , 9 ] do not incorporate active per-
ception. These systems typically rely on wrist
cameras [ 6 , 7 , 8 ] or fixed third-person cam-
eras [ 10 ]. Since wrist cameras move with the
arm, their viewpoints are constrained by ma-
nipulation requirements rather than guided by
perceptual objectives. This limitation becomes especially problematic in scenarios involving visual
occlusion, where wrist cameras are often blocked by the environment and fail to capture task-relevant
information necessary for accurate action inference. Furthermore, during data collection, humans
naturally shift their gaze to guide attention. However, the robot usually perceives the scene from
fixed or mismatched viewpoints. As a result, these systems fail to capture rich human perceptual
behaviors such as searching, tracking, and focusing. This fundamental observation mismatch —

For any questions, please contact: haoyux.me@gmail.com

# Page 2
between what the human sees and what the robot learns from —introduces a critical gap that ulti-
mately hinders the learning of effective policies. Despite its importance, active perception is often
neglected in today’s robotic systems due to the significant system-level challenges it introduces,
including:

• Flexible hardware for human-like gaze control. While humans effortlessly coordinate eye, neck,
and torso movements to direct their gaze in a variety of ways, replicating this capability in robots
is difficult. Most robot systems today rely on fixed or constrained cameras ( e.g ., 2-DoF necks [ 11 ,
12 , 13 , 14 ]), which limit the ability to adjust viewpoints flexibly.
• Synchronized camera-gaze movements. Virtual reality provides a powerful interface for teleoper-
ating robots and capturing human active perception [ 14 , 15 ]. However, designing an interface that
synchronizes human gaze and movement of the robot camera requires precise mirroring of human
motions and real-time streaming of visual feedback. Achieving this demands fast motor control
and low-latency data streaming, both of which remain challenging with today’s hardware.
• Scalable active perception strategies. Human gaze is driven by top-down and bottom-up atten-
tion [ 16 , 17 , 18 , 19 ]. Prior efforts to replicate human gaze behavior in robots typically relied on
hand-crafted heuristics [ 20 , 21 , 22 , 23 ], but such strategies are difficult to generalize across diverse
tasks. A more scalable approach should allow the robot to learn active perception strategies that
maximize task-relevant information gain, without requiring task-specific assumptions.

In this paper, we introduce Vision in Action (ViA) , a bimanual manipulation system that learns
active perception strategies directly from human demonstrations. Our system addresses the above
challenges using the following design choices:

• Flexible robot neck using an off-the-shelf 6-DoF arm. Instead of replicating the intricate biome-
chanics of the human neck and torso through a complex design, we use an off-the-shelf 6-DoF
robot arm as the robot’s neck. This simple yet effective approach enables human-like head move-
ments that approximate the full range of motion produced by coordinated upper-body motion.
• Intermediate 3D representation to decouple human and robot motion in VR teleoperation. Instead
of directly mirroring human head movements and streaming live robot camera views, we use an
intermediate 3D scene representation. This representation enables real-time rendering of novel
views based on the human’s latest head pose, without requiring new observations from the robot.
Consequently, the robot can be slowed down to reflect aggregated head movements rather than
every motion. This asynchronous streaming, control, and rendering bypasses the need for low-
latency robot actuation and data transmission.
• Shared-observation teleoperation as a scalable way to capture active perception strategies. In-
stead of hand-designing a gaze strategy, we let the policy learn the strategy directly from hu-
man demonstrations. By having the human use the same observation space as the robot— seeing
what the robot sees —we effectively capture the human’s complex perceptual strategies across task
stages and scenarios. This enables the visuomotor policy to learn robust gaze behavior, even with
straightforward behavior cloning.

To evaluate our proposed system, we perform experiments on three challenging, multi-stage bi-
manual manipulation tasks involving significant visual occlusions. These tasks include retrieving
objects with interactive perception, rearranging cups in cluttered environments with active view-
point switching, and precisely aligning objects using coordinated bimanual actions. Our experimen-
tal results highlight the critical role of active perception, with ViA outperforming baseline camera
setups—such as wrist cameras and fixed chest cameras—by 45% in success rate. We also conducted
a user study to validate the design of our teleoperation interface. Results are best viewed on our
website: https://vision-in-action.github.io .

2
Related Work

Active Perception and Robot Necks. Active perception has a long-standing history in robotics
and computer vision [ 1 , 24 , 25 , 26 , 18 , 17 , 19 ]. To investigate active perception, many artificial
vision systems ( i.e ., humanoid necks with varying numbers of degrees of freedom) have been de-
veloped [ 11 , 12 , 13 , 27 , 14 , 28 , 29 , 30 , 31 ]. In this paper, we use an off-the-shelf 6-DoF arm as

2

# Page 3
d) Aggregated Head & Arm Movements

b) View Rendering (Low latency)

c) Real-Time Head Pose

a) RGB-Depth Observations

…

Prior Work
Synchronized RGB Teleop

Robot
Human

Head Pose
(Robot Control Latency)

RGB Streaming
(Transmission Latenc y)
Motion

Sick

ViA System
Decoupled Rendering &

Asynchronous Update
e) World Frame

3D Scene

Figure 2: VR Teleoperation Comparison. [Left] Traditional RGB streaming suffers from motion-to-photon
latency due to both RGB data transmission latency and robot control latency, often leading to VR motion
sickness. [Right] Our system mitigates this by: (a, e) streaming a 3D point cloud in the world frame from RGB-
D data, (b, c) performing real-time view rendering based on the user’s latest head pose, and (d) asynchronously
updating the robot’s head and arm poses. This approach enables low-latency viewpoint updates for the user.

an active neck, with a camera mounted to the end effector. While the idea of using a robot arm as
a neck has been explored in prior work [ 32 , 33 , 34 , 15 , 35 ], our approach integrates a novel tele-
operation interface to directly control the robot neck. The majority of prior active vision systems
use various heuristics ( e.g ., hand-designed image filters or object detectors) to compute a measure
of saliency for gaze guidance [ 20 , 21 , 22 ]. There are also works that formulate the next-best-view
problem in terms of uncertainty reduction [ 36 , 37 , 23 ]. In these types of methods, the objective
is defined purely around a perception problem, and do not consider manipulation. Recent works
have also investigated active vision with reinforcement learning approaches [ 38 , 39 , 40 , 41 , 35 , 42 ],
though application of those methods to real-world systems remains challenging. In contrast, our
work learns bimanual manipulation and active perceptual behavior directly from real-world human
demonstrations, without any task-specific assumptions.

Teleoperation Systems.
Recent teleoperation works [ 2 , 3 , 10 , 43 , 44 , 45 , 46 ] have high-
lighted the potential of scaling end-to-end visuomotor policy learning.
However, existing ap-
proaches [ 2 , 3 , 47 , 9 ] typically rely on wrist cameras [ 9 , 48 ] or fixed third-person cameras [ 49 ],
which fail to capture the active perceptual behaviors of humans. To overcome this human-robot
observation mismatch, prior works [ 14 , 15 , 50 , 51 , 52 , 53 , 34 , 54 ] have explored using VR to con-
trol an active head camera [ 14 ], providing immersive, first-person visual feedback through RGB
video streaming. However, these direct camera teleoperation approaches [ 14 , 52 ] often induce mo-
tion sickness [ 55 ], primarily due to motion-to-photon latency—the delay between the user’s head
movement and the corresponding visual update on the VR display [ 56 ]. To address this, our method
introduces an intermediate 3D scene representation that enables real-time view rendering based on
the user’s latest head pose, significantly reducing motion-to-photon latency. Related to our ap-
proach, recent work [ 57 ] introduced a VR teleoperation system that uses radiance fields to render
views from a reconstructed scene. However, unlike our approach, their system lacks physical cam-
era control. Our approach, by contrast, allows users to purposefully control the camera in VR to
maintain task-relevant visibility.

3
The Vision in Action System

The ViA system features a simple yet effective robotic neck design that allows the robot to mimic
human whole-upper-body movements (§ 3.1 ). We introduce a 3D scene interface that renders views
in real-time based on the user’s latest head pose. This interface asynchronously updates the underly-
ing 3D environment while allowing the user to purposefully control the robot’s active camera (§ 3.2 ).
Finally, we propose a visuomotor policy learning framework that leverages active perception (§ 3.3 ).

3.1
Hardware Design

Human active perception relies on coordinated movements of both the torso and neck to adjust head
poses and acquire better viewpoints. However, the common approach of mounting a 2-DoF neck on
a static torso [ 14 ] provides limited flexibility and is insufficient to replicate the full range of motion.

3

# Page 4
To address this, we use an off-the-shelf 6-DoF ARX5 robot arm as a robot neck. This high-DoF neck
design allows the robot to mimic human-like head motions that naturally result from whole-upper-
body movements. The active head camera streams real-time RGB, depth, and synchronized camera
pose data. To meet these requirements, we use an iPhone 15 Pro [ 6 ], mounted on the end effector of
the robot neck, as the system’s primary visual sensor. To enable bimanual manipulation, we use two
additional 6-DoF ARX5 robot arms [ 58 ] each equipped with a fin-ray parallel-jaw gripper. Each
arm is mounted onto a custom 3D-printed shoulder structure.

3.2
Teleoperation Interface

To collect human demonstration data, we designed a teleoperation interface that simultaneously
controls both robot arms and the active neck. For the arm teleoperation, we use a full-scale bimanual
exoskeleton (inspired by GELLO [ 47 ]) that enables joint-to-joint mapping between the human user
and the robot arms. For head teleoperation, we implemented a VR interface that allows the user
to control the pose of the active head camera while observing visual feedback. Our choice of VR
for the head interface was motivated by the need to precisely capture human perceptual strategies.
By constraining the user to use the same observations as the robot, we can record visual attention
patterns that contribute to successful task execution.

Challenge: Exacerbated latency from physical movements. In the VR literature, motion-to-photon
latency, also known as end-to-end latency [ 56 ], refers to the delay between a user’s head movement
and the corresponding visual update on the display. High latency can cause discomfort or motion
sickness. While today’s consumer VR headsets achieve acceptable motion-to-photon latency (below
10 ms) for applications like games [ 59 ], robot teleoperation introduces an additional challenge—
robot control latency . When users move their heads to teleoperate the robot’s camera, there is a
delay between the robot receiving and executing the command, causing the camera control to lag
behind. This additional delay creates a mismatch between the user’s head movement and visual
feedback, leading to potential motion sickness.

Solution: View decoupling through an intermediate 3D scene representation. To overcome this
challenge, we decouple the user’s view from the robot’s view using an intermediate 3D scene rep-
resentation (Fig. 2 ). This allows the user’s viewpoint to update instantly in response to head move-
ments (via rendering), without waiting for the robot to physically match the requested viewpoint.
While the rendered view may contain small regions with missing information (due to the delayed
camera movement), the rendered view stays aligned with the user’s latest head pose —a critical fac-
tor for preserving perceptual continuity and reducing discomfort. Concretely, the interface has three
components:

• Point cloud construction in the world frame. We define the world frame W at the fixed base of the
robot neck. Each RGB-D frame is transformed into this world frame using the camera intrinsics
and the robot head pose ( i.e ., camera extrinsics w.r.t. the world frame) at time t , denoted as W T H ( t ) .
This pose is computed by composing the iPhone’s real-time relative pose with the initial robot head
pose W T H ( t 0 ) , obtained from the robot neck’s joint positions. The resulting point cloud W X ( t ) in
the world frame serves as our intermediate 3D scene representation.
• Low-latency view rendering. From the point cloud W X ( t ) , we render stereo RGB views for the
VR display using the user’s latest head pose in the world frame, denoted as W T user ( t + k ) . This
pose is computed by transforming the VR device’s head pose into the world frame W with a
height offset. This view rendering—where k denotes a short time interval—enables instant visual
feedback for the user. Combined with a high refresh rate (roughly 150 Hz), our system ensures
smooth viewpoint updates with minimal perceived latency.
• Point cloud updating with aggregated head movements. Finally, the robot head pose is updated
to W T H ( t + K ) over a longer time interval K , using the aggregated user head pose, where K is
determined by the robot’s control latency and is much larger than the rendering interval k . Mean-
while, the point cloud is asynchronously updated with new RGB-D observations from the robot at
a lower frequency (10 Hz).

4

# Page 5
Overall, this teleoperation interface balances low visual latency for the user ( < 7 ms) with smooth
action execution on the robot side ( < 10 Hz control frequency), enabling effective and practical data
collection for complex manipulation tasks.

3.3
Learning Active Perception for Bimanual Manipulation

We design a visuomotor policy network based on Diffusion Policy [ 10 ] that leverages our active
head camera setup to learn from human demonstrations. The policy predicts bimanual arm actions
for manipulation and neck actions that mimic human active perception behaviors, conditioned on
visual and proprioceptive observations.

To enable coordinated head and arm movements, we represent the end-effector poses of the neck
and arms in a common world frame. At each time step t , the policy receives the current RGB
image observation I t ∈ N H × W × C
0
from the active head camera as the visual input, along with the
proprioceptive state P t ∈ R 23 . This state includes the end-effector poses (position and quaternion)
of the neck, left arm, and right arm ( ∈ R 7 ), as well as the two gripper widths (2 scalars).

We adopt a DINOv2 [ 60 ] pretrained ViT as the visual encoder for the RGB image I t from the active
head camera. The 384-dimensional classification token is extracted as a compact semantic represen-
tation of the visual scene. The policy outputs a sequence of future actions A t = { a t + 1 ,..., a t + n p } ∈
R n p × 23 , where each action consists of the future end-effector poses of the neck and arms in the world
frame, as well as the gripper widths. Only the first n a ≤ n p actions are executed on the physical robot
(via inverse kinematics). We use a prediction horizon of n p = 16 and an execution horizon of n a = 8,
with the policy operating at 10 Hz.

4
Evaluation

We evaluated our system on three challenging multi-stage tasks (Fig. 3 ) to assess the effectiveness
of various camera setups (§ 4.1 ), visual representations (§ 4.2 ), and teleoperation interface designs
(§ 4.3 ). For each task, we report the stage-wise success, which is defined cumulatively ( i.e ., success at
each stage requires the successful completion of all preceding stages). Detailed task stage definitions
can be found in the supplementary material.

Bag Task : Object retrieval with interactive perception. The robot must (1) open a bag, (2) peek
inside to locate the target object, and (3) take it out. Success requires both active physical interaction
( i.e ., opening the bag to reduce occlusion) and active head movement to inspect the bag’s interior,
demonstrating interactive perception. The wrist camera often suffers from limited visibility due to
occlusions, whereas the active head camera can dynamically adjust its viewpoint to gather task-
relevant information more effectively. We collected 150 demonstrations with five training objects
(banana, carrot, dog, shoe, strawberry) and evaluated on two unseen test objects (a blue elephant,
a green avocado) with 5 rollouts per object—10 trials in total. For both training and evaluation, a
single object is placed in the bag per trial.

Cup Task : Cup arrangement with active viewpoint switching. As illustrated in Fig. 3 , the robot must
(1) find and pick up a cup from shelf A using its right hand, (2) hand it over to its left hand, and
(3) place it on a saucer hidden beneath shelf B. Visual occlusion presents a significant challenge,
requiring active viewpoint switching across different stages: the cup is positioned deep within shelf
A, where upper tiers obstruct wrist cameras, while the saucer is positioned beneath shelf B. We
collected 125 training demonstrations, with the cup randomly placed on either the upper or lower
tier of shelf A and the saucer randomly positioned beneath shelf B. Demonstrations followed a
consistent search strategy (lower tier first, then upper if needed). For evaluation, we used 10 test
configurations, each run twice, resulting in 20 total rollouts.

Lime & Pot Task : Bimanual coordination and precise alignment. The robot must (1) find and place
a lime into a pot, (2) lift the pot using both arms, and (3) precisely align it onto a trivet. Since
the lime may appear on either side of the workspace, the robot must first coordinate and decide
which arm to use for grasping. Lifting the pot requires bimanual grasping , and the final precise
alignment with the trivet is guided by the head camera to ensure precise placement. We collected

5

# Page 6
①

②
③

①

②

③

①

②

③

① ②

③

①

②

③

①

②

③

Figure 3: Task Definitions. We introduce three multi-stage tasks that highlight the critical role of active
perception in everyday scenarios. [Left] Third-person view with red arrows indicating head movements and
blue arrows indicating arm movements. [Middle] Active head camera views across task stages (upper row), and
third-person view of robot actions (lower row). [Right] Test scenarios, including training and testing objects
for the bag task, and different test configurations for the latter two tasks.
260 demonstrations for training. For evaluation, we fixed the pot position and tested 10 different
lime and trivet configurations, each tested twice for 20 total rollouts.
4.1
Policy Learning Camera Setup Comparison

Camera Setups. We evaluate the effectiveness of active head camera setup by comparing it with
two alternative camera configurations for policy learning (Fig. 4 ). During data collection, all camera
streams are recorded. For training, we use different combinations of these views from the same
set of demonstrations, enabling a fair comparison across camera setups. Visual representations are
extracted using a DINOv2 pretrained ViT backbone [ 60 ].

• [ ViA (Ours) ]: Uses a single active head camera as the visual input. Details are described in § 3.3 .
• [ Active Head & Wrist Cameras ]: Combines the active head camera with two wrist cameras.
Compared to [ViA], this setup includes additional wrist views as visual input. Although the
teleoperator does not directly use these views, this comparison evaluates whether they provide
additional useful information for policy learning.
• [ Chest & Wrist Cameras ]: Uses a fixed chest camera and two wrist cameras (omitting the neck).
This is one of the most commonly used camera setups in current robotics systems [ 2 , 3 ].

Results. As shown in Fig. 5 , [ViA] consistently outperforms both alternative camera setups across
all three tasks. Surprisingly, augmenting [ViA] with additional wrist camera observations ([Active
Head & Wrist Cameras]) does not improve performance (a decrease of 18.33% on average). We

6

# Page 7
Head view
Left wrist view
Chest view
Right wrist view

Lime & Pot
Cup
Bag

Third-person view

Figure 4: Policy Learning Camera Setup Comparison. [ViA] uses a single active head camera that dynami-
cally adjusts its viewpoint to capture task-relevant visual information ( e.g ., finding a cup hidden inside a shelf).
In contrast, [Wrist & Chest cameras] policy often fails due to visual occlusions . For example, in the cup task,
the right wrist camera’s view is blocked by the upper shelf tier, resulting in insufficient visual cues for grasping.
The chest camera also fails to capture task-relevant information due to its fixed viewpoint, even when equipped
with a fisheye lens.

Open
Grasp
Take Out

Bag Task

20

40

60

80

100

Success Rate (%)

100

90
90
100

80
80
80

70
70

Find & Pick Up
Hand Over
Place

Cup Task

20

40

60

80

100

80
80
80

65
65
65

25
20
20

Find
Pick Up
BiGrasp & Align

Lime & Pot Task

20

40

60

80

100
95

55
55
55

25
25
20

5
0

ViA (Ours)
Active Head & Wrist Cameras
Chest & Wrist Cameras

Figure 5: Policy Learning Camera Setup Comparison Results. We report stage-wise success rates across
the three tasks to demonstrate the effectiveness of our active head camera [ViA] compared to two baseline
configurations: [Active Head & Wrist Cameras] and [Chest & Wrist Cameras].

hypothesize several reasons for this outcome: First, the active head camera alone already provides
sufficient information, as the teleoperator relies solely on this view to complete the task. Thus, the
visual input from the head camera alone is already task-complete. Second, adding wrist cameras in-
creases input dimensionality without necessarily contributing task-relevant information. Instead, the
additional views may introduce redundant or noisy observations, especially due to frequent occlu-
sions during manipulation. In a low-data regime like ours, the added complexity can hinder learning
by increasing the risk of overfitting or distracting the model with less informative inputs.

Next, compared to the [Chest & Wrist Cameras] setup, it is clear that the chest and wrist cameras
fail to provide sufficient task-relevant information. As shown in the second row of Fig. 4 , the right
wrist camera is completely occluded by the upper shelf tier during cup-grasping, while the fixed
chest camera lacks visibility of the target objects altogether. In contrast, our active head camera
dynamically adjusts its viewpoint, allowing the robot to gather more informative visual input and
improve average task performance by 45%.

4.2
Policy Learning Visual Representation Comparison

Visual Representations. We compare [ViA] with two alternative visual representations for the
policy. All policies use the same active head camera input as [ViA].

• [ ViA (Ours) ]: Uses a DINOv2 [ 60 ] vision backbone for image encoding. Details can be found in
§ 3.3 .

7

# Page 8
Open
Grasp
Take Out

Bag Task

20

40

60

80

100

Success Rate (%)

100

90
90
100

60
60

0
0
0

Find & Pick Up
Hand Over
Place

Cup Task

20

40

60

80

100

80
80
80
90

75

50

65

45

35

Find
Pick Up
BiGrasp & Align

Lime & Pot Task

20

40

60

80

100
95

55
55

40

30

15

40

0
0

ViA (Ours)
ResNet-DP
DP3

Figure 6: Policy Learning Visual Representation Comparison Results. We report stage-wise success rates
across the three tasks to demonstrate the effectiveness of our method [ViA], in comparison to two baseline
approaches: [ResNet-DP] and [DP3].
• [ ResNet-DP ]: A baseline using a ResNet-18 [ 61 ] backbone pretrained on ImageNet [ 62 ], inte-
grated into diffusion policy. Input images are center-cropped to 1:1 aspect ratio and resized to
224 × 224, consistent with [ViA].
• [ DP3 ] [ 63 ]: Uses world-frame point clouds (transformed from the active head camera) as visual
input. The point cloud is cropped to the workspace and downsampled to 1,024 points. This model
is trained from scratch.

Results. As shown in Fig. 6 , our method—leveraging a pretrained DINOv2 ViT representation—
achieves the highest final-stage success rate across all three tasks. Compared to the two baselines,
[ViA] benefits from stronger semantic understanding enabled by the DINOv2 backbone. This allows
the policy to actively find the object first before initiating arm actions. For example, in the lime &
pot task, [ViA] is able to perform long-horizon active search to find the lime, before proceeding with
manipulation. In contrast, a common failure mode of the [DP3] baseline is hallucination, where the
policy misinterprets the scene and issues incorrect actions. For example, in the cup task, [DP3] often
directs the arm to an empty section of the shelf, failing to identify the actual cup location. [DP3]
also completely fails on the bag task due to the imprecise grasping of the bag handle in the open
stage. We hypothesize that this is due to the limited semantic capacity of the [DP3] representation,
which is trained from scratch and lacks pretrained visual priors.
4.3
Teleoperation Interface Comparison

0

2

4

Level

2.0

3.375

Motion Sickness
0

25

50

Avg Time (s)

56.74
52.72

Data Collection

75.0%

25.0%

User Preference

Point-Cloud Rendering (Ours)
Stereo-RGB Streaming

Figure 7: Teleoperation Interface Comparison. We
evaluate our teleoperation interface design based on
three metrics: reported levels of motion sickness, aver-
age duration to complete each demonstration, and over-
all user preference.

In this experiment, we evaluate our VR tele-
operation interface by comparing our system—
which uses a point cloud rendering method—
with
a
conventional
RGB
streaming
ap-
proach [ 14 , 15 , 52 ]. We conducted a user study
with 8 participants of varying levels of expe-
rience with VR and robot teleoperation.
All
participants were first-time users of both sys-
tems and were unaware of which system corre-
sponded to our proposed design. For each participant, the order of system usage was randomized and
labeled as System A and System B. Participants were asked to perform the cup task using both sys-
tems. Each session included a 5-minute practice period followed by a data collection phase in which
participants provided 3 demonstrations. We recorded the completion time for each demonstration
and gathered user feedback through a post-session experience survey.

Results. As shown in Fig. 7 , while our point cloud rendering method results in slightly longer data
collection times compared to stereo RGB streaming, it significantly reduces motion sickness. As a
result, 6 out of 8 participants reported a preference for our system.

5
Conclusion
The ViA system features a simple yet effective neck design that allows the robot to mimic human-
like head movements. We developed a teleoperation interface that renders real-time views based on
the user’s latest head pose, while asynchronously updating the scene by controlling the robot’s active
head camera to gather task-relevant information. For evaluation, we introduced three challenging
multi-stage tasks involving significant visual occlusion for policy learning. Experimental results
highlight the importance of active perception, with ViA significantly outperforming baseline setups.

8
