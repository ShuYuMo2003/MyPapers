# Page 1
Universal Manipulation Interface:
In-The-Wild Robot Teaching Without In-The-Wild Robots

Cheng Chi ∗ 1 , 2 , Zhenjia Xu ∗ 1 , 2 , Chuer Pan 1 , Eric Cousineau 3 , Benjamin Burchfiel 3 , Siyuan Feng 3 ,
Russ Tedrake 3 , Shuran Song 1 , 2

1 Stanford University, 2 Columbia University, 3 Toyota Research Insititute

https://umi-gripper.github.io

for Any Actions

(action diversity)
Human Demonstration
in Any Environment

(visual diversity)

for Many Robots
(embodiment diversity)

Dynamic

Bimanual

Precise

Long-Horizon
7DoF

6DoF

Fig. 1: Universal Manipulation Interface (UMI) is a portable, intuitive, low-cost data collection and policy learning framework. This
framework allows us to transfer diverse human demonstrations to effective visuomotor policies. We showcase the framework for tasks that
would be difficult with traditional teleoperation, such as dynamic, precise, bimanual and long-horizon tasks.

Abstract —We present Universal Manipulation Interface (UMI)
– a data collection and policy learning framework that allows
direct skill transfer from in-the-wild human demonstrations
to deployable robot policies. UMI employs hand-held grippers
coupled with careful interface design to enable portable, low-
cost, and information-rich data collection for challenging bi-
manual and dynamic manipulation demonstrations. To facilitate
deployable policy learning, UMI incorporates a carefully designed
policy interface with inference-time latency matching and a
relative-trajectory action representation. The resulting learned
policies are hardware-agnostic and deployable across multiple
robot platforms. Equipped with these features, UMI framework
unlocks new robot manipulation capabilities, allowing zero-
shot generalizable dynamic, bimanual, precise, and long-horizon
behaviors, by only changing the training data for each task. We
demonstrate UMI’s versatility and efficacy with comprehensive
real-world experiments, where policies learned via UMI zero-
shot generalize to novel environments and objects when trained
on diverse human demonstrations. UMI’s hardware and software
system is open-sourced at https://umi-gripper.github.io .

I. I NTRODUCTION

How should we demonstrate complex manipulation skills
for robots to learn from? Attempts in the field have approached
this question primarily from two directions: collecting targeted
in-the-lab robot datasets via teleoperation or leveraging un-
structured in-the-wild human videos. Unfortunately, neither

∗ Indicates equal contribution

is sufficient, as teleoperation requires high setup costs for
hardware and expert operators, while human videos exhibit
a large embodiment gap to robots.
Recently, using sensorized hand-held grippers as a data
collection interface [ 41 , 50 , 36 ] has emerged as a promising
middle-ground alternative – simultaneously minimizing the
embodiment gap while remaining intuitive and flexible. De-
spite their potential, these approaches still struggle to balance
action diversity with transferability. While users can theoreti-
cally collect any actions with these hand-held devices, much
of that data can not be transferred to an effective robot policy.
As a result, despite achieving impressive visual diversity
across hundreds of environments, the collected actions are
constrained to simple grasping [ 41 ] or quasi-static pick-and-
place [ 50 , 36 ], lacking action diversity .
What prevents action transfer in previous work? We identi-
fied a few subtle yet critical issues:

• Insufficient visual context: While using a wrist-mounted
camera is key for aligning the observation space and
enhancing device portability, it restricts the scene’s visual
coverage. The camera’s proximity to the manipulated
object often results in heavy occlusions, providing insuf-
ficient visual context for action planning.

• Action imprecision: Most hand-held devices rely on
monocular structure-from-motion (SfM) to recover robot
actions. However, such methods often struggle to recover

arXiv:2402.10329v3  [cs.RO]  6 Mar 2024

# Page 2
② Wide-FoV
Fisheye Lens

⑥ Kinematic-based Data Filtering

③ Side Mirrors
for Implicit
Stereo

Human Demonstration Setup
Robot Setup
Observation Space

⑤ C ontinuous Gripper Tracking

✨

✨

✨

① Wrist-mounted
Camera

④ IMU-aware Pose Tracking

Fig. 2: UMI Demonstration Interface Design. Left: Hand-held grippers for data collection, with a GoPro as the only sensor and recording
device. Middle: Image from the GoPro’s 155° Fisheye view. Note the physical side mirrors highlighted in green which provide implicit
stereo information. Right: UMI-compatible robot gripper and camera setup make observation similar to hand-held gripper view.

precise global action due to scale ambiguity, motion blur,
or insufficient texture, which significantly restrict the
precision of tasks for which the system can be employed.

• Latency discrepancies: During hand-held data collec-
tion, observation and action recording occur without la-
tency. However, during inference, various latency sources,
including sensor, inference, and execution latencies, arise
within the system. Policies unaware of these latency dis-
crepancies will encounter out-of-distribution input and in
turn, generate out-of-sync actions. This issue is especially
salient for fast and dynamic actions.

• Insufficient policy representation: Prior works often
use simple policy representations (e.g., MLPs) with ac-
tion regression loss, limiting their capacity to capture
complex multimodal action distributions inherent in hu-
man data. Consequently, even with precisely recovered
demonstrated actions and all discrepancies removed, the
resulting policy could still struggle to fit the data ac-
curately. This further hampers large-scale, distributed
human data collection, as more demonstrators increase
action multimodality.

In this paper, we address these issues with careful design
of the demonstration and policy interface:

• First, we aim to identify the right physical interface

for human demonstration that is intuitive and meanwhile
able to capture all the information necessary for policy
learning. Specifically, we use a Fisheye lens to increase
the field of view and visual context, and add side mirrors
on the gripper to provide implicit stereo observation.
When combined with the GoPro’s built-in IMU sensor,
we can enable robust tracking under fast motion.

• Second, we explore the right policy interface (i.e.,
observation and action representations) that could make
the policy hardware-agnostic and thereby enable effec-
tive skill transfer. Concretely, we employ inference-time
latency matching to handle different sensor observation
and execution latency, use relative trajectory as action
representation to remove the need for precise global
action, and finally, apply Diffusion Policy [ 9 ] to model

multimodal action distributions.
The final system, Universal
Manipulation
Interface
(UMI) , provides a practical and accessible framework to
unlock new robot manipulation skills, allowing us to demon-
strate any actions in any environment while maintaining high
transferability from human demonstration to robot policy.
With just a wrist-mounted camera on the hand-held gripper
(Fig. 2 ), we show that UMI is capable of achieving a wide
range of manipulation tasks that involve dynamic, bimanual,
precise and long-horizon actions by only changing the train-
ing data for each task (Fig. 1 ). Furthermore, when trained
with diverse human demonstrations, the final policy exhibits
zero-shot generalization to novel environments and objects,
achieving a remarkable 70% success rate in out-of-distribution
tests, a level of generalizabilty seldomly observed in other
behavior cloning frameworks. We open-source the hardware
and software system at https://umi-gripper.github.io .
II. R ELATED W ORKS

A key enabler for any data-driven robotics system is the data
itself. Here, we review a few typical data collection workflows
in the context of robotic manipulation.

A. Teleoperated Robot Data

Imitation learning learns policies from expert demonstra-
tions. Behavior cloning (BC), utilizing teleoperated robot
demonstrations, stands out for its direct transferability. How-
ever, teleoperating real robots for data collection poses signif-
icant challenges. Previous approaches utilized interfaces such
as 3D spacemouse [ 9 , 54 ], VR or AR controllers [ 35 , 3 , 13 , 19 ,
31 , 51 , 12 ], smartphones [ 44 , 45 , 22 ], and haptic devices [ 38 ,
47 , 43 , 26 , 4 ] for teleoperation. These methods are either
very expensive or hard to use due to high latency and lack
of user intuitiveness. While recent advancements in leader-
follower (i.e. puppetting) devices such as ALOHA [ 53 , 15 ]
and GELLO [ 46 ] offer promise with intuitive and low-cost
interfaces, their reliance on real robots during data collection
limits the type and number of environments the system can
gain access to for “in-the-wild” data acquisition. Exoskele-
tons [ 14 , 20 ] remove the dependence on real robots during data

# Page 3
collection, however, require fine-tuning using teleoperated real
robot data for deployment. Moreover, the resulting data and
policy from aforementioned devices are embodiment-specific,
preventing reusage for different robots.
In contrast, UMI eliminates the need for physical robots
during data collection and offers a more portable interface for
in-the-wild robot teaching, providing data and policies that
are transferable to different robot embodiments (e.g., 6DoF or
7DoF robot arms).

B. Visual Demonstrations from Human Video
There’s a distinct line of work dedicated to policy learning
from in-the-wild video data (e.g. YouTube videos). The most
common way is to learn from diverse passive human demon-
stration videos. Utilizing passive human demonstrations, pre-
vious works learn task cost functions [ 37 , 8 , 1 , 21 ], affordance
functions [ 2 ], dense object descriptors [ 40 , 24 , 39 ], action
correspondences [ 33 , 28 ], and pre-trained visual representa-
tions [ 23 , 48 ].
However, this approach encounters three major challenges.
Firstly, most video demonstrations lack explicit action infor-
mation, crucial for learning generalizable policies. To infer
action data from passive human video, previous works resort
to hand pose detectors [ 44 , 1 , 38 , 28 ], or combining human
videos with in-domain teleoperated robot data to predict
actions [ 33 , 20 , 34 , 28 ]. Second, the evident embodiment gap
between humans and robots hinders action transfer. Efforts
to bridge the gap include learning human-to-robot action
mapping with hand pose retargetting [ 38 , 28 ] or extracting
embodiment-agnostic keypoints [ 49 ]. Despite these attempts,
the inherent embodiment differences still complicate policy
transfer from human video to physical robots. Thirdly, the
inherent observation gap induced by the embodiment gap
in this line of work introduces inevitable mismatch between
train/inference time observation data, exacerbating the trans-
ferability of the resulting policies, despite efforts in aligning
demonstration observation with robot observation [ 20 , 28 ].
In contrast, data collected with UMI exhibit minimal em-
bodiment gap both in action and observation spaces, en-
abled by precise manipulation action extraction via robust
visual-inertial camera tracking and the shared Fisheye wrist-
mounted cameras during teaching and testing. Consequently,
this enables in-the-wild zero-shot policy transfer for dynamic,
bimanual, precise, and long-horizon manipulation tasks.
C. Hand-Held Grippers for Quasi-static Actions
Hand-held grippers [ 41 , 50 , 10 , 32 , 27 , 25 ] minimize
observation embodiment gaps in manipulation data collection,
offering portability and intuitive interfaces for efficient data
collection in the wild. However, accurately and robustly ex-
tracting 6DoF end-effector (EE) pose from these devices re-
mains challenging, hindering the deployment of robot policies
learned from these data on fine-grained manipulation tasks.
Prior works attempted to address this issue through various
approaches, such as SfM [ 50 , 25 ] which suffers from scale
ambiguity; RGB-D fusion [ 41 ] which requires expensive sen-
sors and onboard compute; external motion tracking [ 32 , 27 ]

(a) Raw Fisheye Image
(b) Rectiﬁed Image

Fig. 3: Fisheye vs Rectilinear (a) UMI policies use raw Fisheye
image as observation. (b) Rectifying a large 155° FoV image to the
pin-hole model severely stretches the peripheral view (outside of blue
line ), while compresses the most important information at the center
to a small area (inside of red line ).

which is limited to lab settings. These devices, constrained
to quasi-static actions due to low EE tracking accuracy and
robustness, often necessitate cumbersome onboard computer
or external motion capture (MoCap) systems, diminishing
their feasibility for in-the-wild data collection. In contrast,
UMI integrates state-of-the-art SLAM [ 6 ] with built-in IMU
data from GoPro, to accurately capture 6DoF actions at the
global scale. The high-accuracy data enables trained BC policy
to learn bimanual tasks. With thorough latency matching,
UMI further enables real-world deployable policy for dynamic
actions such as tossing.
Recently, Dobb-E [ 36 ] proposed a “reacher-grabber” tool
mounted with an iPhone to collect single-arm demonstrations
for the Stretch robot. Yet, Dobb-E only demonstrates policy
deployment for quasi-static tasks and requires environment-
specific policy fine-tuning. Conversely, using only data col-
lected with UMI enables trained policy to zero-shot generalize
to novel in-the-wild environments, unseen objects, multiple
robot embodiments, for dynamic, bimanual, precise and long-
horizon tasks.

III. M ETHOD

Universal Manipulation Interface (UMI) is hand-held data
collection and policy learning framework that allows direct
transfer from in-the-wild human demonstrations to deployable
robot policies. It is designed with the following goals in mind:

• Portable. The hand-held UMI grippers can be taken to
any environment and start data collection with close-to-zero
setup time.

• Capable. The ability to capture and transfer natural and
complex human manipulation skills beyond pick-and-place.

• Sufficient. The collected data should contain sufficient
information for learning effective robot policies and con-
tain minimal embodiment-specific information that would
prevent transfer.

• Reproducible : Researchers and enthusiasts should be able
to consistently build UMI grippers and use data to train
their own robots, even with different robot arms.

The following sections describe how we enable the above
goals through our hardware and policy interface design.

# Page 4
A. Demonstration Interface Design

UMI’s data collection hardware takes the form of a trigger-
activated, handheld 3D printed parallel jaw gripper with soft
fingers, mounted with a GoPro camera as the only sensor
and recording device (see HD1 ). For bimanual manipulation,
UMI can be trivially extended with another gripper. The key
research question we need to address here is:

How can we capture sufficient information for a wide
variety of tasks with just a wrist-mounted camera?

Specifically, on the observation side, the device needs
to capture sufficient visual context to infer action HD2 and
critical depth information HD3 . On the action side, it needs
to capture precise robot action under fast human motion
HD4 , detailed subtle adjustments on griping width HD5 , an d
automatically check whether each demonstration is valid given
the robot hardware kinematics HD6 . The following sections
describe details on how we achieve these goals.
HD1. Wrist-mounted cameras as input observation. We
rely solely on wrist-mounted cameras, without the need for
any external camera setups. When deploying UMI on a robot,
we place GoPro cameras with the same location with respect
to the same 3D-printed fingers as on the hand-held gripper.
This design provides the following benefits:
1) Minimizing the observation embodiment gaps. Thanks
to our hardware design, the videos observed in wrist-
mount cameras are almost indistinguishable between human
demonstrations and robot deployment, making the policy
input less sensitive to embodiment.
2) Mechanical robustness. Because the camera is mechani-
cally fixed relative to the fingers, mounting UMI on robots
does not require camera-robot-world calibration. Hence, the
system is much more robust to mechanical shocks, making
it easy to deploy.
3) Portable hardware setup. Without the need for an external
static camera or additional onboard compute, we largely
simplify the data collection setup and make the whole
system highly portable.
4) Camera motion for natural data diversification. A side
benefit we observed from experiments is that when training
with a moving camera, the policy learns to focus on task-
relevant objects or regions instead of background structures
(similar in effect to random cropping). As a result, the final
policy naturally becomes more robust against distractors at
inference time.
Avoiding use of external static cameras also introduce
additional challenges for downstream policy learning. For
example, the policy now needs to handle non-stationary and
partial observations. We mitigated these issues by leveraging
wide-FoV Fisheye Lens HD2 , and robust visual tracking HD4 ,
described in the following sections.
HD2. Fisheye Lens for visual context.
We use a 155-
degree Fisheye lens attachment on wrist-mounted GoPro cam-
era, which provides sufficient visual context for a wide range
of tasks, as shown in Fig. 2 . As the policy input, we directly

Right
Virtual
Camera
Main
Camera

Left
Virtual
Camera

Ultra-wide angle 155° FOV

Mirror

Extra view
from mirror

✨

Digital
Reﬂection

Raw Image
Policy Observation

(a)
(b)

(c)

Fig. 4: UMI Side Mirrors. The ultra-wide-angle camera coupled
with strategically positioned mirrors, facilitates implicit stereo depth
estimation. (a) : The view through each mirror effectively creates two
virtual cameras, whose poses are reflected along the mirror planes
with respect to the main camera. (b) : Ketchup on the plate, occluded
from the main camera view, is visible inside the right mirror, proving
that mirrors simulate cameras with different optical centers. (c) : We
digitally reflect the content inside mirrors for policy observation. Note
the orientation of the cup handle becomes consistent across all 3
views after reflection.

use raw Fisheye images without undistortion since Fisheye
effects conveniently preserve resolution in the center while
compressing information in the peripheral view. In contrast,
rectified pinhole image (Fig. 3 right) exhibits extreme distor-
tions, making it unsuitable for learning due to the wide FoV.
Beyond improving SLAM robustness with increased visual
features and overlap [ 52 ], our quantitative evaluation (Sec
V-A ) shows that the Fisheye lens improves policy performance
by providing the necessary visual context.
HD3. Side mirrors for implicit stereo.
To mitigate the
lack of direct depth perception from the monocular camera
view, we placed a pair of physical mirrors in the cameras’
peripheral view which creates implicit stereo views all in
the same image. As illustrated in Fig 4 (a), the images
inside the mirrors are equivalent to what can be seen from
additional cameras reflected along the mirror plane, without
the additional cost and weight. To make use of these mirror
views, we found that digitally reflecting the crop of the images
in the mirrors, shown in Fig 4 (c), yields the best result for
policy learning (Sec. V-A ). Note that without digital reflection,
the orientation of objects seen through side mirrors is the
opposite of that in the main camera view.
HD4. IMU-aware tracking.
UMI captures rapid move-
ments with absolute scale by leveraging GoPro’s built-in
capability to record IMU data (accelerometer and gyroscope)
into standard mp4 video files [ 18 ]. By jointly optimizing visual
tracking and inertial pose constraints, our Inertial-monocular
SLAM system based on ORB-SLAM3 [ 7 ] maintains tracking
for a short period of time even if visual tracking fails due to
motion blur or a lack of visual features (e.g. looking down
at a table). This allows UMI to capture and deploy highly

# Page 5
Diffusion Policy

Desired EE pose & grp width

t input

Image
Obs Latency

Gripper
O bs Ltc

Arm
Obs
Ltc

Signal received until t input

Synchronization

Desired Poses

Actual Poses

Action Commands

T

Arm Execution Latency=100ms

Gripper Execution Latency=120ms

(a) Observation Latency Compensation
(b) Policy Interface
(c) Execution Latency Compensation

T

Synchronized Observations

t output

10 Hz

30 Hz

125 Hz

t obs

t act

Interpolation over raw signal

Fig. 5: UMI Policy Interface Design. (b) UMI policy takes in a sequence of synchronized observations (RGB image, relative EE pose,
and gripper width) and outputs a sequence of desired relative EE pose and gripper width as action. (a) We synchronize different observation
streams with physically measured latencies. (c) We send action commands ahead of time to compensate for robots’ execution latency.

dynamic actions such as tossing (shown in Fig 7 ). In addition,
the joint visual-inertial optimization allows direct recovery of
real metric scale, important for action precision and inter-
gripper pose proprioception PD2.3 : a critical ingredient to
enable bimanual policy.
HD5. Continuous gripper control.
In contrast to the
binary open-close action used in prior works [ 41 , 44 , 54 ], we
found commanding gripper width continuously significantly
expands the range of tasks doable by parallel-jaw grippers.
For example, the tossing task (Fig. 7 ) requires precise timing
for releasing objects. Since objects have different widths,
binary gripper actions will be unlikely to meet the precision
requirement. On UMI gripper, finger width is continuously
tracked via fiducial markers [ 16 ] (Fig. 2 left). Using series-
elastic end effectors principle [ 42 ], UMI can implicitly record
and control grasp forces by regulating the deformation of soft
fingers through continuous gripper width control.
HD6. Kinematic-based data filtering. While the data col-
lection process is robot-agnostic, we apply simple kinematic-
based data filtering to select valid trajectories for different
robot embodiments. Concretely, when the robot’s base lo-
cation and kinematics are known, the absolute end-effector
pose recovered by SLAM allows kinematics and dynamics
feasibility filtering on the demonstration data. Training on
the filtered dataset ensures policies comply with embodiment-
specific kinematic constraints.
Putting everything together. The UMI gripper weighs
780g, with an external dimension of L 310 mm × W 175 mm ×
H 210 mm and finger stroke of 80 mm . The 3D printed gripper
has a BoM cost of $73 , while the GoPro camera and acces-
sories total $298 . As shown in Fig. 2 , we can equip any robot
arms with a compatible gripper and camera setup.

B. Policy Interface Design

With the collected demonstration data, we can train a
visuomotor policy that takes in a sequence of synchronized
observations (RGB images, 6 degrees-of-freedom end-effector
pose, and gripper width) and produces a sequence of actions
(end-effector pose and gripper width) as shown in Fig. 5

(b). In this paper, we use Diffusion Policy [ 9 ] for all of our
experiments, while other frameworks such as ACT [ 53 ] could
potentially serve as a drop-in replacement.
An important goal of UMI’s policy interface design is
to ensure the interface is agnostic to underlying robotic
hardware platforms such that the resulting policy, trained
on one data source (i.e., hand-held gripper), could be directly
deployed to different robot platforms. To do so, we aim to
address the following two key challenges:

• Hardware-specific latency. The latency of various hard-
ware (streaming camera, robot controller, industrial grip-
per) is highly variable across system deployments, rang-
ing from single-digit to hundreds of milliseconds. In con-
trast, all information streams captured by UMI grippers
have zero latency with respect to the image observation,
thanks to GoPro’s synchronized video, IMU measure-
ments and the vision-based gripper width estimation.

• Embodiment-specific proprioception. Commonly used
proprioception observations such as joint angles and EE
pose are only well-defined with respect to a specific
robot arm and robot base placement. In contrast, UMI
needs to collect data across diverse environments and be
generalizable to multiple robot embodiments.

In the following sections, we will describe three policy inter-
face designs that address these challenges.
PD1. Inference-time latency matching.
While UMI’s
policy interface assumes synchronized observation streams
and immediate action execution, physical robot systems do
not conform to this assumption. If not carefully handled,
the timing mismatch between training and testing can cause
large performance drops on dynamic manipulation tasks that
require rapid movement and precise hand-eye coordination,
demonstrated in Sec V-B . In this paper, we separately handle
timing discrepancies on the observation and action sides:
PD1.1) Observation latency matching. On real robotic
systems, different observation streams (RGB image, EE pose,
gripper width) are captured by distributed micro-controllers,
resulting in different observation latency.

# Page 6
For each observation stream, we individually measure their
latency (details see § A1 - A3 ). At inference time, we align all
observations with respect to the stream with the highest latency
(usually the camera). Specifically, we first temporally down-
sample the RGB camera observations to the desired frequency
(often 10-20Hz), and then use the capture timestamp of each
image t obs to linearly interpolate gripper and robot propriocep-
tion streams. In bimanual systems, we soft-synchronize two
cameras by finding the nearest neighbor frames, which can be
off by a maximum of
1
60 seconds. The result is a sequence of
synchronized observations that conform to UMI policy, shown
in Fig. 5 (a).
PD1.2) Action latency matching. UMI policy assumes the
output as a sequence of synchronized EE poses and gripper
widths. However, in practice, robot arms and grippers can only
track the desired pose sequence up to an execution latency ,
that varies across different robot hardware. To make sure the
robots and grippers reach the desired pose at the desired time
(given by the policy), we need to send commands ahead of
time to compensate for execution latency, as shown in Fig. 5
(c). See § A4 for execution latency calibration details.
During execution, the UMI policy predicts the action se-
quence starting at the last step of observation t obs . The first few
actions predicted are immediately outdated due to observation
latency t input − t obs , policy inference latency t output − t input and
execution latency t act − t output . We simply discard the outdated
actions and only execute actions with the desired timestamp
after t act for each hardware.
PD2. Relative end-effector pose. End-effector (EE) pose
is central to both UMI’s observation and action space. To avoid
dependence on embodiment/deployment-specific coordinates,
we represent all EE poses relative to gripper’s current EE pose.
PD2.1) Relative EE trajectory as action representation.
Prior works have shown the significant impact of action
space selection on task performance [ 9 ], with experimental ev-
idence favoring absolute positional actions over delta actions.
However, we found that a relative trajectory representation,
defined for an action sequence starting at t 0 as a sequence of
SE ( 3 ) transforms denoting the desired pose at t relative to the
initial EE pose at t 0 , allows the system to be more robust
against tracking errors during data collection and camera
displacements.
PD2.2) Relative EE trajectory as proprioception. Simi-
larly, we represent the proprioception of history EE poses as
a relative trajectory. When observation horizon is set to 2, this
representation effectively provides velocity information to the
policy. Combined with our wrist-mounted camera observation
space, relative trajectory allows our system to be calibration-
free . Moving the robot base during execution will not affect
task performance (Fig. 10 (a)), as long as the objects are still
within reach range, making the UMI framework applicable to
mobile manipulators as well.
PD2.3) Relative inter-gripper proprioception.
When
using UMI in a bimanual setup, we found that providing the
policy with the relative pose between the two grippers to be
critical for bimanual coordination and task success, as shown

Time

8
7
6
5
4
3
2
1
0

Pose

Absolute
Delta
Relative

Inference
start at t=0

Inference
start at t=4

Fig. 6: Relative Trajectory as Action Representation . Relative
trajectory , used by UMI, is a sequence of end-effector (EE) poses
relative to the same current EE pose for each inference step.
In contrast, Delta action represents each action step relative to its
immediate previous action, therefore accumulates error. Absolute
action requires a global coordinate frame for all actions, which is
difficult to define for in-the-wild data collection.

in Sec. V-C . The effect of inter-gripper proprioception is par-
ticularly large when the visual overlap between two cameras is
small. The inter-gripper proprioception is enabled by our map-
then-localize data collection scheme that constructs a scene-
level global coordinate system HD4 . For each new scene, we
first collect a video that builds a map for the scene. Then,
all demonstrations collected in this scene are relocalized to
the same map, therefore sharing the same coordinate system.
Despite the videos from each gripper being relocalized sep-
arately, the relative pose between two grippers at each time
step can be calculated using their shared coordinates.

IV. E VALUATIONS

In our experiment, we aim to evaluate the UMI framework’s
effectiveness for deployable policy learning in three aspects:

• Capability: How well can we transfer UMI demonstra-
tions to effective robot policy? Especially for complex,
dynamic, bimanual, and long-horizon manipulation skills.

• Generalization: Will data collected in the wild within
diverse environments help the policy to generalize to
unseen environments and objects?

• Data collection efficiency: How fast can we collect
manipulation data with UMI? What’s the accuracy of the
SLAM system?
To access capability and generalization, we evaluate UMI
on 4 real-world robotic tasks across both narrow domain and
in-the-wild environments, shown in Fig. 7 . To measure data
collection efficiency, we compare the UMI gripper with human
hand demonstration and a typical teleop interface. See § B for
detailed data collection protocol.

V. C APABILITY E XPERIMENTS

We study UMI’s ability to capture and transfer single-hand,
bimanual, dynamic, and long-horizon manipulation skills with
four tasks. For capability experiments, all tasks are evaluated
in the same environment as data collection but with random-
ized robot and object initial states. To ensure a fair comparison,

# Page 7
…

Toss lego to rectangle bin

Grasp lego block
Toss orange to round bin
Init

Init
Reorient handle to the right
Grasp espresso cup
Final: Place on the saucer
(G1) Other scene
(G2) Other embodiment

Final

Init
Fold left sleeve
Fold right sleeve
Fold in half
Rotate 90 degrees
Final: fold in half

Init
Turn on faucet
Pick up dirty dish
Grasp sponge
Remove ketchup
Place clean dish on ra ck

Turn off faucet & return sponge

Task 2. Dynamic Tossing

Task 1. Cup Arrangement

Task 3. Bimanual Cloth Folding

Task 4. Dish Washing

Final:

Fig. 7: Policy Rollouts. We test UMI on a variety of challenging real-world tasks. Cup arrangement tests UMI’s ability to learn both
prehensile and non-prehensile actions, and to capture multi-modal action distribution (clockwise and counter-clockwise rotation). This task
is evaluated in both narrow-domain and unseen environments as well as two robot embodiments. Dynamic tossing tests UMI’s ability to
capture and transfer rapid human motions as well as precise hand-eye coordination. Bimanual cloth folding tests UMI’s ability to synchronize
two-arm coordination. Dish washing tests UMI’s ability to handle long-horizon tasks that involve multiple rigid, deformable, and articulated
objects. Please check videos on our website for more details.

we use exactly the same initial state across all methods for
both the robot and objects, by manually aligning the scene
against pre-recorded images. See § C for detailed evaluation
protocol and videos for all experiments.
A. Cup Arrangement

Task Place an espresso cup on the saucer with its handle
facing to the left of the robot, Fig. 7 . We defined task success
as when the cup is placed upright on the saucer with its handle
within ± 15° to the left.
Capability (what makes the task diff i cult?)
This
task
tests the system’s ability to learn both prehensile (pick and
place) and non-prehensile actions (i.e., pushing to reorientate
the cup). When the handle faces straight away from the
robot, the two equally valid solutions: rotation clockwise and
counter-clockwise form a multi-modal action distribution. This
task also tests UMI’s ability to sense relative depth through
monocular camera observation and side mirrors.
Performance The training dataset contains 305 episodes
collected by 2 demonstrators, evaluation includes 20 test cases,
with the testing initial state distribution shown in Fig. 8 (a).
UMI can complete the task 20/20. The next paragraphs will
discuss our ablation studies around our key design decisions.

Cross-robot generalization: To demonstrate UMI’s cross-
embodiment generality, we also deployed the same policy
checkpoint on a Franka Emika FR2 robot, shown in Fig. 1
and Fig. 8 . This experiment achieves 18 / 20 = 90% success
rate, with the 2 failure cases being joint limit violations, which
could have been avoided if we had mounted the FR2 robot at
a different location.
No Fisheye lens [HD2] :
To ablate the importance of
having a wide field-of-view (FoV) Fisheye lens, we post-
processed the dataset by rectifying and cropping each image
to a square with 69° horizontal and vertical FoV. This is a
generous analogy of RealSense D415 (69° HFoV, 42° VFoV)
and iPhone wide camera (69° HFoV, 51° VFoV). This baseline
only achieves 11 / 20 = 55% success rate. Beyond the expected
failure mode where the cup is outside of camera view, we
found this baseline policy to perform surprisingly poor even
if the object is visible, with often jittery motions. We suspect
that during training, the poor object visibility forced the policy
to be unnecessarily multimodal.
Alternative action spaces [PD2] : As alternatives to our
relative trajectory as action representation, we also consider
absolute and delta action spaces as illustrated in Fig 6 . Since

# Page 8
(a) All Initial States
(b) Typical Failure Cases of Baselines

No Fisheye

Absolute Action

Exceed
Joint Li mit

Ours (Franka)

Absolute Action

Delta Action

Ours (UR5)

Ours (Franka)

No Inter-gripper Pose

Ours

No Rel
Pose

Right
Sleeve
Left
Sleeve
Fold
Bottom Rotate
Fold
Final
Overall

No Rel
Pose
0.90
0.70
0.35
0.40
0.30
0.30

Ours
0.90
0.90
0.90
1.00
0.80
0.70

Substep Success Rate

No Latency Matching

Ours

No Latency
Matching

Apple Baseball Orange
Green
Block
Red
Block
Orange
Block
Overall

No
Latency
Matching
0.70
0.50
0.70
0.30
0.65
0.60
0.575

Ours
0.80
0.85
0.85
1.00
0.90
0.85
0.875

Per-object Success Rate

Cup Rearrangement
Dynamic Tossing
Bimanual Folding
Dish Washing

ResNet
ViT

Correct

Miss

Poor Coordination

Turn
Faucet
Grasp
Dish
Grasp
Sponge
Wash
Dish
Place
Dish
Overall

ResNet
0.50
0.20
0.90
0.00
0.00
0.00

Ours
(ViT)
1.00
0.90
0.95
0.75
0.75
0.70

ResNet

Ours

(ViT)

Substep Success Rate

Elbow Joint Velocity

w/ Latency
Matching
w/o Latency
Matching

Time

Jitter

(c) Quantitative Comparison

Saucer
not in view
X

0.55

0.9
0.85

0.25

0.8

1.0
0.9

0.575

0.875

0.3

0.7

0.0

0.7

No Fisheye

No Mirror Swap

No Mirror

Fig. 8: Narrow-domain Evaluation Results. (a) Initial states for all evaluation episodes overlayed together. For each task, all methods start
with the same set of initial states, matched manually with reference images. (b) Typical failure mode of the baseline/ablation policy. The
red arrow indicates failure behavior, green arrow indicates desired behavior. (c) Success rate over 20 evaluation episodes, best performance
for each column are bolded. Please check our website for more comparison videos.

the SLAM system outputs pose relative to the first frame of the
mapping video (details in § D ), we can only calculate relative
and delta actions directly using SLAM output. To compute
absolute actions in the robot base frame, we calibrate both
SLAM coordinates and the robot with respect to the same
fiducial markers [ 16 ] placed on the table.
The delta action baseline achieves 16 / 20 = 80% success
rate. The absolute action baseline performs surprisingly poorly
with only 5 / 20 = 25% success rate, demonstrating a noticeable
bias in action selection, likely due to inaccurate calibration
between the SLAM and robot base coordinate frames (Fig. 8
(b)). While theoretically the performance of this baseline could
approach that of relative trajectory with better calibration, this
experiment underscores the difficulty of obtaining action data
with absolute coordinates, even in controlled lab settings.
Effect of side mirrors [HD3] :
To our surprise, directly
providing mirror images decreases the performance from
18 / 20 = 90% (no mirror) to 17 / 20 = 85%. To fully take
advantage of side mirrors, we need to digitally reflect the
content inside mirrors and swap left and right mirror images,
which achieves a 20 / 20 = 100% success rate. We hypothesize

that without digital reflection, the opposite motions observed in
the main and mirrored images might confuse vision encoders,
especially those with translational equivariance.

B. Dynamic Tossing

Task The robot is tasked to sort 6 objects from the YCB
object set [ 5 ] randomly placed on a table by tossing them
to the corresponding bin. The 3 spherical objects (baseball,
orange, apple) should be tossed into the round bin, while the
3 Lego Duplo pieces go into the rectangular bin (Fig. 7 ). The
bins are placed beyond the robot’s kinematic reach range to
highlight the necessity of dynamic action for this task.
Capability: The dynamic tossing task demonstrates UMI’s
ability to capture and transfer fluid and rapid human motions,
precise hand-eye coordination (between RGB and propriocep-
tion) and timing alignment (between robot and gripper).
Performance: We collected 280 demonstration episodes for
this task, with mixed multi and single-object picking and
tossing. Our policy (with inference time latency matching)
achieves 105 / 120 = 87 . 5% success rate, counted by the num-
ber of objects successfully tossed to their corresponding bin.
