# Page 1
T E C H N O L O G I E S
D I G I T A L

AoE: Always-on Egocentric Human Video Collection for
Embodied AI

Bowen Yang ∗ 1 , Zishuo Li ∗ 1 , Yang Sun ∗ 1 , Changtao Miao ∗† 1 , Yifan Yang 2 , Man Luo 1 , Xiaotong
Yan 1 , Feng Jiang 1 , Jinchuan Shi 3 , Yankai Fu 4 , Ning Chen 4 , Junkai Zhao 5 , Pengwei Wang 5 ,
Guocai Yao 5 , Shanghang Zhang 4,5 , Hao Chen 3 , Zhe Li 1 , Kai Zhu ‡ 1

1 Ant Digital Technologies, Ant Group
2 Institute of Automation, Chinese Academy of Sciences

3 Zhejiang University
4 Peking University
5 Beijing Academy of Artificial Intelligence

‡ kaile.zk@antgroup.com

Abstract

Embodied foundation models require large-scale, high-quality real-world interaction
data for pre-training and scaling. However, existing data collection methods suffer
from high infrastructure costs, complex hardware dependencies, and limited inter-
action scope, making scalable expansion challenging. In fact, humans themselves
are ideal physically embodied agents. Therefore, obtaining egocentric real-world
interaction data from globally distributed “human agents” offers advantages of low
cost and sustainability. To this end, we propose the Always-on Egocentric (AoE)
data collection system, which aims to simplify hardware dependencies by leveraging
humans themselves and their smartphones, enabling low-cost, highly efficient, and
scene-agnostic real-world interaction data collection to address the challenge of data
scarcity. Specifically, we first employ an ergonomic neck-mounted smartphone holder
to enable low-barrier, large-scale egocentric data collection through a cloud-edge
collaborative architecture. Second, we develop a cross-platform mobile APP that
leverages on-device compute for real-time processing, while the cloud hosts auto-
mated labeling and filtering pipelines that transform raw videos into high-quality
training data. Finally, the AoE system supports distributed Ego video data collection
by anyone , anytime , and anywhere . We evaluate AoE on data preprocessing quality
and downstream tasks, demonstrating that high-quality egocentric data significantly
boosts real-world generalization.

1. Introduction

Recent strides in embodied foundation models, exemplified by Gen-0 [ 1 ], exhibit “scaling laws”
mirroring those observed in Large Language Models (LLMs): model generalization capabilities
and performance frontiers expand commensurate with the scale of real-world interaction data
[ 1 , 2 ]. This trend underscores that data-driven paradigms have become the cornerstone for
advancing embodied intelligence. However, in stark contrast to the ubiquitous textual corpora

∗ Equal Contribution.
† Corresponding Author.
‡ Project Leader.

arXiv:2602.23893v2  [cs.CV]  2 Mar 2026

# Page 2
Wearable

Setup

On-Edge

Hand
Detection

Action
Recognition

Video
Recording

Data
Uploading

Camera
Trajectory

MANO Mesh
Reconstruction
2D Hand
Segmentation

Quality
Filtering

On-Cloud

Global
Network

Edge-Cloud
Synchronization

Figure 1: Overview of the AoE system. The system leverages neck-mounted smartphones for
ubiquitous egocentric capture ( Left ). Our edge-cloud collaborative pipeline ( Middle ) efficiently
distributes computation: on-device models handle real-time detection and selective uploading,
while cloud servers execute heavy-duty auto-labeling and quality filtering. This design mini-
mizes hardware dependencies, enabling scalable, high-quality data collection in the wild ( Right ).

Table 1: Comparison of data collection paradigms for dexterous manipulation. The AoE retains
the high scalability and low cost ( < $ 20) compares with Teleoperation, UMIs [ 3 , 4 ], Wearables [ 5 ],
Passive Videos [6]. Detailed rating criteria are provided in the Appendix 7.1.

Teleoperation
UMIs
Wearables
Passive Videos
AoE (Ours)

Cost (per user)
> $50k
$300–800
> $2k
Free
< $20

Non-Intrusiveness
★ ✩✩✩✩
★★★ ✩✩
★ ✩✩✩✩
★★★★★
★★★★ ✩
Scalability
★ ✩✩✩✩
★★★ ✩✩
★★ ✩✩✩
★★★★★
★★★★★
Deployment Ease
★ ✩✩✩✩
★★ ✩✩✩
★★ ✩✩✩
★★★★★
★★★★★
Data Quality
★★★★★
★★★★ ✩
★★★★ ✩
★ ✩✩✩✩
★★★★ ✩

available on the web, physical interaction data is notoriously difficult to curate, suffering from
prohibitive acquisition costs, limited diversity, and significant inherent noise. Consequently,
the scarcity of large-scale, high-quality real-world interaction data remains a critical bottleneck
impeding the field of embodied intelligence.

Early teleoperation-based approaches [ 7 , 8 ] capture high-fidelity data but are constrained
by expensive hardware and lab environments, impeding large-scale expansion. To enable
in-the-wild collection, researchers have developed portable solutions, including handheld
interfaces [ 3 , 9 , 4 ] and wearable AR/VR systems [ 10 , 5 , 11 , 12 ]. However, handheld devices
require active manipulation that disrupts natural behavior, while bulky AR/VR setups remain
intrusive, rendering them unsuitable for continuous daily use. Alternatively, learning directly
from egocentric videos [ 2 , 13 ] offers a scalable avenue for pre-training Vision-Language-Action
(VLA) models. Nevertheless, existing datasets [ 6 , 14 ] typically lack the fine-grained interaction
dynamics essential for policy learning. Consequently, extracting high-quality segments from
such noisy streams incurs substantial curation costs, limiting their practical utility for robot
training, as shown in Table 1.

Humans naturally function as ideal physically embodied agents, offering a massive popula-
tion continuously operating within the environments we seek to capture [ 13 , 15 ]. This mirrors
data acquisition paradigms in autonomous driving, where fleets aggregate environmental sig-
nals and behavioral data globally [ 16 , 17 ]. Inspired by this, we propose harvesting egocentric

2

# Page 3
interaction data from these distributed "human agents" in unstructured, everyday scenarios.
Crucially, ubiquitous smartphones provide a low-cost, deployable foundation for visual sensing
and edge computing. Consequently, organically coupling human dexterity with mobile comput-
ing to establish a human-centric, distributed collection architecture constitutes a scalable and
sustainable technical pathway.

To this end, we introduce the Always-on Egocentric (AoE) system , a framework leveraging
ubiquitous smartphones to enable continuous, low-cost, and scalable egocentric video acquisi-
tion, as illustrated in Figure 1. Adhering to a human-centered design, AoE utilizes an ergonomic,
magnetic neck mount that stabilizes the device at the chest. This setup allows the rear camera
to capture physical interactions in natural environments while minimizing disruption to daily
activities, thereby preserving behavioral naturalness and data fidelity. Structurally, the frame-
work constitutes a loosely coupled distributed architecture linking global cloud servers with
edge devices, significantly lowering technical barriers for large-scale contribution. We develop a
cross-platform mobile APP (Android/iOS) that exploits smartphone capabilities for real-time
calibration and pre-processing, eliminating reliance on specialized hardware. Finally, we estab-
lish a robust cloud-based pipeline for automated annotation and filtering. Upon user-authorized
upload, this edge-cloud system performs computation-intensive tasks to transform raw, noisy
video streams into high-quality training segments characterized by high information density
and clear semantics.

We conduct a comprehensive evaluation of the AoE data acquisition system across three key
dimensions: data preprocessing efficacy, real-to-sim reconstruction fidelity, and downstream
policy fine-tuning. Experimental results demonstrate that our collected egocentric interaction
data exhibit high-quality annotations and robust reconstruction capabilities. Crucially, lever-
aging this data significantly enhances the task success rates of embodied models in complex
real-world environments.

To summarize, our key contributions are the following:

• We propose the Always-on Egocentric (AoE) system, a scalable, low-cost framework
utilizing ubiquitous smartphones and ergonomic hardware to enable continuous, non-
intrusive data acquisition in the wild.
• We design a distributed edge-cloud architecture featuring an automated pipeline that
efficiently transforms raw, noisy video streams into high-density, semantically annotated
training segments for embodied learning.
• We demonstrate that AoE-collected data significantly enhances real-to-sim reconstruction
fidelity and improves the task success rates of downstream robotic manipulation policies .

2. Related Works

2.1. Data Collection with In-The-Wild Equipment

Large-scale "in-the-wild" data is vital for embodied intelligence to transcend hardware and
lab constraints [ 18 , 1 , 7 ]. We categorize existing methods by acquisition paradigm: (1) Hand-
held Gripper-based Interfaces: While UMI [ 3 ] and successors (e.g., FastUMI [ 9 ], DexUMI [ 4 ])
lower costs via portable grippers, they remain "active" devices requiring deliberate hardware
manipulation, thereby restricting natural interaction coverage. (2) Wearable AR/VR Systems:
High-precision setups like DexCap [ 10 ], HumanoidExo [ 5 ], AirExo [ 11 ], and VR-based hubs (e.g.,
exUMI [ 19 ], ActiveUMI [ 12 ]) capture complex poses. However, their significant bulk ( > 500 𝑔 )
and power dependencies render them too intrusive for continuous, "always-on" daily use. (3)

3

# Page 4
Hardware Configuration: Ergonomic long-term wearable setup
APP UI Interface: Ego-centric Data Collection

Hand Motion

Detection

Object & Scene

Recognition
Hand
Detection

On-Device Algorithm: Valueable Data Auto-Record

Upload to
Cloud Storage

REC

Auto-Record: ON
00: 15: 42

Ergonomic
Neck Mount

Browse Data

User Checking and

Authorization

algorithm passes

recording start

algorithm fails
recording paused

Always-on Seamless

Data Recording

Download and

Register APP

Wearable Setup
Mounting Detail

Mechanical Clamp

MagSafe-Compatible Mount

Magnetic Mount

Mechanical/Magnetical

Attached Smartphone

… …

Figure 2: Overview of Hardware & Mobile Application. The AOE hardware supports various
ergonomic mounts (Mechanical, MagSafe, Magnetic) with stabilizing straps for robust, all-day
egocentric recording (Left) . The user-friendly UI Interface for users to manage recordings (Up
Right) . On-device intelligence selectively records high-value manipulation data (Bottom Right) .
Secure pipeline that synchronizes user-authorized data to the cloud (Right) .

Passive Egocentric Videos: Existing datasets (e.g., Ego4D [ 6 ] and Epic-Kitchens [ 14 ]) prioritize
recognition over manipulation, suffering from motion blur and occlusion. Consequently, ex-
tracting training-grade segments incurs prohibitive curation costs. Therefore, we introduce AoE
(Always-On Egocentric), a system leveraging ubiquitous smartphones for passive, scalable, and
high-quality data acquisition with minimal user burden and near-zero marginal cost.

2.2. Manipulation Policies Learning from Human Videos

Leveraging human videos addresses robot data scarcity. Early efforts focused on representation
pretraining [ 20 , 21 , 22 ] or extracting proxy signals like trajectories [ 23 , 24 ] and plans [ 25 ]. Recent
policy learning approaches span three paradigms: (1) bridging embodiment gaps via retargeting
on controlled data [ 26 , 27 , 28 ]; (2) utilizing latent action codes to bypass explicit labeling [ 29 , 30 ,
31 ]; and (3) recovering 3D hand motion to construct shared action spaces [ 32 , 13 , 2 , 33 ]. Crucially,
while recent studies [ 34 , 35 ] validate in-the-wild pretraining, they underscore the bottleneck of
curating noisy corpora. Our AoE system resolves this scaling challenge by enabling low-cost,
automated acquisition and high-quality filtering for robust policy learning.

3. Always-On Egocentric Data Collection System

The proposed Always-on Egocentric (AOE) system is a distributed edge-cloud framework
designed for the scalable, low-cost acquisition of egocentric robot learning data. AoE seamlessly
captures in-the-wild activities without disrupting user workflows, automatically filtering and
annotating high-quality samples. The system prioritizes three objectives: (1) Effortless Collec-
tion: Leveraging ubiquitous smartphones and a minimalistic neck mount to democratize data
acquisition across diverse environments. (2) High-Quality Annotation: Employing an auto-
mated pipeline to yield training-ready data with negligible manual intervention. (3) Scalable
Architecture: Utilizing edge-cloud collaboration to support high-concurrency streaming and
distributed data production. Subsequent sections detail our hardware design, mobile software,

4

# Page 5
annotation pipeline, and system implementation.

3.1. Hardware & Mobile Application

Hardware Configuration and Accessibility. While UMI [ 3 ] and exoskeletons [ 5 ] successfully
decouple collection from specific embodiments, their reliance on specialized, costly hardware
hinders "in-the-wild" scalability. We instead propose a democratized ecosystem built entirely on
consumer-grade devices. As shown in Figure 2, our ergonomic neck mounts support mechanical ,
MagSafe , and magnetic interfaces. An auxiliary stabilizing strap minimizes camera shake
during dynamic manipulation. By mounting smartphones at the sternum and leveraging
ultra-wide lenses , we capture high-resolution egocentric views that approximate human vision
without obstructing natural workflows or comfort. Table 1 highlights our decisive cost efficiency:
unlike UMI or wearables ($300–$2000), our assembly costs under $20 . This near-zero marginal
cost, combined with smartphone ubiquity, facilitates highly scalable data acquisition across
diverse real-world scenarios.

Always-On Collection App. We developed a cross-platform application (Android/iOS)
featuring an authorized "Always-On" mode for efficient, privacy-preserving data capture. As
shown in Figure 2), we leverage lightweight on-device vision models, which includes hand
detection, motion tracking, and open-set recognition, the system autonomously triggers record-
ing only during relevant hand-object interactions. This selective strategy eliminates manual
intervention, enabling seamless "24h" collection while minimizing storage and downstream
filtering overhead * . Additionally, the app captures camera intrinsics and sensor metadata to
facilitate automated cloud labeling. The user interface and workflow are detailed in Figure 9
and Appendix 7.2.

Privacy and Data Security. Adopting a privacy-first architecture, our system performs all
model inference and raw data storage locally. To safeguard subjects, an automated pipeline de-
identifies sensitive attributes (e.g., faces, text) prior to upload, while a continuous, non-mutable
audio signal alerts bystanders during recording. Finally, complete data sovereignty is ensured
by requiring manual user review and authorization before any cloud transmission.

3.2. Automated Annotation and Quality Filtering Pipeline

To address perceptual aliasing and limited VLA compliance in long-horizon tasks [ 36 ], AoE
generates cost-efficient atomic-action annotations—comprising end-effector trajectories, camera
trajectories and semantic labels via a 6-stage pipeline:

Camera Calibration : We retrieve factory-calibrated intrinsics via Camera2 [ 37 ], eliminating
device-specific calibration overhead while maintaining sub-pixel stability across conditions.

Atomic Action Segmentation : Qwen3-VL-235B-A22B [ 38 ] segments videos into semantic
atomic clips (Figure3a) avoiding heuristic biases. Human verification corrects potential VLM
hallucinations in boundaries and labels.

Camera Trajectory Estimation : We use Lingbot-Depth [ 39 ] to refine RGBD inputs with supe-
rior stability, while temporal consistency enhances RGB-only metric precision [ 40 ]. Trajectories
are estimated via MegaSAM [41] using robust kernels tuned for motion blur (Figure 3b).

Hand Reconstruction : HaWoR [ 42 ] performs 3D joint recovery, rescaled via depth estimates.

* This mode is only enabled upon explicit user authorization, ensuring the user maintains full control over
the collection periods.

5

# Page 6
机械臂

(b) Scene reconstruction

Camera
trajectory

Camera Intrinsic params. (K)

Camera Trajectory

(c) Hand reconstruction

Det & Seg Tracking
Mano & key points Est.

Left
hand

Right
hand

Body

URDF

Background Replacement

Robot Inpainting

(d) Data Augmentation

Body Erasing

Seg

Overlay

World
Model

Text prompt

Ego Raw Video

Action
Slice & Instruct

Undistort

timeline

Ego Undistorted Video with action label

K&D from phone camera

"atomic_actions": [

{

"start_time": 0.0
"end_time":  5.6
"verb": "hold",
"object": ”carrot",
"description": ”right hand holds carrot",
"bbox": [220, 100, 300, 180]
"confidence": 1.00
…

"atomic_actions": [

{

"start_time": 5.6,
"end_time":  11.0
"verb": "hold",
"object": ”egg",
"description": "left hand holds egg",
"bbox": [80, 120, 140, 180],
"confidence": 0.88
…

atomic_actions": [

{

"start_time": 11.0,
"end_time": 19.3,
"verb": "hold",
"object": "pan",
"description": "left hand holds pan handle",
"bbox": [150, 180, 200, 250],
"confidence": 0.94
}…

Video
clip1

Video
clip2

Video
clip3

(a)

Video

Video

Hands

Figure 3: Overview of the Automatic Annotation and Augmentation Pipeline. (a) Undistort
videos and segment videos into atomic clips. (b) Dense depth maps yield camera trajectories
and scene reconstruction. (c) Hand poses are generated and transformed to world coordinates.
(d) Augmentation employs generative background replacement and simulation-based robot
inpainting.

MANO [ 43 ] outputs are transformed to world coordinates via SLAM poses, employing sliding-
window optimization to enforce kinematic consistency (Figure 3 c).

Data Augmentation : Hand replacement follows Masquerade [ 44 ]: GAN-based removal,
6-DoF robot alignment, and photometric re-rendering. Background replacement employs video
diffusion on contact-free frames conditioned on depth and segmentation (Figure 3 d).

Quality Control : We automatically filter kinematic outliers ( > 3 𝜎 joint velocities) and high
reprojection errors ( > 5px). A 5% manual inspection informs adaptive pipeline adjustments (e.g.,
texture diversity, hand proximity), with failed samples categorized into a hard-negative pool for
re-annotation, effectively closing the data generation loop.

3.3. Distributed System Implementation

To facilitate large-scale robot learning from human demonstrations across geographically dis-
tributed devices, we propose a distributed edge-cloud collaborative architecture. This system
is explicitly designed to overcome three critical bottlenecks inherent in traditional centralized
data collection: high-latency data transfer, inflexible processing pipelines, and limited scalability.
As illustrated in Figure 4, our unified platform establishes a scalable infrastructure for the full
lifecycle of egocentric manipulation data by addressing these challenges through three specific
architectural innovations.

First, to mitigate high-latency transfer from dispersed collectors, we implement an Edge-
Cloud Collaborative Architecture . This component deploys proximity-based edge ingestion
nodes equipped with intelligent routing to minimize transfer latency, while simultaneously
maintaining centralized data consistency through asynchronous cross-region synchronization.

6

# Page 7
Scene Reconstruction Pipeline

Hand Reconstruction Pipeline

Cloud Environment

Task Management &

Scheduling

Cross-Region
Synchronization

Resource Scheduler & Orchestrator

Configurable Pipeline

Definition Engine

Flexible Customization

Preprocessing Pipeline

Data Processing Pipeline Pool

High-Concurrency & Dynamic Scaling

Images

Video

PointCloud

Multi-Modal
Object Storage

Output

Data
Products

Raw Data

Undistorted Data

Manual Modeling

Action Commands

Robot Ego Model

Edge Region 2

Mobile

Phone

Camera

Mount

Edge Node

Edge Region 1

Mobile

Phone

Camera

Mount

Edge Node

Figure 4: Distributed Edge-Cloud Architecture. Enabling low-latency edge-to-cloud synchro-
nization, the system utilizes a configurable, elastically scaled pipeline to generate multi-modal
data for robot policy learning.

Second, addressing the rigidity of pipelines that typically require weeks of re-engineering for
new algorithms, we develop a Customizable Processing Pipeline . By leveraging declarative
workflow orchestration, this design enables modular algorithm integration and hot-swappable
components, allowing for rapid updates without system-level modifications. Third, to resolve
the inefficiencies of fixed resource allocation, we introduce Elastic Scaling mechanisms. Built
on a cloud-native Kubernetes architecture, this module utilizes horizontal pod autoscaling and
intelligent GPU/CPU resource partitioning to dynamically provision compute resources in
response to real-time demand.

Collectively, these innovations yield significant performance improvements: data transfer
latency is reduced from 500ms+ to under 100ms; algorithm integration cycles are shortened
from weeks to days; and the system supports thousands of concurrent devices with minute-level
responses to workload spikes. This architecture effectively enables “anyone, anytime, anywhere”
data contribution, thereby advancing the scalability of robot manipulation policy learning. The
more details in Appendix 7.3.

4. Experiments

We empirically evaluate the AoE system across three core dimensions: (1) validating the pre-
cision of 3D hand pose and camera trajectory tracking; (2) assessing the fidelity of interaction
reconstruction for real-to-sim transfer; and (3) quantifying downstream utility via measurable
performance gains in real-world robotic manipulation tasks. The more experimental setups in
Appendix 7.4.

4.1. Precision of the AoE System

Experimental Setup.
We evaluate EgoFactory on four datasets: (1) EgoDex [ 45 ] test datasets
(about 7 hours); (2) Ego4D [ 6 ] test datasets (about 6.5 hours) with VITRA [ 35 ] annotations; (3) an
10 hours in-house collection (RealSense L515, Figure 5 a); and (4) a small paired AR-smartphone

7

# Page 8
Ours
EgoDex

(d) Hand Reconstruction Comparison
(e) Camera Trajectory
(f) Evaluation

(c) Hand Reconstruction

(a)Test Device 1

Ego4D
EgoDex
Source

3.7
7.4
PA-
MPJPE↓

8.9
11.9
MPJPE↓

0.93
0.90
AUC↑

Ours
(w/o
& w/
depth)

Ego

4D
Ego
Dex
Source

0.26
4.77
1.91
ATE↓

4.4
8.3
16.2
ATE-S↓

0.56
1.84
2.94
RPE-
Trans↓

0.02
0.07
0.08
RPE-
Rot↓

Table 1 ： Hands
Reconstruction Error. PA-
MPJPE and MPJPE are in mm,
AUC is reported as a value
between 0 and 1

Table 2 ： Camera Trajectory
Error. ATE, ATE-S and RPE-
Trans are in mm, RPE-Rot is
in deg.

(b)Test Device 2

Apple
Vision

Phone

Realsense

Figure 5: Comparison of data processing acquisition methods and accuracy. (a) Depth camera
acquisition configuration. (b) AR glasses + smartphone combined acquisition configuration. (c)
Hand modeling comparison using AR glasses + smartphone. (d) Hand modeling comparison
from EgoDex. (e) Camera trajectory comparison after rotation-translation alignment from
EgoDex. (f) Hand reconstruction accuracy and camera trajectory reconstruction accuracy.

set (Figure 5 b). Dataset annotation comparisons utilize raw sensor data.

Evaluation metrics include: (1) Calibration: Deviation from offline checkerboard meth-
ods [ 46 ]; (2) Hand Pose: PA-MPJPE (7-DoF), MPJPE, and AUC; and (3) Trajectory: ATE (7-DoF),
ATE-S (scale-free 6-DoF), and relative pose errors (RPE).

Quantitative Analysis of Calibration. Factory intrinsics (Camera2 API) deviate from offline
target-based calibration [ 46 ] by < 1% (mean 0.64%, std 0.21%). Radial distortion coefficients
remain stable at 10 − 3 magnitude, validating the reliability of factory parameters for spatial
processing without manual recalibration.

3D Hand Pose Estimation. Our pipeline achieves high accuracy across benchmarks (Table
1 of Figure 5 f). While larger errors on EgoDex stem primarily from dataset-inherent motion
blur and misalignment (Figure 5 d), validation against hardware-tracked AR glasses confirms
our model’s precision (Figure 5 c). High AUC scores ( > 0.90) further attest to robust keypoint
detection across diverse scenarios.

Camera Trajectory Estimation. Post 7-DoF alignment, the Absolute Trajectory Error (ATE)
remains < 5 mm across all datasets (Table 2 of Figure 5 f). Although texture-less backgrounds in
EgoDex degrade ATE-S (16.2 mm), our method maintains centimeter-level accuracy even with
monocular RGB input (Figure 5 e), demonstrating robust tracking independent of depth sensors.

4.2. Real-to-Sim Transferability

To assess the real-to-sim transferability of AoE data, we reconstruct simulation-ready digital
twins using the AGILE framework [ 47 ]. This process targets the extraction of dynamic hand-
object interactions (HOI) from monocular videos, prioritizing interaction fidelity over sparse

8
