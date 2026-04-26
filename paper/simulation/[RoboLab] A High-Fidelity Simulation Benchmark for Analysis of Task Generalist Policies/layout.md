# Page 1
RoboLab: A High-Fidelity Simulation Benchmark
for Analysis of Task Generalist Policies

Xuning Yang 1 , Rishit Dagli 2,4 , Alex Zook 1 , Hugo Hadfield 1 ,
Ankit Goyal 1 , Stan Birchfield 1 , Fabio Ramos 1,3 , and Jonathan Tremblay 1

1 NVIDIA, 2 University of Toronto, 3 The University of Sydney, 4 Work done during internship at NVIDIA

Fig. 1: Overview of RoboLab. RoboLab addresses the simulation-to-real gap by evaluating robotics policies on entirely held-out domains. By
featuring a streamlined generation pipeline for new scenes and tasks (top row), RoboLab enables rapid extensibility for testing generalization
capabilities. Our accompanying benchmark introduces visual, relational, and procedural testing axes, paired with robust metrics designed to
reveal how modern models perform when faced with novel, out-of-distribution challenges (bottom row).

Abstract —The pursuit of general-purpose robotics has yielded
impressive foundation models, yet simulation-based benchmarking
remains a bottleneck due to rapid performance saturation and
a lack of true generalization testing. Existing benchmarks often
exhibit significant domain overlap between training and evaluation,
trivializing success rates and obscuring insights into robustness.
We introduce RoboLab, a simulation benchmarking framework
designed to address these challenges. Concretely, our framework
is designed to answer two questions: (1) to what extent can we
understand the performance of a real-world policy by analyzing
its behavior in simulation, and (2) which external factors most
strongly affect that behavior under controlled perturbations. First,
RoboLab enables human-authored and LLM-enabled generation
of scenes and tasks in a robot- and policy-agnostic manner
within a physically realistic and photorealistic simulation. With
this, we propose the RoboLab-120 benchmark, consisting of 120
tasks categorized into three competency axes: visual, procedural,
relational competency, across three difficulty levels. Second, we
introduce a systematic analysis of real-world policies that quantify
both their performance and the sensitivity of their behavior to
controlled perturbations, indicating that high-fidelity simulation
can serve as a proxy for analyzing performance and its dependence
on external factors. Evaluation with RoboLab exposes significant

performance gap in current state-of-the-art models. By providing
granular metrics and a scalable toolset, RoboLab offers a scalable
framework for evaluating the true generalization capabilities of
task-generalist robotic policies.

I. I NTRODUCTION

The pursuit of generality has been a longstanding challenge
in modern robotics. Recent advances have produced impressive
generalist robot policies that demonstrate success in challenging
and novel tasks in the real-world. Despite this progress,
benchmarks for evaluating whether these policies are truly task-
general has been slow. Evaluating models in the real world
remains prohibitively expensive and logistically intractable,
motivating the rise of simulation-based benchmarks as an
appealing alternative.
Current robotics benchmarks [ 19 , 35 , 11 , 16 ] face several crit-
ical limitations: (1) a lack of high-fidelity simulation capable of
supporting real-world policies; (2) rapid performance saturation
on static task sets; and (3) a lack of granular analysis regarding
policy failure modes. For instance, popular benchmarks like

arXiv:2604.09860v2  [cs.RO]  14 Apr 2026

# Page 2
Fig. 2: Three approaches for robotic benchmarks. L EFT : To date, pure simulation based benchmarks have exhibited low visual quality,
creating a large sim2real transfer gap. M IDDLE : Real2sim benchmarks address this issue by using techniques to bring real-world visual
texture into simulation. However, these environments are extremely costly with reported per-scene generation time of ∼ 1hr [ 10 ]. R IGHT : Our
approach achieves a high degree of realism with low overhead.

LIBERO [ 19 ] often utilize nearly identical environments for
both training and evaluation. When policies are fine-tuned
on these simulation-specific demonstrations, the lack of a
meaningful domain gap trivializes the evaluation process and
obscures the model’s true generalization capabilities. Many
existing platforms have limited realism or are difficult to extend
due to using rigid architectures that make it cumbersome to
introduce new objects, tasks, or robots (Fig. 2).
To address these limitations, we present RoboLab (Fig. 1),
a simulation platform and benchmarking suite designed for
rigorous robotics evaluation. Unlike prior benchmarks that
rely on PDDL or rigid scene-graph definitions [ 19 ], RoboLab
introduces an easy-to-use interface that enables human-authored
and LLM-scaled scene and task generation. RoboLab enables
generation and validation of new scenes and tasks from natural
language prompts. This system enables the creation of over
800 diverse scenarios (see supplemental), providing a scalable
framework that mitigates benchmark saturation and ensures
long-term value.
RoboLab introduces novel task axes and robust metrics to
provide deeper diagnostic insights, paired with a streamlined
toolset for generation of scenes and tasks (see Fig. 1).. To pro-
vide a granular assessment of policy behavior in our proposed
benchmark, we evaluate three axes: Visual (perceptual attributes
like color and size), Procedural (action-oriented logic such as
stacking and reorientation), Relational (spatial and linguistic
logic like “and/or”) spanning across three difficulty levels
depending on task length and language nuance. Policy execution
on these tasks is then evaluated with metrics on graded task
completion, failure and error occurrences, and trajectory quality.
Finally, we highlight novel metrics for evaluation; including
sensitivity analysis to identify environmental factors that most
strongly influence policy performance, e.g. , camera placement.
We introduce the RoboLab-120 benchmark, comprising 120
tasks generated via our automated workflow and verified by
humans. These tasks span varying difficulties (65 simple,
38 moderate, 18 complex) and multiple competency axes
(44 relational, 91 visual, and 36 procedural). To prevent
overfitting to the simulation domain, we evaluate policies
trained exclusively on the real-world DROID [ 13 ] dataset. This
creates an environment that reflects “in the wild” conditions;
for instance, the state-of-the-art π 0 . 5 [ 9 ] achieves only a ∼ 30%
success rate on RoboLab-120, highlighting the benchmark’s

difficulty.
In summary, our contributions are:
1) RoboLab : A novel simulation platform designed for
evaluating modern robotics policies with a scalable, LLM-
based workflow capable of procedurally generating over
800 unique scenes and tasks using human-readable USD
and Python interfaces in IsaacLab[21].
2) RoboLab-120 Benchmark : Comprising 120 tasks eval-
uated across three distinct competency axes (visual,
procedural, relational) and supported by four new robust-
ness metrics. We also present five policies evaluated on
RoboLab-120.
3) Policy Analysis : We introduce a suite of analysis tools
that gives insight into the model performance beyond
binary success rates and broader understanding of policy
performance.

II. R ELATED W ORK

Simulation-Based Benchmarks.
Simulation provides a
scalable and reproducible environment for evaluating robot
manipulation policies. Widely used benchmarks such as
RLBench [ 11 ], MetaWorld [ 33 ], and robosuite [ 36 ], Man-
iSkill2 [ 7 ], CALVIN [ 20 ], LIBERO [ 19 ], and BEHAVIOR-
1K [ 15 ], offer standardized task suites for learning and evalua-
tion in simulation across pre-defined task families and object
configurations. However, in these settings, policies are typically
trained and evaluated in the same simulated environments,
which encourages overfitting to simulator-specific quirks, leads
to rapid benchmark saturation, and makes real-world general-
ization hard to assess. [ 35 ]. In our setting, policies are instead
trained on large-scale real-world data (e.g., DROID [ 13 ]), while
high-fidelity simulation is used only as a controlled evaluation
environment, so training and evaluation domains are decoupled
and measured performance more closely reflects robustness in
the real world.

Real-to-sim Evaluation.
Recent work have focused on
leveraging 3D reconstruction to build photorealistic simulation
scenes from real-world videos in order to achieve closer visual
alignment between simulation and real world photorealism
[ 16 , 12 , 10 , 37 ]. These works typically use Gaussian splatting,
3D segmentation, and multi-view inpainting, often operated
at a per-scene level, which entails costly optimization and

# Page 3
Fig. 3: Task progression of a few tasks, illustrating errors encountered during policy rollout. Top row: Although the task is successfully
completed, errors were encountered during execution: 1) The robot drops the milk jug too early, missing the bin. 2) the robot grasps an
orange (wrong object) and puts it in the bin. Mid row: An extraneous object was reoriented before the actual intended object. Final row:
Intended objects were attempted unsuccessfully, and the policy tended to two wrong other objects.

makes it slow to scale beyond a small number of environments
[10, 34, 28]. In contrast, our framework produces large-scale,
photorealistic scenes and tasks within minutes rather than hours,
while preserving sufficient geometric and visual fidelity for
policy evaluation, thereby making real-to-sim benchmarking
practical at the scale needed for modern generalist robot
policies.

III. R OBO L AB
Evaluating real-world, generalist robotics policies in simu-
lation remains a significant challenge. RoboLab is a bench-
marking framework that introduces three novel task axes and
three original metrics tailored for modern robotics systems.
RoboLab enables a multifaceted analysis of Vision-Language-
Action (VLA) models, providing deeper insights into their
scalability and task generalization.

A. RoboLab-120

Inspired by the Large Language Model (LLM) community’s
use of Visual Question and Answering (VQA) benchmarks, we
introduce RoboLab-120 Benchmark that focus on evaluating
specific competency axes spanning three difficulty levels. This

Fig. 4: Example of language instructions in RoboLab-120.

taxonomic decomposition enables fine-grained analysis of
policy capabilities by systematically assessing performance.
Figure 4 shows examples of these questions accompanied by
scene examples.
Visual Competency: Assesses recognition of color , semantics ,
and size , capturing the policy’s capability to link perceptual
attributes with higher-level reasoning.
Procedural Competency: Evaluates the ability to perform tasks
that involve action-oriented reasoning, including affordances ,
reorientation , or stacking .
Relational Competency: Tests understanding of language con-
junctions (e.g., ‘and’, ‘or’), counting , and spatial relationships,
measuring how effectively the policy interprets multi-object
instructions and scene structure.
Tasks from these competencies can span one of the following
difficulty levels: simple, medium, complex. These are deter-
mined as a function of two aspects: whether if the language was
straightforward in describing the task, as well as the number
of required reasoning steps for the task.

B. Metrics for Evaluation

We establish a comprehensive suite of evaluation metrics that
captures the full spectrum of policy performance characteristics.
While task success rate remains a fundamental metric, prior
work [ 14 ] has demonstrated that they fail to reveal nuanced
aspects of policy behavior and failure modes. Unlike approaches
relying on human judgment [ 12 ], we define a set of discrete
and continuous metrics to characterize policy performance. We
regroup these novel metrics as follow, failure cases scoring,
trajectory metrics, and sensitivity analysis.

Failure cases.
In addition to success rate , we compute

# Page 4
Fig. 5: Comparison of policy performance for bowl-in-bin manipulation. Rows represent distinct policies shown in chronological order (left
to right). Successful execution involves grasping the central red bowl and depositing it into the gray bin on the right. Unsuccessful attempts
are characterized by aimless arm trajectories and a lack of object interaction.

a normalized graded score Sc ( T ) =
1
|T |
P

τ ∈T Sc ( τ ) . For
example, for the instruction “pick the lemon and the lime,” the
subtasks τ “pick lemon” and “pick lime” includes steps such
as “grasp” and “drop”. The final task score is the normalized
subtask scores. Our benchmark automatically records instances
of events; including wrong object grasped, object dropped, and
gripper collisions. Fig. 3 demonstrates a successful episode;
however, the policy incorrectly grasped an extraneous object.
Such errors highlight potential biases in the policy not captured
by other metrics.

Trajectory Metrics. Trajectory quality metrics capture char-
acteristics of motion efficiency and optimality. We compute
the following: Spectral arc-length (SPARC) , which evaluates
motion smoothness [ 2 ] via the arc length of the normalized
Fourier magnitude spectrum of the velocity profile. Given a
speed profile v ( t ) of the end effector over time interval [0 , T ]

SPARC = −
Z ω c

0

v
u
u
t

 1

ω c

 2
+

d ˆ V ( ω )

dω

! 2

dω
(1)

where ˆ V ( ω ) = V ( ω ) /V (0) represents the normalized Fourier
magnitude spectrum. Smoother motions yield values closer to
zero, while jerkier trajectories produce more negative values.
We employ an adaptive cutoff frequency ω c = min(10 Hz , ω α ) ,
where ω α = max k ∈K ω k and K = { k | ˆ V ( ω k ) ≥ α } denotes
the set of frequency bins exceeding threshold α = 0 . 05 . This
adaptive strategy ensures that the smoothness evaluation focuses
on relevant frequency components. Lastly, trajectory optimality
is assessed through end effector speed v ( t ) , and path length
l = P N − 1
k =0 ∥ p k +1 − p k ∥ , where p k denotes the end-effector
position at timestep k . Shorter path lengths indicate more direct
trajectories and generally reflect superior motion quality.

C. Sensitivity Analysis

We present a Bayesian framework for evaluating policy
robustness across diverse environmental conditions using
Simulation-Based Inference (SBI). This analysis provides
insight into which scene parameters are most strongly linked
to success and failure outcomes by learning an approximate
posterior distribution over them given evaluation data. Let
θ = ( θ cont , θ disc ) denote the environment parameters comprising
of continuous variables (e.g., object distance, camera displace-
ment) and/or discrete variables. After evaluating policy π under
varied conditions, we generate episodes D = { ( θ i , x i ) } N
i =1
with observed outcomes x i (e.g., task success). The posterior
distribution p ( θ | x ) ∝ p ( x | θ ) p ( θ ) is approximated
using Mixed Neural Posterior Estimation (MNPE), which
trains a neural density estimator q ϕ ( θ | x ) to directly learn
the mapping from observations to parameter distributions.
The resulting posterior q ϕ ( θ | x ) characterizes which scene
variables are most associated with a target observation x . Our
approach provides systematic assessment of which variables
most strongly influence performance outcomes. Further details
are in Appendix B.

D. Robolab scene and task generation

RoboLab offers a user-friendly workflow that mirrors the
process of preparing a real-world robot evaluation (Fig. 1):
1) create a scene by positioning and orienting objects in a
workspace; 2) define a task as language instructions for a goal
state in the scene; 3) instantiate an environment by selecting
a robot, policy, and variations of scene features including
camera, lighting, and backgrounds for a task. We make this
process reproducible and scalable by decoupling task definitions
from environments, allowing reuse over new embodiments and
policies. Our approach automates the process of environment
assembly, reducing manual labor when evaluating a new robot
or policy. In addition, we developed an automated workflow

# Page 5
Fig. 6: Example scene variations, lighting variations, and camera pose variations in RoboLab.

to generate new scenes and tasks to facilitate extension of our
evaluations and to mitigate benchmark saturation in the future.
Formally, define a scene S = { ( b i , p i , q i ) } N
i =1 , where b i
represents an object instance selected from the available catalog
of objects B and p i ∈ R 3 , q i ∈ SO (3) denote its position
and orientation. Define a task T = { S, l } , where l is the
language instruction to complete in the scene. Define a policy
π : O →A where the action space A ∈{A joint , A EE , . . . }
and observation space O = ( O proprio , O rgb , O depth · · · ) is policy
dependent. An environment E = ( T , R , O , A , ξ ) consists of a
task, robot embodiment R , policy parameters ( A , O ), and scene
variations ξ = ( ξ camera , ξ light , ξ background , ξ pose ) . More details on
the specific objects, scenes and tasks in RoboLab can be found
in Appendix A.
1) Scaling Scene Generation: We enable scaling scene
generation through an automated pipeline that: 1) prompts an
LLM to generate a structured scene plan for asset placement;

Fig. 7: Examples of language ablation experiments. Top: Same
scene and goal, but the instruction wording ranges from precise to
increasingly vague. Middle: Same scene, but the instruction specifies
different tasks to perform. Bottom: Same instruction, but the scene
becomes progressively more complex.

2) uses a geometric solver and physics simulation to check
asset placement validity; and 3) refines the scene if it is not
valid. First, the LLM is prompted with a theme (e.g., “messy
counter”) to generate a structured scene plan consisting of a
subset of objects B ⊂B and spatial predicates P governing the
layout. The LLM is provided with the full catalog of objects
B containing names and bounding box dimensions d i ∈ R 3 .
Second, a spatial solver converts the relational predicates P
into valid pose configurations ( p , q ) ). Objects are processed in
dependency order, with support surfaces placed before objects
on those surfaces (Algorithm 1). To check physical stability,
the scene is then forward simulated in Isaac Sim [ 23 ] for 300
steps under gravity. An object b i is flagged as unstable if it’s
maximum Euclidean displacement is larger than a threshold
(typically 0 . 02 m). Third, If any object is unstable, we generate
a text error describing the failure (e.g., “Object ‘apple’ fell off
‘plate’ with displacement 0.15m”). This feedback is provided
to the LLM to refine the scene plan and repeat the process.
Further details, including on the spatial and physical solvers,
are provided in Appendix C.

2) Scaling Task Generation: We enable scaling task genera-
tion through an automated pipeline that: 1) generates task code
from information including the scene and competency axes; 2)
validates code syntax; 3) validates asset selections in the scene;
and 4) refines the task if it is not valid. First, we prompt an LLM
with detailed task information: 1) the scene object catalog B S
and metadata (including bounding boxes and semantics) with
dimensions; 2) task examples demonstrating the task structure;
3) the complete predicate library defining sub-task success
and termination; 4) Competency-axes language templates
with placeholders for objects, spatial verbs, and attributes;
and 5) constraints including difficulty levels and physical
feasibility requirements (e.g., containment size constraints,
stacking stability). The prompt forbids referencing objects
not present in B S and includes previously generated tasks to
prevent duplicates (see Appendix C for details). Second, tasks

# Page 6
TABLE I: Overall performance of VLAs on RoboLab. While recent VLAs exhibit emerging capabilities across diverse task dimensions,
overall success rates and consistency remain limited.

Overall Metrics
Difficulty (succ%)
Procedural (succ%)
Relational (succ%)
Visual (succ%)
Model
Succ% ( ↑ )
Score ( ↑ )
SPARC ( ↑ )
Speed ( ↑ )
simple
moderate
complex
affordance
reorientation
stacking
conjunction
counting
spatial
color
semantics
size

π 0 . 5 [9]
23.3
0.39
− 9.92 ± 6.0
5.7 ± 1.8
26.3
23.2
11.7
13.3
16.7
15.0
56.2
50.0
19.0
17.3
18.3
13.3
π 0 -FAST [27]
15.7
0.29
− 9.53 ± 6.1
4.6 ± 1.7
21.7
11.3
2.9
1.7
3.3
6.7
38.8
40.0
16.9
5.8
11.7
1.7
GR00T N1.6 [24]
2.0
0.10
− 9.25 ± 5.0
4.0 ± 1.8
1.9
3.1
0.0
3.3
0.0
13.3
0.0
0.0
0.0
0.0
2.0
0.0
π 0 [5]
5.2
0.14
− 9.51 ± 3.9
4.4 ± 1.4
8.1
2.6
0.0
0.0
0.0
0.0
20.0
12.9
2.4
0.0
1.7
3.3
PaliGemma [4]
1.5
0.07
− 21.25 ± 14.9
0.9 ± 1.1
1.9
1.5
0.0
0.0
0.0
0.0
1.2
8.6
1.0
0.0
2.3
0.0

are check for syntax validity as code. Third, asset validation
checks that all objects are not in the forbidden set and, for
containment tasks (e.g., “place b i inside b j ”), that inner objects
fit inside containers with some clearance. Fourth, if validation
fails, feedback is gathered into a fix prompt Q fix that includes
the original prompt Q , the invalid output, and an error message
E describing syntax errors or invalid asset references. The fix
prompt is provided to the LLM to refine the task and repeat
the process.
We evaluated our task generation approach using an LLM-
as-judge framework. We generated 812 tasks across 59 scenes
evenly across the competency axes with o1 [ 26 ]. We then
extracted each natural-language instruction and its program-
matic termination conditions from the generated code, and
prompted a second o1 judge to score instruction–criterion
alignment across relation, target, object, and quantifier match,
plus instruction clarity and physical feasibility (each on a 0–
1 scale), and to assign an aligned/partial/misaligned verdict.
Overall, tasks achieved 0 . 91 alignment, 0 . 96 clarity, 0 . 92
feasibility, and 0 . 95 semantic match, with 76% judged fully
aligned (misaligned ≈ 1% ) and covered 88% of objects. These
results show our approach can scale to generate diverse tasks
that are semantically aligned to their language instructions (see
Appendix D).

IV. E XPERIMENTS

We evaluate several off-the-shelf VLA policies on RoboLab-
120, controlled ablations, and environmental perturbations to
identify which competencies generalize and where failures
concentrate. The experiments are designed to address the
following questions: Q1: How well does a real-world policy
perform in our simulated benchmark? Q2: How well does a
policy generalize with language variations? Q3: When and
why does a policy fail?

A. Experiment Setup

We evaluate 120 tasks of varying difficulty levels (65 simple,
38 moderate, 18 complex) and spanning competency axes (44
relational, 91 visual, and 36 procedural). Each task was assigned
to one or more competency axes. Our experiments used the
DROID robot [ 13 ], which is commonly used to benchmark
VLAs [ 10 , 1 ]. DROID has a 7-DOF Franka Panda robot arm
with a Robotiq-2F-85 gripper, an externally mounted ZED 2i
camera with f =2 . 1 mm, and a ZED mini as the wrist camera.
We evaluated VLAs with off-the-shelf checkpoints fine-tuned
on the DROID dataset [ 13 ]: π 0 . 5 [ 9 ], π 0 -FAST [ 27 ], π 0 [ 5 ],
PaliGemma [ 4 ], and GR00T N1.6 [ 24 ]. The action space is
7-DOF Franka joint positions and a 1-DOF binary gripper

command. The environments were composed of a default office-
like background and natural lighting to mimic typical setups in
the DROID dataset [ 13 ], with wrist and external camera poses
designed to match the real-world DROID robot. Each task was
repeated 10 times with a fixed seed to address uncontrolled
stochasticity in the physics simulation and robot policy.

B. Task Results

Table I shows the overall results on our benchmark. Overall
success rates were low, with the best policy ( π 0 . 5 ) reaching
31.9% success. This matches prior observations on out-of-
domain generalization for VLAs [ 35 ]. Below, we discuss π 0 . 5
to illustrate how RoboLab supports targeted diagnosis of policy
capabilities and suggests concrete directions for improvement.
Competency axes also highlighted asymmetric generalization
in relational reasoning tasks: π 0 . 5 handled conjunctions
(76.0% success) and counting (60.0%) better than spatial
relations (23.9%). In visual grounding , performance remained
low across attribute types (35.0% for size, 30.0% for color,
and 21.5% for semantics), indicating brittle language-to-object
binding beyond a narrow set of familiar object descriptions.
Procedural understanding proved most challenging: π 0 . 5
achieved modest success on reorientation (53.3%) but struggled
with affordances (20.0%) and stacking (16.0%). Together, these
results show how RoboLab isolates where generalization fails,
supporting diagnosis that can inform data collection and training
priorities.

C. Ablation Experiments
To further probe robustness and language grounding, we
performed controlled ablations that varied the instruction, scene,
or task in isolation (Fig. 7).

Varying instruction specificity in a fixed scene. Table IIa
shows how VLAs respond to varying levels of instruction
specificity. Results reveal that VLAs lack grounding for abstract
or implied goals. Interestingly, we observe that the for the
vague command “Empty the grey bin”, π 0 . 5 tries to grab
the bin instead of clearing its content. These results indicate
that current VLAs may rely on keyword matching within
instructions rather than demonstrating the linguistic inference
necessary to identify implicit task goals. As shown in Table IIa,
π 0 . 5 exhibits higher performance on specific instructions than
on vague ones, indicating a sensitivity to unde-rspecified task
goals.

Varying scene complexity with a fixed instruction. Table IIb
isolates the effect of scene complexity by increasing the
number of objects to manipulate. Success rates drop as object

# Page 7
TABLE II: Language understanding ablations. (a) VLA perfor-
mance degrades with abstract or vague language instructions.
(b) Performance drops as scene complexity increases. (c)
VLAs show brittle language grounding, with consistent object
confusion patterns across different instructions in the same
scene.

(a) Effect of language specificity on task performance.

π 0 . 5
π 0 -FAST

Task
Succ %
Score
Succ %
Score

Bananas Out Of Bin Task

“Take all the bananas out of the grey bin
and put it on the table.”
50
0.13
30
0.05

“Take the bananas out”
40
0.22
10
0.15
“Empty the grey bin”
10
0.07
70
0.11

White Mugs In Bin Task

“Put the white mugs in the grey bin”
80
0.50
20
0.22
“Put the mugs in the bin”
90
0.50
10
0.11
“Put away mugs”
0
0.00
0
0.00

Remove Measuring Spoons from the Plate Task

“Put the orange measuring cup and the blue
measuring cup outside of the plate”
20
0.47
0
0.31

“Clear the plate”
0
0.08
0
0.10

(b) Effect of scene complexity on task performance.

Scene
π 0 . 5
π 0 -FAST
π 0
GR00T N1.6

Task: “Pack boxed foods into the bin”

1 Box/Can
10
0
0
0
2 Boxes/Cans
0
0
0
0
3 Boxes/Cans
0
0
0
0

Task: “Pack canned foods into the bin”

1 Box/Can
70
30
0
0
2 Boxes/Cans
30
10
0
0
3 Boxes/Cans
20
0
0
0

(c) Instruction sensitivity within fixed scenes.

Task / Prompt
π 0 . 5
π 0 -FAST
π 0
GR00T N1.6

Fruit Plate Scene

“Move an orange or a lime to the wood
bowl”
50
0
0
0

“Move an orange to the white bowl”
0
0
0
0
“Put the onion in the wood bowl”
70
10
20
20
“Put the onion on the plate”
0
0
0
0

Tools Cleanup Scene

“Put hammers in the right bin”
20
0
0
0
“Put hammers in the left bin”
10
0
0
0

Tools Selection Scene

“Select the cordless drill and put it on the
table”
70
50
20
30

“Select the blue hammer and put it on the
table”
0
0
10
0

count increases: from 70% with a single target object to
20% with three objects. More revealing is the type of failure:
VLAs exhibit systematic geometric biases, frequently grasping
cylindrical objects (cans) when instructed to manipulate boxes.
This suggests that training data distributions create strong
shape priors that override language-specified targets, a critical
limitation for real-world deployment where objects vary widely
in geometry.

Varying tasks in a fixed scene. Table IIc shows whether VLAs

TABLE III: Robustness to controlled environmental variations
over two simple tasks (BananaInBowl, BananaAndCubeInBowl).
PaliGemma is excluded as it fails to achieve meaningful results.

π 0 . 5
π 0 -FAST
π 0

Variation
Succ.%
Time (s)
Succ.%
Time (s)
Succ.%
Time (s)

Lighting

Color
96.7
14.5 ± 7.9
93.3
17.9 ± 10.7
6.7
31.1 ± 4.3
Shadows
100.0
16.0 ± 6.0
90.0
12.4 ± 3.5
0.0
-
Dim
90.0
9.1 ± 2.1
70.0
13.1 ± 2.8
70.0
35.5 ± 9.7
Overexposed
100.0
13.9 ± 4.4
100.0
9.6 ± 1.7
0.0
-

Visual Variations

Background
85.0
14.4 ± 8.7
70.0
21.3 ± 10.7
25.0
31.6 ± 11.8
Table texture
87.5
19.0 ± 13.8
60.0
19.0 ± 12.9
22.5
28.1 ± 6.9

Object Pose

10cm
95.0
16.2 ± 9.2
55.0
26.5 ± 13.8
22.5
34.7 ± 13.3
20cm
95.0
19.7 ± 10.2
40.0
21.8 ± 9.5
20.0
37.4 ± 11.2
30cm
62.5
18.9 ± 8.7
35.0
24.3 ± 11.9
17.5
24.3 ± 11.9

Camera Pose

external
85.0
17.4 ± 11.3
45.0
27.7 ± 10.6
50.0
27.4 ± 16.0
wrist
60.0
21.9 ± 13.4
25.0
20.1 ± 9.9
10.0
35.3 ± 10.4

TABLE IV: Overall success rate (%) on RoboLab-120 across
language specificity levels.

Model
Vague
Default
Specific

π 0 . 5
16.8
23.3
25.8
π 0 -FAST
9.7
15.7
15.2
GR00T N1.6
1.8
2.0
2.2
π 0
3.4
5.2
6.5
PaliGemma
1.5
1.5
1.0

can flexibly respond to different instructions while holding
the scene fixed. Results expose brittle language grounding:
π 0 . 5 was highly sensitive to object choice; 70% success for
“Select the cordless drill and put it on the table” but 0%
when replacing “cordless drill” with “blue hammer”. Error
analysis reveals consistent object confusion patterns: visually
similar distractors (pumpkin vs. orange, drill vs. hammer)
frequently override language-specified targets (see Table. VII
in Appendix for detailed analysis). These findings indicate that
VLA language grounding is highly sensitive to the specific
object-instruction pairings seen during training, rather than
reflecting generalizable language-to-object binding.

D. Sensitivity and robustness

We perform a set of variations given two basic tasks and
observe the outcome, for example, we only change the target
object to pick or the target for place, this is akin to domain
randomization [ 29 ] as illustrated in Fig. 6. The following
variations are considered, variations 1) in wrist and external
camera poses; 2) object poses; 3) in visual features, including
background and table textures; and 4) in lighting, including
saturation and hue. Table III illustrates the results for all
experiments.

Visual and Lighting variations.
We vary the lighting
conditions via color temperature shifts, lighting exposure
and strong directional light that generates shadows as the
robot is moving. Lighting: VLAs were robust to changes
in lighting conditions, with 90–100% success across shadow

# Page 8
TABLE V: Overall success rates across real and simulation environ-
ments across 6 selected simple tasks.

Environment
π 0 . 5
π 0 -FAST
π 0
PaliGemma

Real
79.5
34.1
63.2
0.0
Sim
74.0
42.0
18.0
4.0

variations, color temperature shifts, and 500 × intensity changes.
Visual appearance: Variations over 10 background textures
and 4 table textures had minimal impact ( < 5% degradation),
suggesting generalization to scene appearance changes.

Camera variation sensitivity analysis. We infer posteriors
over camera displacement conditioned on task success (Fig. 8).
Camera poses were randomized in both orientation and position
for 10 episodes each. Displacement is calculated with respect
to the nominal position of the cameras. Across all policies,
the wrist-camera posterior is sharply concentrated near zero,
indicating that successful execution often required the wrist
camera to remain close to its nominal pose, while performance
is more tolerant to external camera position changes. This
indicates performance is critically dependent on wrist camera
than external camera.

Object pose variation sensitivity analysis. We randomize
initial object poses via a uniform distribution of 10cm, 20cm,
and 30cm within its nominal placement (usually in front of
the robot) for 10 episodes each. We then infer posteriors over
initial object poses conditioned on task success (see Fig. 8),
relative to the robot pose. We observe a strong peak over 0.5m
from the robot’s origin, suggesting that objects placed at this
distance has the highest probability of success, likely due to
reachability.

E. Real Robot Verification

We evaluated the same policies on a small set of six
simple real-robot tasks and compared success rates to matched
simulation evaluations (Table V). π 0 . 5 achieved 79.5% success
in the real world, close to its 74.0% success in simulation,
suggesting that RoboLab can provide a reasonable proxy for this
policy on these task types. π 0 -FAST achieved 34.1% success in
the real world and 42.0% in simulation, showing a similar trend.
π 0 was a notable outlier, reaching 63.2% success on the real
robot but only 18.0% in simulation; qualitatively, this policy
appeared tuned to reliably grasp single objects, which matched
the selected real-robot tasks. We leave deeper investigation of
policy-specific sim-to-real deviations to future work.

V. L IMITATIONS

While RoboLab provides a flexible and scalable framework
for evaluating language-conditioned manipulation, it currently
focuses on rigid-body tabletop scenes and does not fully
capture the challenges of deformable object manipulation (e.g.,
cloth, cables, bags). Moreover, many contact-rich skills that
require precise force control, compliant interaction, or complex
frictional dynamics are underrepresented and dependent on the

physics simulation fidelity, limiting Robolab’s coverage of fine-
grained, low-level control tasks. Finally, although evaluation
in high-fidelity simulation is a strong proxy for real-world
performance, a residual visual distribution shift remains. This
gap needs to be characterized further both by analyzing the
behavior and robustness of the visual perception stack and
through extensive validation on real-world deployments.

VI. C ONCLUSION

Recent benchmarking efforts have made significant strides in
scalable robot evaluation, but they primarily assess robustness
to perturbations of training environments rather than true task
generalization to novel scenarios. RoboLab addresses this gap
by evaluating real world policies in a high-fidelity simulation,
structured evaluation vectors that decompose policy competence
into visual, procedural, and relational dimensions, and a set
of sensitivity analysis set of novel analysis that provides
insight into policy behavior for robotics. Our benchmarking
framework enables the community to critically answer the
question of generalization and performance . At the same time,
the framework is designed to be pragmatically usable: new
tasks can be authored in minutes by arranging objects on a
tabletop and attaching language instructions, and a generative
scene–task–environment workflow that supports continuous
benchmark evolution.

R EFERENCES

[1] Pranav Atreya, Karl Pertsch, Tony Lee, Moo Jin Kim,
Arhan Jain, Artur Kuramshin, Clemens Eppner, Cyrus
Neary, Edward Hu, Fabio Ramos, et al.
Roboarena:
Distributed real-world evaluation of generalist robot
policies. In Proceedings of the Conference on Robot
Learning (CoRL 2025) , 2025.
[2] Sivakumar
Balasubramanian,
Alejandro
Melendez-
Calderon, and Etienne Burdet. A robust and sensitive
metric for quantifying movement smoothness.
IEEE
Transactions
on
Biomedical
Engineering ,
59(8):
2126–2136, 2012. doi: 10.1109/TBME.2011.2179545.
[3] Prithviraj Banerjee, Sindi Shkodrani, Pierre Moulon,
Shreyas Hampali, Shangchen Han, Fan Zhang, Linguang
Zhang, Jade Fountain, Edward Miller, Selen Basol,
Richard Newcombe, Robert Wang, Jakob Julian Engel,
and Tomas Hodan. HOT3D: Hand and object tracking in
3D from egocentric multi-view videos. CVPR , 2025.
[4] Lucas Beyer, Andreas Steiner, Andr ´ e Susano Pinto,
Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim
Neumann, Ibrahim Alabdulmohsin, Michael Tschannen,
Emanuele Bugliarello, Thomas Unterthiner, Daniel Key-
sers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey
Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong,
Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko
Bo ˇ snjak, Xi Chen, Matthias Minderer, Paul Voigtlaender,
Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi
Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut,
Jeremiah Harmsen, and Xiaohua Zhai. Paligemma: A
