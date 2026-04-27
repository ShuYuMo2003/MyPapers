# Page 1
π 0 . 7 : a Steerable Generalist Robotic Foundation
Model with Emergent Capabilities

Physical Intelligence
Bo Ai, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Greg Balke, Kevin Black, George Bokinsky, Shihao Cao, Thomas Charbonnier,
Vedant Choudhary, Foster Collins, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Maitrayee Dhaka, Jared DiCarlo, Danny Driess,
Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachlan Groom, Haroun Habeeb,
Hunter Hancock, Karol Hausman, Gashon Hussein, Victor Hwang, Brian Ichter, Connor Jacobsen, Szymon Jakubczak, Rowan Jen, Tim Jones,
Gregg Kammerer, Ben Katz, Liyiming Ke, Mairbek Khadikov, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Brendon LeCount, Sergey Levine,
Xinyu Li, Adrian Li-Bell, Vladislav Lialin, Zhonglin Liang, Wallace Lim, Yao Lu, Enyu Luo, Vishnu Mano, Nandan Marwaha, Aikys Mongush,
Liam Murphy, Suraj Nair, Tyler Patterson, Karl Pertsch, Allen Z. Ren, Gavin Schelske, Charvi Sharma, Baifeng Shi, Lucy Xiaoyang Shi,
Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Will Stoeckle, Jiaming Tang, Jimmy Tanner, Shalom Tekeste, Marcel Torne,
Kyle Vedder, Quan Vuong, Anna Walling, Haohuan Wang, Jason Wang, XuDong Wang, Chris Whalen, Samuel Whitmore, Blake Williams,
Charles Xu, Sukwon Yoo, Lili Yu, Wuming Zhang, Zhuoyang Zhang, Ury Zhilinsky

https://pi.website/pi07

Robot Data

Demonstration Data

close the microwave
open the upper left cabi

put the ketchup in teh fridge pick up the mitten case to the left

load the lower rack
sharpie

throw away the silver spoon

close the dishwasher

Autonomous Data

Specialist-Level Dexterity

pick up the knife
chop the zucchini

Quality

Length

Language
Instructions

TRAINING

π 0.7 Vision-Language-Action Model

prompt

Episode
Metadata
Subgoal Images

action expert

INFERENCE

High-Level Policy
World Model
Desired Metadata

Quality
Length

Out-of-the-Box Performance Without Fine-Tuning

Non-Robot Data

Multimodal Web Data

Egocentric Human Data

Cross-Embodiment Transfer

Fig. 1: We introduce π 0 . 7 , a steerable generalist robot foundation model that can perform dexterous tasks across many tasks, environments, and robots. π 0 . 7 is
trained with diverse prompts that contain not only the task description, but detailed language, generated subgoal images, and episode metadata. This provides
richer context about not only what to do, but also how to do it, making it possible for π 0 . 7 to leverage broad set of both robot and non-robot data and
compose the skills in this data in new ways to solve new tasks.

Abstract —We present a new robotic foundation model, called
π 0 . 7 , that can enable strong out-of-the-box performance in a wide
range of scenarios. π 0 . 7 can follow diverse language instruc-
tions in unseen environments, including multi-stage tasks with
various kitchen appliances, provide zero-shot cross-embodiment
generalization, for example enabling a robot to fold laundry
without seeing the task before, and perform challenging tasks
such as operating an espresso machine out of the box at
a level of performance that matches much more specialized

RL-finetuned models. The main idea behind π 0 . 7 is to use
diverse context conditioning during training. This conditioning
information, contained in the prompt, makes it possible to
steer the model precisely to perform many tasks with different
strategies. It is conditioned not just on a language command
that describes what it should do, but on additional multimodal
information that also describes the manner or strategy in which
it should do it, including metadata about task performance and
subgoal images. This enables π 0 . 7 to use very diverse data,

arXiv:2604.15483v1  [cs.LG]  16 Apr 2026

# Page 2
including demonstrations, potentially suboptimal (autonomous)
data including failures, and data from non-robot sources. Our
experiments evaluate π 0 . 7 across numerous tasks with multiple
robot platforms, on tasks that require speed and dexterity,
language following, and compositional task generalization.

I. I NTRODUCTION

I am a part of all that I have met.

Alfred, Lord Tennyson, Ulysses

Foundation models work on the principle that generalist
capabilities emerge from training on large and diverse datasets.
For example, large language models can not only recall facts
and semantic knowledge, but they can also compose that
knowledge in new ways, solving problems that require unlikely
connections, applying user-defined formats (e.g., JSON), and
performing chain-of-thought reasoning. This kind of composi-
tional generalization is arguably the cornerstone of generalist
capabilities, but it has proven elusive in the domain of physical
intelligence. While robotic foundation models such as vision-
language-action models (VLAs) have advanced significantly in
size and capability, their ability to generalize to new tasks or
recombine skills in new ways has so far been limited. Unlike
language models, which can compose different capabilities
from their training data to solve new problems, prior VLAs
not only lack the ability to solve new tasks, but often struggle
to fluently perform all of the instructions they were trained on
without task-specific fine-tuning.
In this paper, we present a new model, π 0 . 7 , that exhibits
strong signs of compositional generalization — enabling it to
follow diverse language instructions, attain performance com-
parable to more specialized fine-tuned models on dexterous
tasks, and even compose these behaviors in new ways. This is
enabled by leveraging large and diverse datasets, including
data from many robots with diverse strategies, suboptimal
data from autonomous execution (including both data from
RL post-trained agents as well as failures), and non-robot
data from videos of humans performing tasks and general
multimodal data from the internet. However, using such data
na¨ıvely does not lead to success: with a diversity of examples
that differ in terms of both strategy and task performance, a
na¨ıve training process would lead to a model that averages
together different modes in the dataset and produces sub-
optimal results. In training π 0 . 7 , we address this challenge
by annotating the data with detailed context annotations that
contain not only information about what to do but also how to
do it and provide this knowledge to the model using a variety
of multimodal conditioning signals. In this way, each episode
teaches the robot about nuanced concepts and skills that it
could use not only to perform the training tasks effectively,
but also to compose in new ways to solve new tasks. Our
proposed prompt structure includes detailed language labels,
strategy metadata, and multimodal information such as subgoal
images. This allows us to resolve the ambiguity in large
and diverse datasets, learn from suboptimal behaviors without
hurting performance, and obtain broad generalization across
instructions, embodiments, and environments.

The idea that detailed prompts or context can improve the
performance of foundation models has been explored in other
fields. For example, models for image and video generation
utilize prompt expansion to produce high-quality generations.
Our approach has many parallels to such methods. However,
in robotics, simply captioning the data with more detailed text
is not enough — the details that determine task success and
proficiency might be more subtle (e.g., information about the
overall quality of the episode), or might simply be hard to
express with language alone (e.g., the particular appearance of
a cleanly folded t-shirt). Therefore, in addition to using more
detailed text, our model adds a range of additional metadata
to the prompt, as shown in Fig. 1 , including information about
episode quality (strategy metadata), the control modality used
by the robot, and subgoal images. Some of this information
can be provided or omitted at test time, but including it in
training results in a model that can more effectively compose
the concepts that it was trained on and exhibit a variety of
emergent capabilities.
In our evaluation, we show that π 0 . 7 exhibits a number of
capabilities that go beyond prior robotic foundation models:

• Out-of-the-box performance: π 0 . 7 can reliably perform
highly dexterous, long-horizon tasks such as using an
espresso machine, folding laundry, taking out a trash bag,
folding a box, and peeling vegetables, without any task-
specific post-training and in a variety of environments.

• Instruction generalization: π 0 . 7 can follow a diverse set of
language instructions in unseen environments and demon-
strates robust generalization to complex, unseen language
references. For example, π 0 . 7 can follow a diverse set
of open-ended instructions in entirely unseen kitchen and
bedroom environments.

• Cross-embodiment generalization: π 0 . 7 can enable zero-
shot cross-embodiment transfer, making it possible to trans-
fer dexterous tasks such as folding a t-shirt to a robot
that was never trained to perform any laundry folding task,
matching the performance of expert operators teleoperating
the robot on their initial attempts.

• Compositional task generalization: π 0 . 7 can be instructed
to perform new tasks by composing skills in previously
unseen ways. For example, we can prompt π 0 . 7 to use new
kitchen appliances, such as loading a sweet potato into an
air fryer, or prompt it to perform tasks in new ways.

Through ablation and scaling studies, we also empirically
demonstrate that there is a strong synergy between diverse
datasets and detailed contexts: our approach enables learning
from mixed-quality data and non-standard data sources with-
out hurting the model performance, and diverse data boosts
the model performance when detailed context information is
provided during training.

II. R ELATED W ORK

Generalist robot manipulation policies. There is a large
body of work studying the development of generalist robot
policies. These generalist policies are sometimes trained from

# Page 3
scratch [ 1 – 6 ], but are more commonly initialized using pre-
trained vision-language models [ 7 – 23 ] or pre-trained video
generation models [ 24 – 28 ]. Various works have developed
architectural components of VLAs such as memory [ 29 – 37 ],
hierarchy for long-horizon planning [ 13 , 14 , 38 – 40 ], and
goal image conditioning [ 41 ]. We develop a VLA model
that incorporates all three of these components in a single
model, building on top of the π 0 . 6 -MEM architecture [ 37 , 42 ].
While generalist policies are most often trained on robot
demonstration data, prior works have shown how to derive
benefits from incorporating web data [ 7 ], egocentric videos of
humans [ 25 , 43 – 49 ], and autonomous robot experience [ 50 –
52 ] into pretraining. We incorporate all of these data sources
and find that the combination of diverse data with detailed
prompting yields a model with strong signs of compositional
generalization and performant out-of-the-box behavior.
Generalization across tasks and embodiments. Much prior
work has aimed to learn robot policies that generalize not only
to different environments, objects, and backgrounds, but also
to entirely new tasks and embodiments. Often, this is done
by leveraging human video data, either for general represen-
tation learning [ 53 – 58 ], by directly supervising with human
motions [ 45 , 59 – 65 ], or by extracting 2D point tracks [ 66 –
69 ]. Other work has aimed to improve generalization by di-
rectly leveraging Internet pre-trained foundation models during
training or inference [ 70 – 76 ]. With the increasing availability
of large, cross-embodiment robot datasets [ 77 ], there has also
been work on explicitly improving cross-embodiment trans-
fer between robots [ 78 – 84 ]. Rather than leveraging existing
datasets, some works have proposed specialized hand-held
devices that can be used to collect data that can then generalize
to various robot embodiments [ 85 , 86 ]. In this work, we
find that the right prompting allows our model to leverage
diverse robot, human, and Internet data to achieve strong
generalization across tasks and embodiments.
Prompting robots with subgoal images. A core architectural
component of our model relative to π 0 . 6 -MEM is to allow the
model to be prompted using goal images, including generated
subgoal images. Conditioning robot manipulation policies on
goal images and videos has been explored in a large body of
work. Some of these works utilize user-provided images [ 87 –
90 ], while others condition the policy on generated goal
images from a separate model [ 91 – 98 ] or in a chain-of-thought
fashion [ 41 ]. Alternatively, image and video generation can be
integrated into policy training objectives [ 24 – 26 , 99 – 101 ] to
improve policy representations and produce more generaliz-
able actions. We view our contribution as complementary to
these works: we do not aim to propose a new architecture or
model design, so much as a methodology for enabling VLAs
to utilize more diverse data sources, together with an empirical
analysis showing that it leads to strong indications of composi-
tional generalization. To our knowledge, our empirical results
go significantly beyond the quantitative improvements reported
in prior works, showing zero-shot transfer of dexterous skills
like laundry folding to a different robot and generalization to
novel object interactions such as operating an air fryer.

III. F LOW -B ASED V ISION -L ANGUAGE -A CTION M ODELS
VLAs are trained by starting from a pre-trained vision-
language model (VLM) backbone, and adapting it for robotic
control. The training dataset D contains robot trajectories,
which are sequences of observations o t and actions a t . The
observations o t = [ I 1
t , . . . , I n
t , q t ] consist of n camera images
I i
t and the joint configuration of the robot q t , while the actions
a t consist of joint or end-effector commands.
VLAs are typically trained to predict an action chunk ,
corresponding to a short trajectory of future actions a t : t + H ,
based on a recent history of observations o t − T : t (often a
shorter horizon of actions, ˆ H < H , is executed). The action
chunk can be generated by an “action expert”, a smaller
transformer that attends to the VLM backbone and enables
fast inference at runtime. The action expert typically uses
a flow matching [ 102 ] (or diffusion) objective that captures
the multi-modality of the robot actions. To learn effective
representations, our model also uses the knowledge insulation
(KI) training recipe [ 103 ]: the VLM backbone is supervised
with FAST tokens [ 104 ], and while the action expert attends
to all of the activations in the VLM backbone, gradients from
the action expert do not flow into the VLM backbone, such
that the VLM is trained via the comparatively stable discrete
cross-entropy loss.
In addition to the observation and action, each training
example for the VLA is accompanied by a prompt or context ,
which we denote with C t . Conventionally this corresponds to
a language instruction ℓ t , such that C t = ( ℓ t ) , provided by a
human annotator (e.g., “clean up the kitchen”).
In designing π 0 . 7 , we explore how additional information
added to the context for each training example can enable
learning from diverse and heterogeneous datasets (including
suboptimal behaviors and failures). As we will show, training
with this data leads to a model with greater robustness and
dexterity, and makes it possible for the model to generalize
more broadly. The training objective for the VLA π θ corre-
sponds to an approximate log-likelihood given by

max
θ
E D [log π θ ( a t : t + H | o t − T : t , C t )] .
(1)

Note that a flow matching action expert optimizes an approx-
imate lower bound rather than a closed form log-likelihood
[ 10 ]. The dataset D typically consists of high-quality human
demonstration trajectories; however, as mentioned, we use a
broader dataset, which includes failed episodes and suboptimal
autonomous rollouts, as well as other data sources such as
egocentric human video data. We will show how using a suf-
ficiently detailed and informative context C t makes it possible
to incorporate such diverse data and, perhaps surprisingly, even
results in better policy performance and generalization.

IV. π 0 . 7 O VERVIEW
π 0 . 7 is our newest robotic foundation model that builds
on the existing VLA architecture from π 0 . 6 [ 42 ] and the
MEM memory system [ 37 ] and extends it with multi-modal
context conditioning. The model consists of a VLM back-
bone initialized from the Gemma3 4B-parameter VLM [ 106 ]

# Page 4
Æ ÎÐÒÏ ËÈÇÈÑÊÌÉÄÊÃÅÄÃÂÌÁÀ¾ÈÑÊÍ½Ñ¿Â¼

pk}3	kw	u
p4	3	k3	u

 k }
 } k }
 	.R	R

	!	d`\50Y-S&WWVc
eT	RRS&Qb

 }54R5	 

R	d

7}RÓ

ÙØ
753!0	.	4-4+
EMLHPMDAOLJGFCABN@<F?FKF;L99DF?8

§4dVd	£
EMLHPMDAOLJGF8«;²NF ?8

}	 	.

 k }eR	R

R	d

p4	3	k3	u
pk}3	kw	u

}	 	.

 } k }

R	d

#"

 

)'&
!
&#

		


 

Fig. 2: Architecture overview. The π 0 . 7 model is a 5B-parameter VLA consisting of a 4B VLM backbone, a MEM -style video history encoder and a 860M
parameter action expert. The model’s context includes multiple distinct modalities, including language commands, episode metadata that describes the data
quality and strategy, and multimodal inputs such as subgoal images. At runtime, the language commands are produced by a high-level semantic policy based
on the same architecture, and the subgoal images are produced by a lightweight world model based on the BAGEL image generation model [ 105 ].

(including a 400M-parameter vision encoder), and a flow
matching action expert with 860M parameters. The model
has about 5B total parameters. The vision encoder is also
initialized from Gemma3 and follows the design of the MEM
video history encoder [ 37 ], applying both temporal and spatial
compression over history observations and outputting a fixed
number of tokens for any number of history frames. An
overview of the model architecture is provided in Fig. 2 , and
Sec. VI-B describes the architecture in more detail.
Our previous models, π 0 , π 0 . 5 , and π 0 . 6 , use a short
textual description of the task as the context. In training
π 0 . 7 , we expand the context to include additional information
and modalities: more expressive language commands, episode
metadata, and subgoal images, making it possible to train on
diverse and potentially suboptimal data.

V. D IVERSIFYING THE P ROMPT

In this section, we describe each part of the prompt con-
tained in the context C t used by π 0 . 7 . The model is trained to
handle prompts that contain each of these components, though
it is trained with each component randomly dropped out so that
it can also handle any subset, providing flexibility at test time.

A. Subtask instructions

Following π 0 . 5 [ 14 ], we include intermediate, higher-level
text that captures the next semantic subtask as part of the
prompt in addition to the overall textual task description ℓ t
(e.g., “clean the kitchen”). We denote this intermediate text
by ˆ ℓ t (e.g., “open the fridge door”). During inference, ˆ ℓ t may
be produced by a learned high-level policy or a human (or

be omitted) and may change over time. We collect data from
a diverse set of tasks and scenarios, and then annotate the
segments with detailed textual descriptions.
Conditioning the model on the semantic subtask also en-
ables us to verbally coach the model step-by-step. Since the
model is trained to follow diverse language instructions, it
can follow the live instructions from the human in a new task,
e.g., loading a sweet potato into an air fryer (Fig. 14 ). After
coaching we can take the verbal coaching data to finetune
π 0 . 7 as a high-level policy that maps the robot observations,
task specification, and history of past subtask instructions to
the new subtask instruction (Fig. 2 bottom left). This high-
level policy then guides the robot to perform the task fully
autonomously.

B. Subgoal images

While subtask instructions are effective at conveying the
high-level intent of the task, they may lack details that matter
for execution — e.g., “open the fridge door” does not specify
how the robot arm should grasp the handle. Subgoal images
address this by depicting the desired near-future state of the
scene in images, providing a richer specification of what the
world should look like after successful progress of the task.
We consider multi-view subgoals g t = [ G 1
t , . . . , G n
t ] ,
where G i
t is the desired near-future image for camera i .
Multi-view subgoals simultaneously specify environment- and
object-centric outcomes (often easiest in the base view) and
arm/gripper outcomes (often easiest in wrist views), improving
spatial grounding for control.
At runtime, the subgoal images are produced by a

# Page 5
lightweight world model , which takes in the same subtask
instruction ˆ ℓ t as the main model, but benefits from web-
scale pre-training on videos and image editing tasks and is
thus capable of generalizing to diverse tasks and scenarios.
Generated subgoal images that are grounded in the robot’s
current observation can often more clearly disambiguate the
objective for the policy than a language instruction, resulting
in improvements in language following and generalization. We
denote this model as g ψ and it is trained with the objective

max
ψ
E D g
h
L CFM

g ⋆
t , g ψ ( o t , ˆ ℓ t , m )
i
,

where L CFM is the standard flow matching loss [ 102 ], g ⋆
t
is the future subgoal image, m is the episode metadata
from Sec. V-C , and the dataset D g is a subset of segments
from Sec. V-A that are annotated with especially high-quality
subtask labels ˆ ℓ t . The image frames at the end of the segments
serve as the ground-truth subgoal, i.e., g ⋆
t = o t end .
Following SuSIE [ 93 ], our world model is initialized using
an off-the-shelf image generation and editing model with
web-scale pre-training. We initialize from BAGEL [ 105 ], a
14B mixture-of-transformers model capable of image under-
standing, editing, and generation. By augmenting our world
model training with web data, non-robot data sources such
as egocentric human videos, and other video data, we can
acquire semantic and physical concepts from these other data
sources and then transfer them into π 0 . 7 via subgoal images.
Implementation details are in Appendix C .

C. Episode metadata

A key goal in expanding the context provided to the model
is to train on a broader, more diverse dataset of trajectories.
Instead of just using high-quality demonstration data, π 0 . 7
leverages lower quality demonstrations (including failures) and
even autonomous data from prior models. Since we still want
π 0 . 7 to perform the task as well as possible at test time,
we need to appropriately label these diverse trajectories with
information about how the task was performed so that the
model can correctly contextualize them. To this end, we add a
variety of “episode metadata” information to the context with
attributes of the given training episode. We denote the set of
metadata m , which may contain various labels including

• Overall speed : the length of the episode in timesteps. We
discretize the values in an interval of 500 steps, i.e., values
between 1750 and 2250 are binned to “2000 steps”. Often
faster speed also corresponds to higher quality, e.g., the
episode has fewer mistakes.

• Overall quality : task execution quality expressed as a score
between 1 and 5, with 5 being the highest quality.

• Mistake : label indicating whether the robot made a mistake
within a given action segment (e.g., failing to grasp an
object or performing the wrong subtask). These labels are
provided by humans coarsely annotating our data.
The π 0 . 7 model is thus trained with ground-truth episode
speed and manual annotations of the episode quality and mis-
take segments from a diverse data mixture. The data diversity

(e.g., episodes of varying speed) provides the necessary signals
for the model to learn to correlate such metadata with the
target action. At runtime, the model can then be instructed to
perform the task at high speed, with high quality, and without
mistakes, through metadata prompting.

D. Control mode

We also consider using different control modes for the low-
level action execution. Specifically, we include both joint-
level and end-effector actions during training and use a text
identifier c ∈{ joint , ee } to designate the control mode in
the prompt. Then at runtime, we can pick the control mode
depending on the task.

E. Full prompt and training details

<Multi-view observation><Multi-view subgoals>
Task: peel vegetables. Subtask: pick up the
peeler. Speed: 8000. Quality: 5. Mistake:
false. Control Mode: joint.<Proprioception>

Combining all of the context information together, the
example above illustrates a potential prompt that may be
provided to the model.
During training, we randomly drop out each part of the
prompt, which provides π 0 . 7 with the flexibility to use any
subset of the prompt components at test time (e.g., running
with or without subgoal images). First, we find that the model
trains significantly faster when given the subgoal images —
the action prediction task essentially becomes an “inverse
dynamics” problem inferring the robot action between the
current and future frames. Thus we only add the visual
subgoal images to 25% of the examples in each batch in
training. Among the examples with subgoal images we also
drop out the subtask instruction ˆ ℓ t 30% of the time as often
visual subgoal can substitute the equivalent textual subtask
description in richer details. For episode metadata, we drop
it entirely 15% of the time, and additionally each component
(overall speed, overall quality, and mistake label) is dropped
with 5% probability individually. We do not apply dropout for
the control mode.

VI. T HE π 0 . 7 M ODEL AND T RAINING R ECIPE

We now discuss how we incorporate the different context in
π 0 . 7 model by training on diverse data, as well as the details
about the model architecture, training, and inference.

A. Training datasets

The training dataset for π 0 . 7 consists of demonstration
data for a wide range of tasks with many different robot
platforms (both static and mobile, with single arm or biman-
ual) in diverse environments (in-house lab-like and home-
like environments, and in-the-wild home environments), au-
tonomous data from a large amount of policy evaluations,
human interventions within policy rollouts, open-source robot
datasets, egocentric human video data, and auxiliary non-robot
data sources from the web, including object localization and
attribute prediction, visual question answering, and text-only

# Page 6
Fig. 3: Prompt overview. π 0 . 7 uses diverse modalities of context in the prompt, including: subtask instructions, subgoal images, and episode metadata. We
train the model with dropout for each component, and then prompt the model flexibly combining modalities. For example, when using the UR5e bimanual
manipulator to fold a shirt, we use subgoal image and metadata prompting.

prediction. We also include video-language tasks including
video captioning of in-house robot data and from the web.
In a significant departure from classic VLA training
pipelines, we make heavy use of suboptimal robot data in
training. This includes both lower quality demonstrations
(failure episodes or success episodes with a substantial amount
of mistakes) and data collected by prior versions of our
models during model evaluation experiments 1 . For example,
we use data collected by the π ∗
0 . 6 model during RL training as
additional examples, effectively allowing π 0 . 7 to distill their
behavior. Incorporating the episode metadata into the context
allows our model to effectively use all of this evaluation data
and, as we will see in Sec. IX-A , enables it to attain similar
performance as models that are specialized for high perfor-
mance on individual tasks with RL. This corresponds to a kind
of “distillation” process, where the generalist π 0 . 7 model can
inherit the capabilities of RL-trained specialists. Suboptimal
data also diversifies the possible states and scenarios in a given
task and leads to stronger robustness, enabling the model to
even sometimes outperform RL-trained or generally, single-
task post-trained policies, in highly dexterous tasks.

B. Model architecture

The major architectural modifications of π 0 . 7 model com-
pared to previous π 0 . 5 and π 0 . 6 models include the use of

1 We exclude autonomous data collected in any generalization-focused
evaluation task (including ones in Sec. IX ) from training.

the history vision encoder from MEM [ 37 ] and visual subgoal
images in the context. The model takes as input up to four
camera images (front view, two wrist views, and optionally
rear view), each with up to six history frames, and up to three
subgoal images (omitting the rear view). The history frames
are processed through the vision encoder and compressed to
the same number of tokens as a single frame; subgoal images
are processed through the same encoder. Both the camera
observations and subgoal images are first resized to 448x448
pixels. For sampling history frames we use a stride of 1
second, and the entire history frames are dropped out entirely
with probability 0.3. The rear view image (when available) is
dropped out with probability 0.3 as well.

We employ a block-causal masking scheme, such that the
observation tokens and the subgoal image tokens use bidirec-
tional attention within themselves, and goal-image tokens can
additionally attend the observations. The following text tokens
use causal attention (see attention mask visualization in the
appendix). We also feed the proprioceptive state q t (including
the history states) of the robot into the model backbone. Unlike
π 0 . 6 that uses discretized text tokens to represent q t , π 0 . 7
follows MEM and embeds the state using a linear projection that
maps the state dimension to the backbone dimension. Each
history state is treated as an individual token; if the history
frame is dropped out, the corresponding state token is masked
out as well.

The more lightweight “action expert” is a 860M-parameter

# Page 7
transformer that is trained to predict continuous actions us-
ing flow matching objective. We use adaptive RMSNorm to
inject timestep information for flow matching. The number
of action tokens processed by the action expert is fixed at 50,
representing an action chunk of 50 steps. The 50 tokens attend
bidirectionally to each other and can also attend to the VLM
backbone activations.
π 0 . 7 also employs the training-time version of real-time ac-
tion chunking (RTC) [ 107 , 108 ] for generating smooth action
trajectories in the presence of inference delay. During training,
we simulate delays of 0 to 12 timesteps, corresponding to a
maximum inference latency of 240ms on a 50Hz robot.

C. Training with subgoal images

When training π 0 . 7 to handle subgoal images, we need
the model to accommodate goals with different delays and
different levels of image quality, including images generated
by our world model. This requires carefully selecting which
subgoals are provided as context to the model when training.
We train on a combination of real images from future timesteps
of the training trajectory and generated images. We found the
following sampling scheme to be effective for selecting the
timesteps for the real images: with probability 0 . 25 , we sample
the end-of-segment images (consistent with the prediction
target for the world model), and with probability 0 . 75 we
sample future images uniformly from 0–4 seconds ahead of the
current timestep. In addition to these real images, we mitigate
the train-test mismatch between real and generated images by
also sampling a large number of subgoal images from the
world model, and constructing additional training examples
with these generated images added into the context of π 0 . 7
instead of the real future images.

VII. P ROMPTING π 0 . 7 AT RUNTIME

At runtime, we configure π 0 . 7 to run with different forms
of context depending on the desired behavior without any
task-specific post-training . For any task we always prompt
the model with the control mode and episode metadata. For
choosing the episode metadata, we follow

• Overall speed: set per-task to the 15 th percentile of the
episode length from the task.

• Overall quality: always set to 5, which is the highest score.

• Mistake: always set to false, meaning no mistake.
The subtask instruction ˆ ℓ t is provided either by a learned high-
level language policy or by a human supervisor for coaching
(see Sec. V-A ). When the subgoal images are used, we refresh
the subgoal images whenever the semantic intent changes (i.e.,
new ˆ ℓ t ), or after ∆= 4 seconds have elapsed since the last
subgoal image was produced, whichever happens first. See
Algorithm 1 for the full workflow. We apply asynchronous in-
ference: the visual subgoal and subtask instruction generation
happens in separate threads and the VLA inference always
uses the latest ones available.
For all experiments we use 5 denoising steps to generate
the 50-step action chunks and execute ˆ H ∈{ 15 , 25 } steps out
of the chunk. Since each prompt component is trained with

Algorithm 1 Prompting π 0 . 7 at test time

1: Input: initial observation o 0 , task instruction ℓ , episode
metadata m , control mode c
2: Initialize subtask ˆ ℓ (from high-level policy or coaching)

3: g ⋆ ∼ p ψ ( g ⋆ | o 0 , ˆ ℓ, m )

4: C = { ℓ, ˆ ℓ, g ⋆ , m, c }
5: a t : t + H ∼ π θ ( a | o t − T : t , C )
▷ Optional: CFG
6: for t = 0 , 1 , 2 , . . . do
7:
if ˆ ℓ changed or ∆ -second timer elapsed then
8:
g ⋆ ∼ p ψ ( g ⋆ | o t , ˆ ℓ, m )
▷ Non-blocking (async)

9:
C = { ℓ, ˆ ℓ, g ⋆ , m, c }
10:
end if

11:
if ˆ H steps elapsed since last inference then
12:
a t : t + H ∼ π θ ( a | o t − T : t , C , a t : )
▷ Async w/ RTC
13:
end if
14:
Execute a t
15: end for

dropout, π 0 . 7 can also be used with classifier-free guidance
(CFG) [ 109 ] for any part of the prompt, for example to guide
the generated actions toward higher speeds. Concretely, each
action denoising step follows

∇ a log π θ ( a t : t + H | o t , C t )+

β ( ∇ a log π θ ( a t : t + H | o t , C t ) −∇ a log π θ ( a t : t + H | o t , C uncond
t
)) ,

where C uncond
t
denotes the set of context used in “uncondi-
tional” mode and β is the CFG weight. While any part of the
context could be dropped out, we apply CFG on the episode
metadata to elicit strong performance in dexterous tasks. We
use moderate values of β ∈{ 1 . 3 , 1 . 7 , 2 . 2 } .

VIII. R OBOT SYSTEM DETAILS

Fig. 4: Illustrations of some of the robots in our experiments. We evaluate
π 0 . 7 on a variety of robots, including bimanual mobile manipulators (left),
static bimanual robots (middle), and a bimanual UR5e setup (right) that we
use for cross-embodiment experiments.

We deploy π 0 . 7 in a variety of robot platforms (Fig. 4 ),
including bimanual mobile manipulators with two 6 DoF
arms, static bimanual manipulators with lightweight 6 DoF
arms (“BiPi”), and a bimanual UR5e system with Robotiq

# Page 8
“take out the trash”

“open toaster oven”
“close the toaster oven”

“grasp and turn the bottom
right oven toaster knob
with the right gripper”

“pick up the white plate
in the overhead cabinet
with the right gripper”

“put bagel on
the white plate”

Take Out
Trash

Toasting
a Bagel

Fig. 5: Illustration of selected evaluation tasks. We evaluate π 0 . 7 on a number of tasks, and two of the more longer-horizon ones are visualized here. For
some tasks such as “Take Out Trash”, we provide a coarse instruction like “take out the trash” and π 0 . 7 performs the full long-horizon task. For other tasks
which do not appear in the training data for π 0 . 7 such as “Toasting a Bagel”, we can leverage the strong language following capabilities of π 0 . 7 to coach it
to perform the task with a series of detailed instructions that break down the task step-by-step.

grippers, which we use for cross-embodiment experiments.
Additional generalization and language following experiments
use a single-arm 6 DoF system, using the same arms as the
BiPi platform. Note that while a large fraction of our data
is collected with arms that resemble the BiPi platform, the
UR5e arms that we use for cross-embodiment testing are
significantly longer, have a different morphology, and are
much heavier. In practice, the UR5e arms need to employ a
different manipulation strategy due to the shape of the arms,
their positioning over the table (on the sides rather than at one
edge), and the shape of the gripper and fingers, making cross-
embodiment transfer to this platform a significant challenge.
All manipulators use parallel-jaw grippers. The UR5e robots
run at 20 Hz, while all other robots run at 50 Hz. Each robot
has a front-facing camera as well as wrist camera on each
arm, and the mobile robots also have a rear-facing camera.
The action output of the π 0 . 7 model is applied on each robot
using a simple PD controller. For commanding end-effector
movement, we apply numerical inverse kinematics to convert
target end-effector poses into target joint positions.

IX. E XPERIMENTAL E VALUATION

In our experiments, we evaluate how well π 0 . 7 can leverage
diverse data sources to enable strong out-of-the-box per-
formance, broad generalization, and more effective transfer,
leveraging a variety of context modalities. Specifically, we
study how well π 0 . 7 can perform complex and dexterous tasks
out of the box, particularly in comparison with more special-
ized RL-finetuned models (Sec. IX-A ), evaluate its ability to
flexibly follow instructions to do a variety of different tasks
(Sec. IX-B ), study its transfer capabilities across embodiments
(Sec, IX-C ), and test its ability to compose skills in previously
unseen ways to do new tasks (Sec. IX-D ). Finally, we perform
controlled experiments to study how the performance of π 0 . 7
scales with increased task and context diversity in our robot
datasets (Sec. IX-E ).

A. Out-of-the-box performance on challenging tasks

π 0 . 7 achieves high performance on dexterous tasks without
task-specific post-training. In our first set of experiments, we
study how well π 0 . 7 can master dexterous tasks that were seen
in the training data, but where the goal is to perform these tasks
as robustly and efficiently as possible. This is surprisingly
difficult for prior robotic foundation models: often the best-
performing policies are fine-tuned for specific downstream
tasks, even if they use generalist pre-training [ 42 , 50 ]. We
aim to answer: can the general-purpose π 0 . 7 model match the
performance of task-specific fine-tuned models on a variety of
dexterous manipulation tasks?
We use the tasks shown in Fig. 6 . These include the espresso
making, box building, and laundry folding tasks that we
previously used to evaluate the RL-trained π ∗
0 . 6 models [ 50 ],
where we can directly compare the speed and robustness of
the single general-purpose π 0 . 7 model to the individual RL-
finetuned specialist π ∗
0 . 6 models. We also study a number
of other dexterous tasks, including some tasks from our
previous “Robot Olympics” experiments (making a peanut
butter sandwich, turning a shirt inside-out, and driving through
a door) and a number of additional dexterous tasks, such as
fully slicing up a zucchini, peeling a few fruits and vegetables
(zucchini, cucumbers, and carrots), and a long-horizon task
that involves replacing a trash bag in a trash can. We find
that π 0 . 7 achieves performance that is competitive with the
RL specialists used in the π ∗
0 . 6 release [ 50 ] for all of the
tasks considered in the paper directly out of the box (Fig. 6 ,
first row), and even outperform the specialists in throughput
in the difficult laundry and box building tasks. Additionally,
we compare π 0 . 7 to SFT specialists trained on top of π 0 . 6 for
a number of other dexterous tasks, and find that π 0 . 7 is again
able to closely match the performance of all specialist policies
(Fig. 6 , second row).
To understand how the training recipe of π 0 . 7 affects per-
formance, we additionally compare π 0 . 7 with two ablations
on the tasks from the π ∗
0 . 6 release: π 0 . 7 (no eval data), which
holds out all autonomous evaluation episodes from training
