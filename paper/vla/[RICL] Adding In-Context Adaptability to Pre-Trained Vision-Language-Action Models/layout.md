# Page 1
RICL : Adding In-Context Adaptability to Pre-Trained
Vision-Language-Action Models

Kaustubh Sridhar 1 , Souradeep Dutta 2 , Dinesh Jayaraman 1 , Insup Lee 1

1 University of Pennsylvania, 2 University of British Columbia
ksridhar@alumni.upenn.edu

Abstract: Multi-task “vision-language-action” (VLA) models have recently
demonstrated increasing promise as generalist foundation models for robotics,
achieving non-trivial performance out of the box on new tasks in new environments.
However, for such models to be truly useful, an end user must have easy means to
teach them to improve. For language and vision models, the emergent ability to
perform in-context learning (ICL) has proven to be a versatile and highly useful
interface to easily teach new tasks with no parameter finetuning. Unfortunately,
VLAs pre-trained with imitation learning objectives do not naturally acquire ICL
abilities. In this paper, we demonstrate that, with the right finetuning recipe and a
small robot demonstration dataset, it is possible to inject in-context adaptability
post hoc into such a VLA. After retraining for in-context learning ( RICL ), our sys-
tem permits an end user to provide a small number (10-20) of demonstrations for
a new task. RICL then fetches the most relevant portions of those demonstrations
into the VLA context to exploit ICL, performing the new task and boosting task
performance. We apply RICL to inject ICL into the π 0 -FAST VLA, and show that it
permits large in-context improvements for a variety of new manipulation tasks with
only 20 demonstrations per task, without any parameter updates. When parameter
updates on the target task demonstrations is possible, RICL finetuning further boosts
performance. We release code and model weights for RICL - π 0 -FAST alongside
the paper to enable, for the first time, a simple in-context learning interface for new
manipulation tasks 1 .

Keywords: Vision-Language-Action (VLA) models, In-Context Learning (ICL),
Retrieval-Augmenetd Generation (RAG)

1
Introduction

Robot learning is undergoing a transformative moment with the emergence of the first generation of
general-purpose Vision-Language-Action (VLA) models, capable of performing a wide spectrum of
robotic tasks — a development with profound practical implications. Such models [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ]
could address persistent challenges in robotics, including data scarcity, robustness, and generalization.

A natural point of comparison for these VLAs is large language models (LLMs). One important
factor in the widespread adoption of LLMs today is that they appear to be able to quickly learn new
tasks, simply through providing a few examples as “context” alongside the query, with no parameter
tuning. This capability, called in-context learning (ICL) [ 9 ], emerges naturally in LLMs pre-trained
for next-token prediction, due to the nature of web text data. Even better, one need not even manually
provide these few examples. Instead, a retrieval mechanism could automatically fetch the most
relevant data from a large corpus and place them into the LLM context. This retrieval-augmented
generation (RAG) mechanism is widely adopted as a versatile interface to improve a base LLM [ 10 ].

1 Website: https://ricl-vla.github.io

9th Conference on Robot Learning (CoRL 2025), Seoul, Korea.

arXiv:2508.02062v1  [cs.RO]  4 Aug 2025

# Page 2
(a) Task: "pick up the poke ball and put it in the tray". π 0 -FAST-DROID [L] picks up the distractor (duck) instead (language grounding issue).
RICL - π 0 -FAST-DROID [R] actually moves the unseen object (pokeball) with only RAG and ICL.

(b) Task: "pick up the bagel and put it in the toaster". π 0 -FAST-DROID [L] aimlessly wanders and cannot figure out the grasp or motion
(adaptation issue). RICL - π 0 -FAST-DROID [R] almost completes the task (only with RAG and ICL) but drops the unseen object (bagel) at the
end of the novel motion–a combination of an unfamiliar grasp at its rim, its unique initial vertical position, and the twist-and-lift motion.

(c) Task: "move the idli plate to the sink". In π 0 -FAST-DROID’s best test rollout shown here, it still struggles with the grasp and motion for
this novel object (adaptation issue) or moves the apple (distractor) instead (language grounding issue). RICL - π 0 -FAST-DROID can perform the
novel motion (gripper in depressions) on the unseen object (idli plate) in this new scene with new camera positions/orientations and with lighting
changes (which is different from the table where the priming demonstrations were collected).

(d) Task: "use the squeegee to clean the counter". π 0 -FAST-DROID oscillates without success. It gets close, but it cannot figure out the grasp
or the motion (adaptation issue). RICL - π 0 -FAST-DROID adapts to novel object (squeegee) and motion (part lifting, part dragging) in the new
scene. Notice the pellets dropping into sink showing contact with the surface.

(e) Task: "push the lever on the toaster". π 0 -FAST-DROID [L] aimlessly wanders. It cannot figure out the precise location or the grasp
(adaptation to a variant of training object issue). RICL - π 0 -FAST-DROID [R] completes the precise task, only with RAG and ICL, and with
elicited latent actions not in the retrieval data (more information in Section 6 ). This long-tail task appears infrequently in the DROID dataset.

(f) Task: "open the door of the bottom shelf". π 0 -FAST-DROID [L] aimlessly wanders. It cannot figure out the motion to adjust or avoid the
top door acting as obstacle (adaptation issue). RICL - π 0 -FAST-DROID [R] completes the task, with this variant of a seen object (this particular
shelf) and novel motion (precise door opening adjusting for the obtructing top shelf), only with RAG+ICL. This is also a long-tail task.

Figure 1: Qualitative comparison between π 0 -FAST-DROID [L] and RICL - π 0 -FAST-DROID [R] , with 20
task specific demonstrations for RAG and ICL, on new tasks, including novel objects, motions, and scenes.
Additional comparisons can be found in Figure 8 in Appendix D .

Unfortunately, VLAs are trained with imitation learning objectives on relatively narrow demonstration
datasets. As one would expect, this does not naturally produce any in-context learning abilities. This
means that “improving” a pre-trained VLA today means tuning its parameters on a new demonstration
dataset [ 2 ]. To make it possible for an end user to easily improve a VLA with no parameter tuning,
we ask the following question:

How can we inject in-context learning abilities into a pre-trained VLA?

2

# Page 3
Once this is done, we should be able to painlessly boost the VLA’s performance on any task, including
handling unseen objects, novel motions, and new scenes that don’t exist in the VLA training data.

Our solution is to r etrain for i n- c ontext l earning ( RICL , pronounced “rickle”). RICL borrows from
prior recipes [ 11 , 12 ] to train generalist models for in-context learning and RAG. In particular,
while REGENT [ 12 ] trained generalist game-playing agents from scratch, RICL uses this approach
to instead post-train an off-the-shelf VLA priming it to use its context effectively. The resulting
RICL -VLA can improve the base VLA’s performance for any target tasks without a single gradient
update, instead adapting purely through retrieval-augmentation and in-context learning, with only
10-20 demonstrations in its retrieval buffer. We demonstrate this on various manipulation tasks
depicted in Figure 1 where a state-of-the-art (SOTA) VLA fails but our RICL -VLA adapts simply via
RAG and ICL. We further find that it is possible to get even better task performance by “finetuning
like you pretrain” [ 13 ]: we optimize the RICL objective on the same demonstrations as used above
for ICL, and get large performance boosts.

2
Related Work

Training VLAs and multi-task generalist agents: There has been a spate of work in recent years
on training multi-task agents in simulated settings like games [ 14 , 12 , 15 , 16 , 17 , 18 ] and in recent
months on training VLAs for robotics [ 1 , 2 , 3 , 4 , 6 , 7 , 5 , 8 ]. To our knowledge, there are only three
prior attempts to train general agents with in-context learning (ICL) capabilities [ 16 , 12 , 18 ], none
for general-purpose robotics. This is the focus of RICL : we show how to post-train a pre-trained VLA
to effectively learn in-context.

In-context learning for robotics: The in-context learning abilities of large language models (LLMs)
and vision-language models (VLMs) have already been found to be very useful in robotics: with
suitable representations (such as keypoints or code), these models can in-context learn imitation
policies [ 19 , 20 , 21 , 22 ] or value functions [ 23 , 24 ]. But, using LLMs and VLMs requires these
methods to run completely (or mostly) open-loop [ 19 , 20 , 21 ] and using only LLMs makes them lose
significant visual information [19, 21]. Both these drawbacks affect their ability to adapt.

3
Background on ICL, RAG, and π 0 VLAs

In-Context Learning (ICL) is the property of sequence to sequence models that allows them to
predict an output (such as an action ˆ a t ) for a new input (such as state s t ) given a few examples of
input-output pairs in the context (such as state-action pairs { ( s ′ , a ′ ) , ( s ′′ , a ′′ ) , ... } ). The input to the
model is the concatenation of the context and the new input.

Retrieval-Augmented Generation (RAG) refers to the strategy commonly used to help LLMs
predict an answer for a query. This is achieved by obtaining the information necessary for the answer
by searching through a dataset and then placing said information in the context of an LLM.

Of the three methods for embodied agents mentioned in Section 2 that learn to in-context learn,
MTT [ 16 ] and ICRT [ 18 ] place a few complete demonstrations in their context while REGENT
[ 12 ] retrieves select states and actions (from the same few complete demonstrations) to place in its
context. The RAG+ICL method employed by REGENT outperforms the former ICL method across
held-out games and held-out simulated robotics tasks. REGENT demonstrates that combining the
two i.e. retrieving specific examples to place into context from a demonstration dataset offers a
computationally less intensive, higher performing alternative to directly placing demonstrations in
context. We adopt this idea in RICL .

π 0 -FAST [ 1 ] is a state-of-the-art auto regressive VLA model that takes images, language instruction,
and proprioceptive state as input and predicts actions. It can be deployed (in a variety of scenes)
zero-shot on robot embodiments that are a part of its training data and few-shot after finetuning on
new robot embodiments. π 0 -FAST-DROID [ 1 ] is a VLA that was created by further finetuning
π 0 -FAST on the large DROID dataset [ 25 ]. The DROID dataset was collected with the Franka
DROID platform, shown in Figure 3 , across many research labs (additional details in Appendix B ).

3

# Page 4
Figure 2: Architecture of RICL -VLAs, specifically that of RICL - π 0 -FAST.

4
RICL and creating in-context learning capable RICL -VLAs

This work aims to combine the best of both worlds from Section 3 – i.e. , it aims to quickly convert a
VLA that can be generally deployed like π 0 -FAST-DROID into one that also has in-context adaptation
capabilities like REGENT. Once an in-context learning capable VLA has been created, “teaching” it
to improve its performance on a new task is as simple as downloading the model, collecting a few
demonstrations, and providing them as a retrieval dataset. Then, the in-context learning capable VLA
should instantly have much better success rates on this new task than the baseline VLA.

Re-training for In-Context Learning ( RICL ): RICL enables the aforementioned conversion of a
pre-trained VLA to a in-context learning capable-VLA (that we call a RICL -VLA). In RICL , a VLA
is post-trained on sequences of query images/states and many images, states, actions, and action
obtained from the retrieval demonstrations as depicted in Figure 2 . The query information at time
t consists of three images (top image t , side image t , wrist image t ), a language prompt describing
the task, and proprioceptive state s t . We use the term "query" following terminology from RAG for
LLMs. The retrieved neighbors also consist of three images, the same text prompt, proprioceptive
state, and action chunk ( i.e. an array of actions over many time steps). The retrieved information is
placed in the context with the closest neighbor (to the query) on the left and farther away neighbors
towards the right. The closest neighbor’s images, states, and actions is represented with a single ′ ,
the second closest with a double ′′ and so on (see Figure 2 ). This finetuning utilizes a few "priming"
demonstrations. These demonstrations are called "priming" demonstrations since their role is to prime
the VLA to use its context effectively. Further, as depicted in Figure 2 , only the LLM is finetuned
during RICL while the image encoder is kept frozen. RICL -VLAs perform retrieval by embedding
only the top query image with an off-the-shelf DINO-v2 [ 26 ] image encoder and comparing it with
the embeddings of top images of the demos in the retrieval buffer with an ℓ 2 distance metric.

Like REGENT [ 12 ], the predicted action ˆ a t involves a distance-weighted interpolation between the
action tokens of the closest retrieved action a ′ and the final output of the large language models. We
refer to this as the action interpolation layer and depict it within the green box above the LLM in
Figure 2 . This distance corresponds to the distance between the DINO embeddings of the query
top image and the closest retrieved top image. The action interpolation layer assumes a maximum
number of action tokens numbering N act and combines the one-hot encoding of each token of a ′ with
the corresponding token output by the LLM π θ ( retrieved, query ) as follows:

π θ
RICL -VLA ( retrieved, query ) = e − λd one-hot ( a ′ ) + (1 − e − λd ) σ ( π θ ( retrieved, query ))
(1)

where σ represents the Softmax function and d denotes the ℓ 2 distance between the DINO embeddings
of top image t and its nearest neighbor top image ′ . The RICL -VLA performs the above interpolation
for each of the N act tokens. These tokens are then detokenized by the FAST tokenizer to obtain an
action chunk that can be executed on the robot.

4

# Page 5
Unlike the process to train REGENT [ 12 ], RICL on the other hand, only predicts and minimizes the
cross-entropy loss over the query (prompt, s t ) tuple and predicted action chunk during training
whereas REGENT [12] predicted and minimized the loss over all retrieved and query actions.

The RICL -VLA, after having been primed to use its context in RICL , can now be deployed on a target
task, which can include unseen objects and novel motions, with just a few task-specific demonstrations
for RAG and ICL, and without any further training on those demonstrations.

Further finetuning of a RICL -VLA: If a RICL -VLA is further finetuned on those target task
demonstration–that it was only retrieving from and throwing into its context previously–it can
significantly improve its performance, outperforming a VLA directly vanilla finetuned on those
unseen task demonstrations. This finetuning process on the few task-specific demonstrations is
done exactly like RICL – i.e. , a retrieval-augmented finetune of the RICL -VLA (which has the action
interpolation layer) with the same objective of minimizing the cross-entropy loss over the query
(prompt, s t ) tuple and predicted action chunk. At deployment, the finetuned RICL -VLA still retrieves
from the same data that it is finetuned with, i.e. no extra data (hyperparameters in Appendix C ).

5
Experimental Setup

Training ( a.k.a. , Priming) data for RICL : We collect 20 demonstrations with the Franka DROID
platform (see Figure 3 ), randomizing the initial position of the primary object, in 20 pick and place
tasks (total 400 demonstrations). The exact list of tasks, an image of all the objects used, and
more details about the platform are in Appendix C . We perform RICL – starting from the weights
of π 0 -FAST-DROID, fully finetuning its LLM, and keeping its image encoder frozen– with the
hyperparameters detailed in Appendix C . We call the model obtained after three epochs of training as
RICL - π 0 -FAST-DROID.

Figure 3: [LEFT] Our Franka DROID setup, annotated.
[RIGHT] Franka DROID, including the top camera and
right camera, moved to a new scene (kitchen sink).

In each task, for each demonstration, and for
each state in that demonstration, we use that
state as the query and retrieve four neighbors
from the other 19 demonstrations to create train-
ing input sequences. In this way, we end up
training the model in the same way we would
deploy the model. We collect data as detailed
above since such a dataset is not available.

Evaluation tasks with unseen objects, novel
motions, different scenes: We evaluate all
methods on the following tasks, which involve a task-relevant unseen object and/or a completely
novel motion in two different scenes (tabletop and kitchen sink). “Unseen” here means that these
objects and motions are not in either the RICL priming data or the DROID data [ 25 ]. We checked
the latter by searching over the DROID dataset’s language annotations (and some recordings when
language was not adequate). First, we start with a simple task that primarily tests language grounding.

• ( pokeball ) "pick up the pokeball and put it in the tray": has an unseen object (pokeball) with a
couple of distractors on the table.

Next, we test on simple tasks that test both language grounding and adaptation to novel motions.

• ( idliplate ) "move the idli plate to the left": has an unseen object (idli plate) with a apple
(distractor) sitting on the plate. It also requires the robot to do an unfamiliar grasp to move the
uniquely shaped plate with depressions to the right.

• ( squeegee ) "move the squeegee to the right and try to drag it": has an unseen object (squeegee).
It also requires the robot to do a novel motion of slightly lifting the handle of the squeegee while
keeping its rubber on the table to drag it across the table.

We then test on versions of the above two simple tasks in a new scene (kitchen sink area) with new
camera positions/orientations (no calibration necessary), new lighting, and new distractors.

5

# Page 6
• ( sink-idliplate ) "move the idli plate to the sink": all of the previous challenges & a new scene.

• ( sink-squeegee ) "use the squeegee to clean the counter": all of the challenges of the previous
task with pellets to be cleaned up on the counter in the new scene.

We also test on the long-tail of the training task distribution. These tasks consist of variations of
objects in the DROID dataset. These object classes appear infrequently in the dataset.

• ( toaster ) "push the lever on the toaster": has a different version of an object (a particular toaster
brand) in the DROID dataset. This task tests the long tail of the training task distribution. It also
requires a precise placement and movement to push the lever down.

• ( door ) "open the door of the bottom shelf": has a different version of an object (a particular shelf)
in the DROID dataset. This task also is in the long tail. It requires a novel and precise motion that
can handle the large top shelf acting as an obstacle when reaching the bottom door’s handle.

Finally, we test on a longer horizon task that is a composition of many simple tasks.

• ( bagel ) "pick up the bagel and put it in the toaster": has an unseen object (bagel). It also requires
the robot to do a composite novel motion–an unfamiliar grasp on the edge of the bagel, the
twist-and-lift motion, and placing in the slot.

Retrieval data in evaluation tasks: In all the evaluation tasks, we collect 20 demonstrations,
randomizing the initial position of the primary object, for RICL - π 0 -FAST-DROID to retrieve from
and throw in its context to adapt to the task.

Evaluation metrics and comprehensive randomization: We collect 10 test rollouts for all methods
on all evaluation tasks with randomly chosen initial positions and orientations in each rollout. We
set these intial positions and orientations all across the table. The distractors, if any, are kept
approximately in the same region of the table but they are also not fixed in place. We calculate the
success of the full tasks, in addition to tracking intermediate checkpoints for a better understanding
of the progress of each method on each task.

Baselines and ablations: We compare RICL - π 0 -FAST-DROID with vanilla π 0 -FAST-DROID on
all tasks. We also compare with ’Retrieve and play’, a 1 nearest neighbor baseline from [ 12 ], which
simply outputs the first retrieved action a ′ . We also compare with a trained-from-scratch Diffusion
Policy baseline. We perform these two comparisons on the simpler evaluation tasks ( pokeball ,
idliplate , squeegee ). Upon observing their low success rates, we leave them out for the more
complex tasks. In the tasks performed in a new scene ( sink-idliplate , sink-squeegee ), we aim
to test RICL ’s ability to retain π 0 ’s helpful scene-generalization capabilities while adapting in-context
to a new task and hence only test these two methods. We also ablate the number of demonstrations
used by each method in the idliplate task.

Further finetuning: We further finetune RICL - π 0 -FAST-DROID on each evaluation task on the 20
demonstrations collected for retrieval. For comparison, we also further (vanilla) finetune π 0 -FAST-
DROID on these same 20 demonstrations in each task.

6
Experimental Evaluation

Generalization to unseen objects and novel motions and new scenes: We plot the quantitative
results across tasks and methods in Figure 4 . We observe that RICL - π 0 -FAST-DROID outperforms
π 0 -FAST-DROID, especially in earlier checkpoints of the task, but also in overall task success. In
aggregate, across all evaluated tasks, π 0 -FAST-DROID obtains a complete task success rate of 2.5%
and a checkpoint completion rate of up to 21.25%. On the other hand, RICL - π 0 -FAST-DROID obtains
a significantly improved complete task success rate of 31.25% and a checkpoint completion rate of
up to 83.75%.

We particularly note that RICL - π 0 -FAST-DROID has significantly improved language grounding
to move towards the unseen objects just based on contextual information. More importantly, RICL -

6

# Page 7
Figure 4: Success rates of 10 test rollouts from various methods across various tasks represented by stacked bar
plots. The lowest bar (dark blue) in each stacked column represents full task success rate, and other bars are the
success rates for reaching earlier waypoints. Gray regions represent the fraction of runs that did not even reach
the first waypoint for the task. We note that π 0 refers to π 0 -FAST-DROID and RICL to RICL - π 0 -FAST-DROID
in the plots. We also plot the performance of various methods vs the number of demonstration in the idliplate
task on the bottom right.

π 0 -FAST-DROID also overcomes the adaptation issue faced by π 0 -FAST-DROID. Where π 0 -FAST-
DROID struggles with grasps and motions, RICL - π 0 -FAST-DROID demonstrates the ability to
infer novel grasps and motions from its context as evidenced in six tasks (both squeegee tasks,
sink-idliplate , bagel , toaster , and door ). We plot the qualitative results depicting key test
rollouts and behaviors of π 0 -FAST-DROID and RICL - π 0 -FAST-DROID in Figure 1 and Figure 8 . We
also provide side-by-side comparisons and detailed explanations of rollouts in the same Figures.

Unexpectedly, we observe in some tasks that RICL - π 0 -FAST-DROID seems to predict and execute
action sequences that are not like the motions in the retrieval dataset. For example, in idliplate ,
RICL - π 0 -FAST-DROID moves to the left of the idli plate, closes its gripper and pushes the plate to
the right. But, all 20 demonstrations in the retrieval buffer were collected with the motion of dipping
the gripper into the depressions and moving the plate to the right. Hence, RICL - π 0 -FAST-DROID
has seemingly elicited latent actions or knowledge to accomplish this task (also seen in toaster ).

Significantly improved performance after further finetuning the RICL -VLA: We observe a
significant improvement in performance after further finetuning RICL - π 0 -FAST-DROID on each
task’s 20 demonstrations. In aggregate, across all evaluated tasks, π 0 -FAST-DROID-finetuned
obtains a complete task success rate of 31.67%, while RICL - π 0 -FAST-DROID-finetuned obtains a
complete task success rate of 61.67%. In fact, we not only see that the RICL -VLA with finetuning is
significantly better than the base VLA with finetuning (at almost double the aggregate performance).

7

# Page 8
(a) Task: "pick up the poke ball and put it in the tray"

(b) Task: "move the idli plate to the right"

(c) Task: "open the door of the bottom shelf"
Figure 5: Qualitative visualization of the reactivity and robustness of RICL - π 0 -FAST-DROID-finetuned on 20
task-specific demonstrations in a dynamic test rollout. In the above, a human randomly perturbs and displaces
the primary object during the test rollout. Additional results can be found in Figure 9 in Appendix D .

But we also observe comparable performance in complete task success rate, in aggregate, between
the RICL -VLA (at 31.25%), which only uses RAG and ICL, and the base VLA with finetuning on the
target task data (at 31.67%).

We hypothesize that further finetuning our VLA is significantly better than doing so with the base
VLA simply because our VLA can use all of its capacity to focus on interpolating amongst the
retrieved images, states, and actions to predict a new action and does not have to memorize any data.
This is in line with the observed parameter-efficiency & performance advantages of RAG LLMs [ 27 ].

We qualitatively demonstrate the reactivity and robustness of RICL - π 0 -FAST-DROID by randomly
perturbing and displacing objects during a test rollout as shown in Figure 5 and Figure 9 .

Ablating the number of retrieval/finetuning demonstrations: We plot the ablation results, ablating
the number of demonstrations used in the retrieval buffer or for finetuning, for idliplate in the
bottom right of Figure 1 . We found that too few demonstrations (such as 5) results in RICL - π 0 -
FAST-DROID starting to behave like π 0 -FAST-DROID to the extent where in one test rollout, it
too moves the apple instead of the idli plate. This demonstrates the requirement for atleast 10
demonstrations. Also, from this figure, we conclude that more retrieval demonstrations help RICL -
π 0 -FAST-DROID improve, towards catching up with π 0 -FAST-DROID-finetuned. We also see that
RICL - π 0 -FAST-DROID-finetuned is significantly better than π 0 -FAST-DROID-finetuned at every
number of demonstrations.

No loss-of-capabilities results: One might wonder: does RICL post-training come at the cost of
losing the ability in the base VLA to perform without any retrieval data? To evaluate this, we test
RICL - π 0 -FAST-DROID with randomly chosen priming demonstration in the retrieval buffer, rather
than any meaningful task-specific demonstrations, on three tasks: "move the can to the tray", "pick
up the marker and put it in the tray", and "place the apple next to the can". It obtains an 80% success
rate, just like π 0 -FAST-DROID, demonstrating that RICL has not led to any loss of capabilities.

7
Conclusions and Future Work

We have demonstrated how RICL can be used to convert a VLA to a RICL -VLA that can use its
context to adapt to completely new tasks, including unseen objects and novel motions, with just RAG
and ICL. We found the RICL -VLA to even have comparable performance, in aggregate, with the base
VLA finetuned on target task data. We have also demonstrated a significant boost in performance
when a RICL -VLA is further finetuned on task-specific demonstrations. In future work, we believe
that scaling up RICL in both the number of priming demonstrations and parameter size will further
boost the performance of the in-context learning capable RICL -VLA.

8
