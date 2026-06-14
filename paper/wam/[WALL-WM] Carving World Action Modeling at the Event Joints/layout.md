# Page 1
WALL-WM: Carving World Action Modeling
at the Event Joints

X Square Robot Team

WALL-WM is a World Action Model that shifts video-action learning from chunk-centric optimization
to event-grounded Vision-Language-Action pretraining, using semantically coherent action events as
the atomic unit of learning. Existing WAMs commonly initialize from multimodal or video foundation
models and then optimize fixed-length action chunks conditioned directly on the current observation
and instruction. Although convenient, this chunk-centric formulation creates a fundamental granularity
mismatch. Language describes semantic goals and events, vision evolves through continuous scene
dynamics, and actions operate at control-level timescales; forcing all three into the same fixed-length
prediction window turns VLA training into short-horizon correlation fitting. This not only underuses
the pretrained visual-semantic prior, but can actively overwrite it with chunk-specific action shortcuts,
weakening compositionality and long-horizon generalization. WALL-WM addresses this mismatch by
organizing both supervision and data around semantic events. Specifically, it pairs event-grounded
VLA pretraining with a data ecosystem built from event-level captions and cluster-balanced sampling,
enabling scalable learning over diverse behaviors, scenes, and task structures. From the same event-
pretrained backbone, WALL-WM supports two complementary inference modes. The event mode
consumes next-event descriptions and enables variable-length execution chunks, while the unified
mode uses a VLM with Staircase Decoding to condition conventional fixed-length chunk inference
while preserving a gradient-continuous VLA path. Together with Muon-optimizer-based large-scale
pretraining infrastructure, WALL-WM provides a practical scale-up recipe for general-purpose WAMs.
Experiments show that WALL-WM generalizes broadly across language, scenes, and tasks, achieving
state-of-the-art performance in large-scale real-world generalization evaluation.

“Carve nature at its joints.”

— after Plato, Phaedrus 265e

Date: June 2026
Code: https://github.com/X-Square-Robot/wall-x

1
Introduction

Recent progress in embodied foundation models ( 97 , 41 , 9 , 36 , 8 , 14 , 17 ) has increasingly been driven by large-
scale priors inherited from multimodal understanding models and video foundation models ( 20 , 37 , 13 , 95 ). In
this report, we use VLA to broadly denote embodied foundation models that predict actions from visual-language
inputs, and WAMs to denote world-action models that explicitly couple future observation modeling with action
prediction.

Most existing embodied foundation models adapt these priors by predicting fixed-length action chunks from
the current observation and language instruction. This chunk-centric formulation is effective and convenient,
but it hides a structural mismatch: it cuts embodied dynamics by an external clock, while language, vision, and
action evolve at different semantic and temporal scales. Language specifies goals and events; vision evolves
through continuous scene dynamics; action operates at control-level time scales and is sensitive to contact,
timing, and small perturbations. Thus, adapting pretrained visual-semantic priors to embodied control is not
merely a fine-tuning problem. It is first a question of where to place the atomic unit of video-action learning.

This distinction is the starting point of WALL-WM:

Fixed chunks cut by clock; semantic events cut by embodied dynamics.

arXiv:2606.01955v1  [cs.RO]  1 Jun 2026

# Page 2
Figure 1 Conceptual illustration of modality hierarchy and WALL-WM’s general performance. Left: a stylized
alignment landscape over semantic abstraction and spatial-temporal precision. Text provides coarse semantic alignment,
vision provides denser spatial-temporal grounding, and action requires fine-grained contact-sensitive precision. Tactile-force
input, when available, is treated as an optional contact-rich signal rather than a required modality. Right: WALL-WM shows
clear advantages across manipulation-task performance and video-generation metrics, indicating that event-grounded
pretraining improves both executable control and future-observation modeling. Bottom: representative real-robot task
snapshots provide illustrative examples of the physical tasks.

The rest of the argument follows from this mismatch. The central challenge is not multimodal fusion in the usual
sense, but geometry-preserving alignment . Text, vision, and action do not share the same notion of neighborhood:
semantically similar instructions may induce different visual trajectories, while visually similar states may
require incompatible control responses. As illustrated in Fig. 1 , text provides coarse semantic alignment, vision
supplies denser spatial-temporal grounding, and action requires the finest local precision. A useful WAM must
connect these modalities without flattening them into a single undifferentiated embedding space.

Video is the natural scaffold between language and action. Internet-scale video pretraining captures rich visual
dynamics that would otherwise have to be learned from embodied interaction alone( 37 , 20 ). Video is also
semantically structured enough to align with language at event boundaries, yet temporally dense enough to
expose the timing, transitions, and state changes needed for action execution. Future video shares causal
temporal structure with action, making visual-to-action grounding and inverse dynamics possible ( 20 , 43 ). In
this sense, video offers an embodiment-light bridge from high-level semantic intent to low-level execution.

Turning this scaffold into a WAM is not a short adaptation stage. It is a prior-preserving lift : the model must
inherit the semantic and temporal structure learned from large-scale video, while acquiring the controllability,
contact sensitivity, and causal grounding required for embodied action. Simply appending an action decoder can
attach actions to a visual prior, but it does not determine the unit at which the prior should become executable.
Without such a unit, joint optimization can collapse toward the most data-rich modality or overwrite useful
visual-semantic structure with short-horizon action correlations.

This lift imposes two requirements. First, a WAM must preserve the video prior : video generation models
favor semantic invariance, visual plausibility, and temporal smoothness, whereas embodied control requires
sensitivity to action-induced divergence and contact transitions ( 86 , 44 ). Second, a WAM must provide temporal
grounding : language instructions often describe global tasks or semantic events, while observations and actions

2

# Page 3
unfold over many frames and control steps. Fixed-length chunks are poorly matched to both requirements.
They can be too short to contain a complete semantic event, yet too long to preserve clean causal separation
between context and prediction target.

The same reasoning clarifies what must be lifted from a video prior into a general-purpose WAM:

T1. Reasoning: convert global instructions and task progress into event-structured intent.

T2. Visual prediction: preserve the caption-to-video inductive bias while making future observations control-
lable by executable events.

T3. Fine manipulation: expose timing, contact transitions, and local state changes required for action
execution.

These layers cannot be obtained by stacking a VLM, a video model, and an action head as independent modules.
They require a training regime whose alignment unit is simultaneously meaningful to language, visible in
video, and executable through action.

This gives three design principles for the alignment unit:

• Geometry preservation: connect language, video, and action without collapsing their native structures
into one shared space.

• Prior preservation: remain compatible with the caption-to-video structure inherited from video founda-
tion models.

• Executable causality: provide a prediction target with clear temporal support, while allowing duration
to follow the task rather than a fixed clock.

These principles rule out the fixed-length action chunk as the fundamental unit. A chunk is convenient for
batching and deployment, but it is not a natural object shared by language, video, and action. It may cut
through the middle of a semantic behavior, merge multiple behaviors into one target, or require historical
context merely to determine what the chunk is supposed to mean.

WALL-WM therefore replaces the fixed-length chunk with an action-grounded semantic event : a temporally
coherent segment of executable behavior, such as reaching, grasping, lifting, moving, or placing, that is
expressible in language, observable in video, and realizable through action. Unlike a fixed temporal chunk,
which follows an external clock, an action-grounded semantic event begins and ends when the underlying
executable behavior changes. It therefore satisfies the principles above: language names the event, video
grounds its spatial-temporal evolution, and action realizes it through control.

WALL-WM instantiates this principle through event-grounded WAM pretraining. Event captions are paired with
corresponding video and action segments, and the model is trained to denoise future video and action over
event-aligned intervals. This does not merely use events as auxiliary conditions; it grounds the training problem
itself at the event level. The result is a prior-preserving route from video foundation models to executable
world-action models.

At inference time, the same event-pretrained backbone supports two complementary modes:

Event mode. The system rolls out in event space. A VLM, human, or agent proposes the next-event description,
and WALL-WM executes the corresponding variable-length video-action segment before observing the
next state. This mode follows the natural duration of the task rather than a fixed control horizon.

Unified mode. Conventional fixed-length chunk inference is retained, but the chunk is no longer conditioned
on a raw global instruction alone. A VLM with Staircase Decoding supplies event-structured latent
reasoning over task progress, producing a latent event representation that guides the next local chunk
while preserving a gradient-continuous VLA path.

Fig. 2 summarizes these two schemes, and Sec. 3 provides the detailed formulation.

This report presents WALL-WM as both a model instance and a training roadmap for prior-preserving scale-up
of WAMs. Its main components are:

3

# Page 4
Event Caption

Figure 2 Next-event training and equilong-chunk schemes. In prior-aligned training, the event caption, event video,
and event action describe the same semantic interval, giving a well-posed caption-to-video/action target. In equilong-chunk
mode, a global instruction alone is ambiguous for a local chunk; adding history windows restores a well-posed next-chunk
prediction problem.

• Event-grounded WAM pretraining. WALL-WM treats action-grounded semantic events as the atomic
training unit, pairs them with event captions, and trains a video-action denoiser that preserves the
inherited video prior while turning event-aligned visual evidence into executable action.

• Two inference modes from a single event-pretrained backbone. Event mode performs event-space
rollout with variable-length executable segments. Unified mode supports fixed-length chunk prediction
by using VLM-based Staircase Decoding to generate event-structured latent reasoning while maintaining
a gradient-continuous VLA path.

• Scale-up infrastructure for event-grounded embodied modeling. WALL-WM combines an event-
grounded data ecosystem, cluster-balanced sampling, and Muon-optimizer-based large-scale pretraining
infrastructure to support scalable WAM training across diverse behaviors, scenes, and tasks.

WALL-WM demonstrates broad generalization across language instructions, scenes, and tasks, achieving strong
performance in large-scale real-world generalization evaluation. Overall, WALL-WM connects text, video, and
action through action-grounded semantic events. We position WALL-WM less as a short adaptation of a video
foundation model and more as a prior-preserving scale-up methodology for the next generation of embodied
foundation models.

2
Related Work

2.1
Vision-Language-Action Models

A growing line of embodied foundation models, commonly referred to as Vision-Language-Action (VLA) policies,
extends pretrained vision-language models with action interfaces that map visual observations and natural-

4

# Page 5
language instructions to executable motor commands ( 97 , 41 , 70 , 9 , 36 , 8 , 14 , 77 , 68 ). By inheriting web-scale
semantic priors from VLM pretraining, these models exhibit strong generalization across objects, scenes, and
language instructions, and offer a unified alternative to modular perception–planning–control pipelines. Recent
work has explored a broad range of design choices within this observation-to-action paradigm, including efficient
action tokenization and chunking for long-horizon control ( 90 , 60 , 41 ), diffusion- or flow-matching-based
action experts ( 16 , 45 , 74 , 9 ), lightweight and data-efficient backbones ( 75 ), latent-action pretraining and
cross-embodiment routing ( 82 , 11 ), knowledge-insulated fine-tuning that preserves pretrained priors during
adaptation ( 19 ), spatially enhanced or 3D-aware policy representations ( 61 , 56 ), and visual chain-of-thought
reasoning interleaved with action generation ( 87 , 89 ). Despite their broad semantic capabilities, existing
VLAs still face systematic limitations. First, their underlying VLMs are pretrained predominantly on static
image–text data ( 97 , 9 ), so even large-scale supervised fine-tuning on teleoperation datasets ( 41 , 8 ) primarily
learns action imitation rather than an explicit model of how the physical world evolves under intervention.
Second, most VLAs formulate control as a reactive observation-to-action mapping without action-conditioned
future prediction or an explicit temporal-dynamics prior ( 45 , 14 ). Third, generalization to genuinely novel
motions, skills, embodiments, or environments often still requires substantial task- or robot-specific adaptation
data ( 36 , 17 ). These limitations have motivated a parallel line of research that grounds policies in generative
models of how the world evolves, which we review next.

2.2
Generative Embodied World Models

Generative world models have recently emerged as a promising paradigm for embodied AI, where agents learn to
predict future states of the physical world and use such predictions for planning and control. Early model-based
reinforcement learning methods such as PlaNet, Dreamer, and DreamerV3 ( 28 , 27 , 29 ) learn compact predictive
states for control, while the JEPA family, including V-JEPA and LeWorldModel ( 2 , 54 ), shows that forecasting
in feature space can capture meaningful temporal and physical structure without explicit pixel reconstruction.
More recent embodied world models further leverage generative video and world-action modeling objectives
to learn robot–environment dynamics from large-scale heterogeneous data. These models use predicted
future frames or intermediate visual representations as plans, from which executable actions are recovered
through inverse-dynamics models ( 20 ), intermediate-feature action decoders ( 59 ), dense correspondence ( 43 ),
planning-oriented trajectory decoding ( 21 , 94 ), or synthesized demonstrations transferred from human videos
and novel scenes ( 6 , 5 , 15 , 37 , 51 ). Recent work further suggests that video generators encode useful 3D and
interaction priors ( 32 , 42 , 93 , 78 , 46 ), motivating their use as embodied dynamics models. Along this direction,
LaDi-WM, AdaWorld, Motus, LDA-1B, and MotuBrain ( 33 , 23 , 7 , 53 , 69 ) explore latent diffusion, structured
visual forecasting, latent-action conditioning, and unified video–action modeling, while LingBot-VA ( 44 ) and
related unified denoising architectures ( 26 , 22 , 95 , 13 , 83 , 76 , 12 , 88 , 47 ) jointly model future prediction
and action generation. More recent methods such as Fast-WAM ( 86 ) further improve inference efficiency by
avoiding explicit video decoding or future imagination at test time. Overall, these advances indicate that
explicit future-state modeling can improve sample efficiency, robustness, and generalization.

2.3
Latent Reasoning

A complementary line of work routes chain-of-thought (CoT) reasoning through compact latent representations
rather than emitting full textual reasoning traces, motivated by inference efficiency and the ability to explore
semantic-level reasoning trajectories beyond discrete token sequences ( 30 , 24 , 38 , 39 , 50 , 92 ). LaDiR ( 38 )
encodes reasoning steps into blocks of continuous “thought tokens” via a VAE and refines them with a latent
diffusion model; its reinforcement-learning extension ( 39 ) further optimizes latent trajectories to preserve
solution diversity. Recent VLA methods further extend this idea to physically grounded latent reasoning:
LaST 0 ( 49 ) introduces a latent spatio-temporal CoT that captures future visual dynamics, 3D structure, and
proprioceptive states for robotic manipulation; LaST-VLA ( 52 ) distills geometric constraints and world-model
foresight into latent spatio-temporal representations for autonomous driving; and LaRA-VLA ( 3 ) progressively
transfers textual and visual CoT supervision into latent reasoning dynamics for efficient action generation.
In contrast, WALL-WM uses a staircase parallel decoder to inject and propagate latent CoT tokens through
staggered depths of the VLM backbone. This design preserves as much of the pretrained VLM’s hierarchical
visual-linguistic priors and causal computation structure as possible, while amortizing the layer cost of latent
reasoning.

5

# Page 6
3
Architecture Design: Event-Centric World Action Modeling

3.1
Overview

As summarized in Fig. 3 , WALL-WM instantiates event-centric world action modeling through a prior-aligned,
multimodal pretraining stack: a video tower inherited from a Wan Series text-to-video model ( 72 ) and
a randomly initialized action DiT are layer-coupled, with pretrained encoders held fixed and cross-modal
alignment learned through layer-wise video-action couplings. Pretraining is organized at the event level: each
sample is an atomic ( V 𝑒 , a 𝑒 ) event carved out of a long-horizon episode rather than a fixed-length chunk, and
the model is trained to denoise event-aligned futures from the current observation. At each block, the action
stream cross-attends to the matched video features without modifying the video stream.

Formally, WALL-WM models 𝑝 𝜃 ( V 𝑒 , a 𝑒 | V 0 , s , 𝑐 𝑒 ) , where V 0 is the current multi-view observation (one keyframe
per camera), s the current proprioceptive state, ( V 𝑒 , a 𝑒 ) the event-aligned future multi-view video and end-
effector trajectory (both with event-dependent length), and 𝑐 𝑒 the per-event caption that describes the same
action-grounded semantic event. The remainder of Section 3 follows the two paradigms of Fig. 2 : next-event
prediction with the event-centric layout in subfigure (A) for pretraining and event-mode inference, and next-chunk
prediction with the observation-centered layout in subfigure (B) for unified-mode deployment. First, Sections 3.2
and 3.3 build the shared video-action denoiser from event-centric pretraining through layer-wise coupling and
temporal alignment, extending to the history-augmented observation-centered window. Second, Section 3.4
attaches inference-time language pathways to the same event-pretrained backbone: a vision-aware VLM bridge
supports next-event conditioning in event mode and instruction grounding in unified mode, while Staircase
decoding supplies event structure for fixed-length chunk prediction without token-by-token autoregression.

3.2
Multi-View Visual World Events Modeling

The video tower inherits the Wan single-view DiT and extends it to multi-view, multi-embodiment video
generation. The inherited within-view computation stays in place; we graft three additions onto the same
backbone: (i) multi-view adaptation from single-view priors , which runs rearranged cross-view self-attention
over per-frame multi-view tokens and merges the result back through a zero-initialized output projector; (ii)
Camera RoPE , which gives each camera a learnable, calibration-free identity in that cross-view branch so the
same DiT can operate across heterogeneous multi-embodiment camera setups; and (iii) cross-view geometric
masking , a complementary training-time pair that strengthens cross-view geometric consistency. The tower is
then trained on event latents under Wan-style 𝑣 -prediction flow matching ( 31 , 64 , 48 ).

Multi-View Adaptation from Single-View Priors.
When 𝑁 𝑣 > 1, each DiT block runs an additional cross-view
branch after the usual within-view Wan self-attention. Each camera stream still enters as its own batch item;
inside a view-attention block we regroup hidden states frame by frame, concatenate all spatial tokens from the
𝑁 𝑣 cameras at each latent frame into one sequence, and run self-attention on that joint layout—with rotary
codes that also encode the camera axis (Camera RoPE, next paragraph). The attention weights are initialized
from the block’s within-view self-attention; its output is passed through a zero-initialized projector and added
back to the per-view stream under an AdaLN gate. Let h 𝑉
𝑖 denote the hidden states at DiT block 𝑖 (in the per-view
Wan layout), CrossViewAttn 𝑖 (·) the rearranged cross-view self-attention above, 𝑊 view its zero-initialized output
projector, and 𝑔 𝑖 the AdaLN gate on that branch; then

h 𝑉
𝑖
← h 𝑉
𝑖 + 𝑔 𝑖 𝑊 view CrossViewAttn 𝑖
  h 𝑉
𝑖

,
𝑊 view initialized to 0 .
(1)

Because 𝑊 view starts at zero, this branch contributes nothing at initialization and cross-view exchange turns on
only as the projector learns during training. The within-view Wan stack is otherwise unchanged, so pretrained
appearance and language-alignment behavior are preserved while cross-view exchange is learned on top.

Camera RoPE.
To support large-scale multi-embodiment training, Camera RoPE gives each view a learnable
rotary identity without feeding calibration to the model at runtime. We extend RoPE ( 67 ) with a view axis,
partitioning each head’s frequency bank over ( 𝑓, ℎ, 𝑤, view ) ; the view rotation is produced from a learnable
per-view embedding shared across all view-attention layers. Adding or removing a camera therefore only
changes the embedding table. The rotary code tells the network which camera each token came from; during
training, the sight-cone mask below further specifies which other tokens it may plausibly correlate with.

6

# Page 7
Figure 3 Overall framework of WALL-WM. WALL-WM implements event-centric world action modeling as a layer-coupled
video-action denoiser: given the current multi-view observation and a next-event instruction, it jointly denoises future video
latents and the corresponding end-effector trajectory. The figure is organized as three subfigures. (a) supplies instruction:
event mode routes next-event language into the shared text conditioner (T5 ( 62 ) embeds in the figure), whereas unified
mode routes through a Staircase decoder to CoT latents. (b) is the event-centric world model—a Multi-View Video DiT
denoises video latents and an Action Transformer denoises actions; Execute and Rollout close the control loop above. (c)
summarizes the spatial-temporal fusion that threads 𝑁 𝑣 = 3 camera streams through the stack: view folding that preserves
the inherited within-view self-attention with its 3D RoPE positional encoding (S1), cross-view mixing in each DiT block
(S2), token-axis ViewConcat (S3), and one-way block-wise coupling of fused video keys/values into the action tower at
every depth (S4).

Cross-View Geometric Masking.
View attention alone can mix tokens across cameras even when their patches
have no shared field of view, and when co-visible regions do exist the network often recovers them through
the shorter temporal path inside a single view, so cross-view attention is either abused as a generic feature
mixer or left under-trained on the pairs that matter. At training time, we address both failure modes with a
complementary pair of geometry-aware masks built from the same per-robot calibration; both are dropped at
inference so rollout stays calibration-free.

Sight-cone attention masking. For two video tokens 𝑢 = ( 𝑣 𝑢 , ℎ 𝑢 , 𝑤 𝑢 ) and 𝑢 ′ = ( 𝑣 𝑢 ′ , ℎ 𝑢 ′ , 𝑤 𝑢 ′ ) on the same latent frame,
we say the pair is co-visible if the viewing frustums of their related patches in the original videos intersect (there
could be observations of the same region from different viewing angles in pixel space). For computational
convenience, we model each frustum as a cone 𝐶 ( 𝑢 ) = ( p 0 ( 𝑢 ) , ˆ v ( 𝑢 ) , 𝛾 ( 𝑢 )) with apex at the camera center, axis ˆ v
toward the patch center, and half-apex angle 𝛾 initialized to tightly enclose the patch (optionally scaled by
𝑙 ≥ 1). Cone parameters follow per-robot extrinsics, intrinsics, and distortion; intersections are tested in parallel
within a depth-of-field band [ 𝑑 min , 𝑑 max ] :

p ( 𝑢, 𝑡 ) = p 0 ( 𝑢 ) + 𝑡 ˆ v ( 𝑢 ) ,
(2)

( 𝑡 1 , 𝑡 2 ) = arg min
( 𝑡 1 ,𝑡 2 )
∥ p ( 𝑢, 𝑡 1 ) − p ( 𝑢 ′ , 𝑡 2 )∥ 2 ,
(3)

ˆ 𝑡 1 = clamp ( 𝑡 1 , 𝑑 min , 𝑑 max ) , ˆ 𝑡 2 = clamp

arg min
𝑡 2

p ( 𝑢, ˆ 𝑡 1 ) − p ( 𝑢 ′ , 𝑡 2 )
2 , 𝑑 min , 𝑑 max



,
(4)

𝐶 ( 𝑢 ) intersects 𝐶 ( 𝑢 ′ )
⇐⇒
p ( 𝑢, ˆ 𝑡 1 ) − p ( 𝑢 ′ , ˆ 𝑡 2 )
2 ≤ ˆ 𝑡 1 𝛾 ( 𝑢 ) + ˆ 𝑡 2 𝛾 ( 𝑢 ′ )
( approximately ) .
(5)

7

# Page 8
This yields a binary mask
M sc [ 𝑢, 𝑢 ′ ] = 1
⇐⇒
𝐶 ( 𝑢 ) intersects 𝐶 ( 𝑢 ′ ) .
(6)

We add ( 1 −M sc ) · (−∞) as an attention bias in every view-attention block, forbidding cross-view routing
across geometrically incompatible patches; the mask is computed once per sample and reused across depth. At
inference the bias is dropped, recovering unmasked attention with Camera RoPE. M sc closes the loop with
Camera RoPE above: rotary codes identify which camera each token came from, and the sight-cone mask
identifies which other tokens it may correlate with.

Tube patch masking. The second mechanism creates an explicit demand for cross-view attention. With
probability 𝑝 tube we pick a view 𝑣 ∗ and a 𝑘 × 𝑘 spatial window with 𝑘 ∈{ 𝑙 min , . . . , 𝑙 max } , uniformly sampled inside
𝑈 = { 𝑢 | ∃ 𝑢 ′ , 𝑣 𝑢 ′ ≠ 𝑣 ∗ , M sc [ 𝑢, 𝑢 ′ ] = 1 } so recovery from other views is possible. The resulting tube —the same spatial
window across all latent frames of 𝑣 ∗ —is masked in the noised input z 𝑉
𝑡 by replacing the tokens with pure noise;
with nested probability 𝑝 cond
tube the same tube is also masked on the conditioning channel y . The reconstruction
target is unchanged, but the masked tube has no within-view temporal shortcut, so recovery must route through
the other 𝑁 𝑣 − 1 views. Non-trivial ( 𝑝 tube , 𝑝 cond
tube ) are used when hardening cross-view correspondence; how the
masked tube enters the video loss is specified in the flow-matching paragraph below.

U ’

V ’

Mask all τ region in x i
with probability P tube

Latents x t x t + 1 x t + 2 x t + 3 ...

Latents x t x t + 1 x t + 2 x t + 3 ...

Camera C ’

Figure 4 Cross-view masking in WALL-WM’s view attention. (a) Sight-cone mask. A token pair ( 𝑢, 𝑢 ′ ) is allowed to attend
if and only if their back-projected sight cones share a 3D region in front of both cameras; the resulting binary mask M sc is
added as a ( 1 −M sc )(−∞) attention bias, with intra-view pairs always allowed (matrix diagonal). (b) Tube mask. A spatial
window on one view 𝑣 ∗ is masked across all latent frames—and, with nested probability, also on the InP conditioning
channel y —leaving cross-view attention to the other views as the only path that can recover the masked content; the in-tube
𝑣 -prediction loss is up-weighted as in the flow-matching objective below. Both masks are training-only, so the runtime path
stays calibration-free.

Why the two are complementary. Sight-cone masking acts on attention topology : it makes the cross-view graph
reflect physical visibility, but does not push traffic along the remaining edges. Tube patch masking acts on input
content : it removes information recoverable through temporal self-attention, but does not block geometrically
nonsensical correlations. Used together at training time, the model may attend across views only where
geometry permits and is required to do so to minimize the video objective below ( Figure 4 ).

Video Flow-Matching Objective.
The video tower inherits Wan-style flow matching and is trained on event
latents before the action tower is attached ( Section 5.1 ). Given clean latents z 𝑉
0 , noise ε 𝑉 , and a sampled video

8
