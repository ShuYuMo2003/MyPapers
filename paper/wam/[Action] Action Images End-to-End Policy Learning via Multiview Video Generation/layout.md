# Page 1
Action Images: End-to-End Policy Learning
via Multiview Video Generation

Haoyu Zhen 1 ∗ , Zixian Gao 1 ∗† , Qiao Sun 1 † , Yilin Zhao 2 ,

Yuncong Yang 1 , Yilun Du 3 , Pengsheng Guo 4 ,

Tsun-Hsuan Wang 4 , Yi-Ling Qiao 4 , and Chuang Gan 1

1 UMass Amherst
2 NVIDIA
3 Harvard University
4 Genesis AI

https://ActionImages.github.io

Text prompt: place blue chip bag in grey bowl

2. Pixel-grounded Action Images

View 2
View 1

Robot Videos

1. Observations from Any View

4. Zero-shot 3D Policy
3. Video-action Joint Generation

Action Videos

Fig. 1: Action Images turns policy learning as multiview video generation:
7-DoF actions are translated into pixel-grounded action images that explicitly
track robot-arm motion, enabling a zero-shot policy directly from a unified
video backbone

Abstract. World action models (WAMs) have emerged as a promising
direction for robot policy learning, as they can leverage powerful video
backbones to model the future states. However, existing approaches of-
ten rely on separate action modules, or use action representations that
are not pixel-grounded, making it difficult to fully exploit the pretrained
knowledge of video models and limiting transfer across viewpoints and
environments. In this work, we present Action Images, a unified world
action model that formulates policy learning as multiview video gener-
ation. Instead of encoding control as low-dimensional tokens, we trans-
late 7-DoF robot actions into interpretable action images: multi-view

* Equal contribution.
† This work was done when two of the authors were remote interns at UMass.

arXiv:2604.06168v2  [cs.CV]  15 Apr 2026

# Page 2
2
H. Zhen et al.

action videos that are grounded in 2D pixels and explicitly track robot-
arm motion. This pixel-grounded action representation allows the video
backbone itself to act as a zero-shot policy, without a separate policy
head or action module. Beyond control, the same unified model sup-
ports video-action joint generation, action-conditioned video generation,
and action labeling under a shared representation. On RLBench and
real-world evaluations, our model achieves the strongest zero-shot suc-
cess rates and improves video–action joint generation quality over prior
video-space world models, suggesting that interpretable action images
are a promising route to policy learning.

1
Introduction

World action models [20, 27, 34, 65, 70] have made rapid progress in predicting
future observations, but turning this predictive ability into policy generalization
remains an open challenge. In particular, strong video generation does not auto-
matically produce a strong policy: a model may successfully synthesize plausible
future frames, yet still fail to decide how to act in unseen environments. This gap
between video generalization and policy generalization is a central bottleneck for
world models.
A key reason is that action is still not represented in a form that world
models can naturally generalize. Existing approaches typically follow one of two
paths. Some [12, 34, 65, 70, 72] attach a separate policy head or action module
on top of a world model, asking an additional network to decode control from
learned video features. Others [27] adapt video models to action generation using
representations that are not spatially grounded in image space. In both cases,
the model’s predictive knowledge of the world is only indirectly connected to
acting. As a result, the burden of generalization is shifted to a specialized control
moduel, which is often exactly where transfer breaks down.
In this work, we formulate policy learning as video generation and address
policy generalization at the representation level. We propose multi-view action
videos, a robotics world modeling framework that translates robot actions into
interpretable action images and models them together with observations in a
unified video-space representation of observation and action. Instead of treating
7-DoF control as low-dimensional signals or latent action codes, we convert each
action into a pixel-grounded action representation that explicitly tracks robot-
arm motion in image space across multiple views. This design makes action native
to the video model itself: the same video backbone can observe, predict, condition
on, and generate action, enabling a zero-shot policy. By grounding action in
pixels rather than in an external interface, we obtain a more generalizable policy
model that transfers more naturally across viewpoints and embodiments.
A key design choice is to represent these action images as multi-view videos.
The motivation is not merely to add more visual observations, but to bridge the
gap between 2D image and the 7-DoF robot action in the 3D space. A single
view often provides only a ambiguous projection of motion, making it difficult for
the model to infer the full action consistently from pixels alone [70, 73]. Using

# Page 3
Action Images: End-to-End Policy Learning via Multiview Video Generation
3

multiple views makes the pixel-grounded action image more reconstructable,
while also improving robustness when some motion is partially occluded.
Beyond control, the same unified video-space representation of observation
and action supports multiple tasks within a single model. Because observation
and action share the same generative space, the model can perform video-action
joint generation, action-conditioned video generation, and action labeling under
one backbone and one training objective. These capabilities emerge without a
separate policy head or action module, showing that a robotics world model can
be trained not only to predict the world, but also to act in it through a common
visual representation.
In summary, our contributions are as follows:

– We identify the gap between video generalization and policy generalization
as a central limitation of current robotics world models, and argue that this
gap can be addressed at the level of action representation.
– We propose multi-view action representation, which translate robot control
into interpretable action images forming a pixel-grounded action represen-
tation, and use this representation to build a zero-shot policy without a
separate policy head or action module.
– We show that this design yields a more generalizable policy model and pro-
vides a unified video-space representation of observation and action that
supports video-action joint generation, action-conditioned video generation,
and action labeling within a single robotics world model.

2
Related Work

2.1
Robotics World Models.

Originating from Reinforcement Learning [41, 53], world models typically take
actions and the current state as input and predict future states [2, 9]. In recent
years, learning world models for diverse robotic applications [5, 17, 32, 60, 69]
has garnered significant interest. With the success of video generation mod-
els, lots of work has developed robotics world models based on video genera-
tion [8, 12, 16, 30, 52, 70, 72]. These video-based approaches typically adopt
a two-stage pipeline, where future observations are first predicted and actions
are then generated based on these predictions. More recently, joint video-action
generation has been explored to unify modeling and control [27, 34]. In par-
ticular, DreamZero [65] demonstrates strong zero-shot generalization and cross-
embodiment transfer. However, these methods encode actions with additional
action modules, leaving much of the pretrained video knowledge underused; we
instead use multi-view action images so the backbone itself is a zero-shot policy.
Concurrent work [33] also investigates video-based formulations for robot policy
learning. Our approach differs in representing actions as pixel-grounded multi-
view images that encode full 7-DoF control, enabling a unified video-action space
and eliminating the need for separate modules.

# Page 4
4
H. Zhen et al.

2.2
Generalist Robot Policy Models.

Policy models map current states to future actions [46, 58]. Developing generalist
control policies that can succeed in diverse tasks and can be lightweightly fine-
tuned to adapt to downstream tasks has long been a central goal [7, 18, 35, 42, 54,
63, 67]. While multiple advances in Vision-Language-Action (VLA) models [6,
22, 28, 74], Diffusion Policy [10, 44], and Reinforcement Learning [21, 42] have
greatly promoted the generalizability of policy models, their diversity is still
limited to relatively narrow task distributions and they struggle to zero-shot
generalize to new environments [13, 71]. In parallel, strong capabilities of video
generation foundation models in predicting future frames and modeling physical
dynamics have inspired policy learning approaches [20, 34, 37]. However, how
to turn video prediction into transferable control remains nontrivial; our action-
frame representation bridge this gap by making action native to the video space.

2.3
4D Generation Models.

“4D” here refers to 3D plus time. Optimization-based methods employ Score
Distillation Sampling, which distills pre-trained diffusion models into specific
4D representations [3, 49, 51, 64]. Recent work [36] explores native 4D gener-
ation, which is trained directly on 4D datasets. Due to the lack of large-scale
pretraining assets, a branch of research leverages the rich semantic priors in pre-
trained video generation models and integrates reconstruction methods to lift
2D frame sequences into 4D results [24, 43, 59, 62, 66]. However, these con-
tributions mostly focus on single-avatar or simple scene generation. Close to
our method, [4, 61] leverage multiview generation to produce complex dynamic
4D scenes that can be replayed at any specified camera pose and timestamp.
However, for robotic tasks, 4D generation is typically limited to a fixed single
view [16, 70, 73]. Although [40] has leveraged multi-view inputs and introduced
a geometry-consistent supervision, they still do not generalize well beyond their
training scenes.

3
Method

Robotics world models have recently shown strong capability in modeling dy-
namics, especially when built on large pretrained video backbones. However,
these advances in video prediction do not directly translate into strong policy
generalization. To address this limitation, we build a unified video-space rep-
resentation of observation and action, where robot control is translated into
interpretable action images that form a pixel-grounded representation. We first
introduce how 7-DoF robot actions are converted into multi-view action videos
(Sec. 3.1), then describe how this representation can be decoded back into contin-
uous control with only minor information loss (Sec. 3.2), and finally present the
training of a unified world-action model that enables a zero-shot policy (Sec. 3.3).

# Page 5
Action Images: End-to-End Policy Learning via Multiview Video Generation
5

Up Point &
Gripper Openness

Normal Point

Position Point

2D Gaussian Point

Normal Point

Fig. 2: Action as image. We convert each 7-DoF robot action into three semantic
3D points (position, normal, and up), project them into image space, and render them
as RGB Gaussian heatmaps. The blue channel further encodes gripper openness in the
low-response background, producing a pixel-grounded action representation.

3.1
Action as Images

Our central idea is to represent robot action in the same modality as robot
observation. Instead of treating action as a low-dimensional control vector that
must be interpreted by a separate policy head, we convert each action into
interpretable action images and model it directly in video space. This yields a
pixel-grounded action representation that is aligned with the robot RGB stream
and can therefore be processed by the same video backbone. As illustrated in
Fig. 2, this design turns action modeling into a tracking-like visual prediction
problem: the model does not need to infer control from abstract tokens, but
instead learns to localize and reason about robot-arm motion.
From 7-DoF action to semantic 3D points. At each time step t , the robot
action is a t = [ p t , θ t , g t ] ∈ R 7 , where p t ∈ R 3 is the end-effector position,
θ t ∈ R 3 denotes its orientation, and g t ∈ R is the gripper openness. We convert
this 7-DoF action into three semantic 3D points: a position point, a normal point,
and an up point. The position point is the end-effector position, q pos
t
= p t . The
other two points are defined by rotating two canonical axes attached to the
end-effector and extending them by a small length ℓ :

\
m
a t h b f { q } ^ { \ t
e xt {up
}
} _ t = \ m a th b f { p}_t + \ell \, \mathbf {R}(\boldsymbol {\theta }_t)\mathbf {e}_x, \qquad \mathbf {q}^{\text {normal}}_t = \mathbf {p}_t + \ell \, \mathbf {R}(\boldsymbol {\theta }_t)(-\mathbf {e}_z),
(1)

where R ( θ t ) ∈ SO (3) is the rotation matrix derived from the action orientation.
Here, the up point follows a canonical in-plane direction of the gripper, while

# Page 6
6
H. Zhen et al.

the normal point follows the direction normal to the robot gripper plane. To-
gether, these three points capture end-effector pose in a form that can be directly
projected into image space.
Multi-view action image rendering. Given a camera view v , we project
the three semantic 3D points into image space using the camera intrinsics and
extrinsics. Denoting the corresponding projection function by π ( v )
t
( · ) , we obtain

\m a t h b
f
{ u } ^
{
\ t ext
{ p
o s},(v) } _ t
=
\ p i
^
{ ( v)}_t(
\
m a
t hb f { q
}
^ { \ t e
x
t {p
o s }}_t),\quad \mathbf {u}^{\text {normal},(v)}_t = \pi ^{(v)}_t(\mathbf {q}^{\text {normal}}_t),\quad \mathbf {u}^{\text {up},(v)}_t = \pi ^{(v)}_t(\mathbf {q}^{\text {up}}_t).
(2)

We then render these projected points into an action image A ( v )
t
∈ R H × W × 3 us-
ing 2D Gaussian. The red channel encodes the position point, the green channel
encodes the normal point, and the blue channel encodes the up point together
with the gripper openness, as shown in Fig. 2. Let G ( x ; u , σ ) denote a 2D Gaus-
sian centered at pixel u . The red and green channels are defined as

\ m
a th b f {A } ^ { ( v ) }_t ( : , :
,
1 ) =
\ m a
t hc a l {G } ( \ c d o t ;\ma t h b f
{ u } ^{\text {pos},(v)}_t,\sigma ),\quad \mathbf {A}^{(v)}_t(:,:,2) = \mathcal {G}(\cdot ;\mathbf {u}^{\text {normal},(v)}_t,\sigma ).
(3)

For the blue channel, we first render the up point as a Gaussian map,

\ t i
l de { \ ma t h b f { A} } ^ { (
v
) } _ t(:,:,3) = \mathcal {G}(\cdot ;\mathbf {u}^{\text {up},(v)}_t,\sigma ),
(4)

and then inject the binary gripper openness signal into low-response regions:

\ m
a t hb f {A }

^ { ( v ) }
_ t (i ,j ,3 )
= \ b
e g in  { ca s e s } \
t i ld e { \
mathbf {A }}^{(v)}_t(i,j,3), & \tilde {\mathbf {A}}^{(v)}_t(i,j,3) > 0.25,\\ 0.25 \cdot g_t, & \text {otherwise}, \end {cases}
(5)

In this way, the blue channel preserves the projected up point while also encoding
gripper openness in a simple and spatially consistent form. The resulting image
is an interpretable action image. Stacking these frames over time yields an action
video for each view,

\ m a t h c a l
{ A } ^ { ( v ) }
= \ { \ m a t h b f {A}^{(v)}_1,\dots ,\mathbf {A}^{(v)}_T\} \in \mathbb {R}^{T\times H\times W\times 3}.
(6)

Since these action videos have the same spatial and temporal structure as the
corresponding robot RGB observations O ( v ) ∈ R T × H × W × 3 , they naturally form
a unified video-space representation of observation and action.
Benefits. Representing actions as interpretable action images provides two key
benefits. First, it makes action prediction spatially grounded: the model learns
control through visible robot-arm motion rather than through abstract action
tokens. Second, it is naturally compatible with pretrained video backbones, al-
lowing the same model to reason over observation and action without an action
module. In this way, our zero-shot policy is obtained by turning the robot action
into a visual prediction problem. Because the representation is pixel-grounded
and multi-view, it transfers more naturally across viewpoints, motion patterns,
and robot embodiments, leading to a more generalizable policy model.

# Page 7
Action Images: End-to-End Policy Learning via Multiview Video Generation
7

Main View
Action Frame A !

"

2. Ray Casting

Action 𝒂 [𝟏…𝑻]

𝑎 !

3. Back Projection

Side View
Action Frame A !

(

1. Select point
from Heatmap

4. Select Best Match

Fig. 3: Action images decoding. A 2D heatmap point is selected in the main view,
lifted to 3D by ray casting and side-view matching, and repeated for all semantic points
to recover the original 7-DoF action.

3.2
Action Images Decoding

A useful action representation should not only be easy to generate, but easy to
decode back into continuous robot control. We therefore design a simple decoding
method that maps the generated multi-view action videos back to the original
7-DoF action. The decoder first reads the gripper state directly from the blue
channel, then reconstructs the underlying 3D semantic points from multi-view
heatmaps, and finally converts them back into the action vector. In this way, the
same unified video-space representation of observation and action can be used
both for generation and for control.
Decoding gripper openness. The blue channel stores both one projected
semantic point and the gripper openness, where the latter is written into low-
response background regions. Let A ( v )
t (: , : , 3) denote the blue channel of the
action image at time t and view v . We estimate gripper openness by averaging
only the low-response pixels:

\ h
a
t {g } _

t =

\

f rac { 1 } { 0
. 2 5 }
\ cd ot \ f
r a c { 1} {| \ O m e g a
_ t |}  \ su m _ {( i ,j,v)\in \Omega _t} \mathbf {A}^{(v)}_t(i,j,3), \ \ \ \\Omega _t = \{(i,j,v)\mid \mathbf {A}^{(v)}_t(i,j,3) < 0.25\}. (7)

Reconstructing 3D semantic points from multi-view heatmaps. For
the remaining action information, we decode each semantic point from its cor-
responding heatmap using a simple multi-view geometric procedure. As illus-
trated in Fig. 3, we first select a 2D point from the heatmap in the main view
by weighted averaging:

\ha
t

{

\ma t hbf
{ u} } ^
{ ( 1 ) } _ t = \ f r a

c

{ \ sum
_ {i , j
} \mathbf {H}^{(1)}_t(i,j) \begin {bmatrix} i+0.5, & j+0.5 \end {bmatrix}^{\!\top } }{ \sum _{i,j} \mathbf {H}^{(1)}_t(i,j) }.
(8)

where H (1)
t
∈ [0 , 1] H × W is the heatmap in the main view. This gives the centroid
of the heat distribution and serves as the 2D anchor point for decoding.

# Page 8
8
H. Zhen et al.

Starting from this point, we cast a ray from the main-view camera center
through ˆ u (1)
t , and sample a set of candidate 3D points along the ray between a
near plane and a far plane. Each candidate is then projected into the side view,
where it is scored against the corresponding side-view heatmap. We choose the
3D point whose projection best matches the side-view response. Concretely, if
{ x t,k } K
k =1 denotes the sampled 3D candidates along the ray, then we select

\ h at {\m
a thb f {x
}
}
_ t =
\ a rg \
m
ax _{\mathbf {x}_{t,k}} \; \mathbf {H}^{(2)}_t\!\left (\pi ^{(2)}_t(\mathbf {x}_{t,k})\right ),
(9)

where π (2)
t
( · ) is the side-view projection and H (2)
t
is the side-view heatmap.
In practice, this procedure is repeated for each semantic point heatmap in the
action image, yielding a set of reconstructed 3D points. The main view provides
the image-space anchor for ray casting, while the side view resolves the depth
ambiguity by selecting the best match along the ray.
From reconstructed points back to 7-DoF action. Once the semantic 3D
points are reconstructed, the original action can be recovered directly. Let ˆ q pos
t
,
ˆ q up
t , and ˆ q normal
t
denote the decoded 3D points. We recover the position as
ˆ p t = ˆ q pos
t
, define ˆ e x
t = norm (ˆ q up
t
− ˆ q pos
t
) and ˆ e z
t = norm (ˆ q pos
t
− ˆ q normal
t
) , then
obtain ˆ e y
t = ˆ e z
t × ˆ e x
t , from which the end-effector orientation ˆ θ t is determined.
The final decoded action is ˆ a t = [ˆ p t , ˆ θ t , ˆ g t ] .
Discussion. When the predicted heatmaps are accurate, the remaining decod-
ing error is dominated not by representation mismatch, but by discretization. In
particular, the 3D reconstruction accuracy is mainly determined by (i) the sam-
pling interval along the ray, which controls depth precision, and (ii) the spatial
resolution of the heatmaps, which controls localization precision in image space.
As a result, the information loss introduced by the action-frame parameteriza-
tion is minor and predictable: finer ray sampling and higher image resolution
directly improve the fidelity of the decoded action.

3.3
Training Unified World Action Model

With robot actions represented as interpretable action images, control becomes
a pixel-grounded visual signal rather than an abstract low-dimensional vector.
This converts action modeling into the same video-space problem as observation
modeling, yielding a unified video-space representation of observation and action.
As shown in Fig. 4, we build a unified world action model by fine-tuning a large
pretrained video generator (Wan 2.2 [56]) to jointly model multi-view robot
videos and multi-view action videos under diverse supervision patterns.
Multi-view video-action tokenization and packing. For each camera
view v , we have an RGB observation clip V ( v )
1: T ∈ [0 , 1] T × H × W × 3 and the aligned
action-frame clip A ( v )
1: T ∈ [0 , 1] T × H × W × 3 . We first encode both streams into
the backbone latent space by the 3D-VAE [29, 56], and then concatenate them
temporally to form a single input sequence

\
m
a t h b
f { X } _ v
\; =
\
; \b i g [ \ ; \ m athbf {V}_{1:T}^{(v)}\;,\;\mathbf {A}_{1:T}^{(v)}\;\big ] \in \mathbb {R}^{(2T)\times h\times w\times c},
(10)
