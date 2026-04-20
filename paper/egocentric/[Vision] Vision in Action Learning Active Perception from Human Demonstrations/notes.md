# Notes

## Paper
- Title: Vision in Action: Learning Active Perception from Human Demonstrations
- Folder: D:\Papers\paper\egocentric\[Vision] Vision in Action Learning Active Perception from Human Demonstrations

## Motivation
- 这篇文章的核心动机不是单纯“让机器人看得更好”，而是让机器人学到真正和任务推进相关的 active perception。作者在 introduction 一开始就把 active perception 拆成了三类具体作用：`search`、`reduce occlusion`、`focus on action-critical regions`。
- 作者认为现有 imitation learning 系统的一个明显问题是视角受限。很多方法依赖 wrist camera 或 fixed camera，但 wrist camera 的视角受 arm motion 约束，fixed camera 又常常看不到关键区域，所以在遮挡场景下很难获得 task-relevant information。
- 另一个更关键的问题是 `observation mismatch`。人类在示教时会自然转头、调整视角、主动搜索，但机器人最后学习时往往不是在相同 observation space 下进行，因此很难真正学到 human perceptual strategies。
- 作者把 active perception 的难点总结为三类：`flexible hardware for human-like gaze control`、`synchronized camera-gaze movements`、`scalable active perception strategies`。也就是说，这篇工作的难点不是只加一个 head camera，而是同时处理 embodiment、teleoperation interface 和 policy learning。

## Method
- 方法主线可以概括成三部分：`6-DoF robot neck`、`intermediate 3D scene representation`、`shared-observation teleoperation`。
- 在硬件上，作者没有专门设计复杂 humanoid neck，而是直接用一个 `off-the-shelf 6-DoF arm` 作为 robot neck。这样做的好处是简单，但能提供比较接近人类上半身带来的 head motion 自由度。
- 在系统上，作者最核心的设计是用一个中间 `3D scene representation` 解耦用户头部运动和机器人真实相机运动。系统先把头相机采到的 RGB-D 通过 `W T_H(t)` 变换到 neck base 定义的 `world frame W`，形成点云 `W X(t)`。然后，当用户头动时，系统不等待机器人真的把头转过去，而是直接用用户最新头姿 `W T_user(t+k)`` 从点云中渲染新的 VR 视角。之后再在更长的时间尺度 `K` 上，把聚合后的 head motion 发给机器人真实执行。
- 这意味着遥操作中其实有两条并行链路：一条是快链路，负责根据用户最新头姿实时渲染视角；另一条是慢链路，负责机器人实体的异步跟随。核心不是“用户头一动，机器人相机立刻到位”，而是“用户先在渲染空间里即时获得正确视角，机器人实体随后异步追上”。
- 我们前面讨论过的 `height offset`，更像是在把 VR 头显位姿映射到机器人 `world frame` 时加上的固定高度校准项。原文没有给出更细的公式定义，所以更稳妥的理解是：它不是“把头部动作变成相对动作”本身，而是在相对头动已经被测得之后，对渲染视点的高度做坐标系对齐。
- 在数据采集上，作者强调 `shared observation`。这里并不是说人类和机器人看到的东西要做到严格像素级一致，而是尽量让人类操作员和机器人共享同一个 observation space，减少 demonstration 和 policy learning 之间的 observation mismatch。
- 在学习部分，作者基于 Diffusion Policy。输入包括 active head camera 的 RGB 图像，以及 neck、双臂、夹爪的 proprioception；输出则是未来 neck 和双臂的 end-effector pose 以及 gripper width。也就是说，这篇文章的关键不是单纯“加了个可动摄像头”，而是把 `head movement` 本身并入了 policy 要学习的 action space。

## Experiments
- 实验主要围绕三个方向展开：`camera setup comparison`、`visual representation comparison`、`teleoperation interface comparison`。任务则包括 `Bag Task`、`Cup Task`、`Lime & Pot Task`，分别对应 interactive perception、active viewpoint switching、以及 bimanual coordination + precise alignment。
- 在 camera setup 对比里，作者比较了 `ViA(active head only)`、`active head + wrist cameras`、`chest + wrist cameras`。结果显示单 active head camera 的方案最好，甚至比加上 wrist cameras 还强。作者的解释是：head view 本身已经是 task-complete 的，而额外 wrist views 在 low-data regime 下可能只会带来冗余、遮挡和噪声。
- 在 visual representation 对比里，作者用相同的 active head camera 输入，比较 DINOv2、ResNet-DP 和 point-cloud based DP3。结果是 DINOv2 表现最好。作者把这点归因于更强的语义先验，使 policy 能先“找到目标”，再执行 manipulation。
- 在 teleoperation interface 对比里，作者用 user study 比较了 `point-cloud rendering` 和传统 `stereo RGB streaming`。结果是 point-cloud rendering 会让数据采集时间略长，但显著降低 motion sickness；8 个参与者里有 6 个更偏好作者的系统。
- `Bag Task` 是这篇文章里很有代表性的实验，因为它强调的是 `occlusion + interactive perception`：机器人要先打开袋子来减少遮挡，再通过 active head movement 去看袋子内部，然后取出目标物体。这个设计和作者在 introduction 里讲的 active perception framing 是一致的。
- 不过，`Bag Task` 也有一个需要保留的批判性理解。就正文描述来看，作者明确强调了训练/测试物体类别变化，但没有像 `Cup Task` 和 `Lime & Pot Task` 那样清楚写出 bag position 或 configuration randomization。因此，这个实验更强地支持“active head setup 在遮挡场景中有价值”，但没有特别强地证明 policy 学到了强泛化的 `online search strategy`。对于单任务 policy 来说，学成某种固定套路，例如“开包后往某个方向探头看”，是完全可能的。

## Overall Takeaway
- 这篇文章最有意思的地方在于，它把 `active perception` 从一句很容易说空的话，落成了一个完整系统问题：硬件、VR 接口、共享观测、行为克隆、任务评测一起设计。
- 它的强项在于问题 framing 和系统整合，尤其是 `6-DoF neck + 3D rendering + shared observation` 这条完整链路。
- 它相对没完全回答的问题，是实验还没有彻底排除 task-specific habit 的解释，尤其在 `Bag Task` 里，active head movement 也可能部分是固定套路，而不一定是强泛化的搜索策略。
