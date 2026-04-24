# HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations

## 一句话结论

这篇文章的核心价值是把 UMI 从“手腕视角的局部操作”扩展到“带主动感知的 whole-body mobile manipulation”，并且确实在不使用机器人遥操作数据的前提下，把纯人类示范迁移到了双臂移动操作机器人上；对做 UMI、human data、跨 embodiment manipulation 的人来说，值得细看。

## 核心创新

- 在 UMI 的双手 handheld interface 上加入头戴相机与 ARKit 多设备对齐，采集同步的双腕视角、头部视角、深度、位姿和 gripper 宽度，形成可扩展的 robot-free mobile manipulation 数据采集方案。
- 为了跨 embodiment 迁移，不直接把头部 RGB 和 6DoF head pose 硬塞给策略，而是设计了 embodiment-agnostic 的 3D egocentric 表示，把 head image 提升到点云/pointmap + 视觉 token 的 3D 表征。
- 将 head action 从“模仿人类头部 6DoF 轨迹”放松成 “3D look-at point”，把主动感知意图保留下来，同时避免人和机器人 neck DoF、高度、运动范围不匹配导致的不可行控制。
- 用以 end-effector 精确跟踪为主目标、同时带稳定性/碰撞/重心/姿态约束的 whole-body IK controller，把 policy 输出的 hand-eye 目标转成可执行的全身动作。

## 方法概述

- 数据采集：系统使用 3 台 iPhone，两台装在 gripper 上，一台装在头部。借助 ARKit 多设备协同，把三路视频、深度和 6DoF pose 放进统一坐标系，以 60Hz 记录示范。
- 策略形式：基于 Diffusion Policy，输入短时 observation window，输出未来一段 action horizon。
- 视觉表示：wrist camera 仍保留 2D 表示来提供局部接触线索；head camera 不直接走 RGB imitation，而是先得到 pointmap，再与 DINOv3 patch feature 结合成 3D token，并在 gripper 坐标系下做 arm/body masking，减少人机外观和视角差异。
- 动作表示：双手仍预测 end-effector 动作；head 不预测 6DoF pose，而是预测一个 look-at point。这样机器人只需把 gaze 朝向该点，而不必几何上复刻人的头部姿态。
- 坐标系设计：把观测与动作统一变换到 left-gripper frame，而不是 egocentric frame。这样空间推理围绕真正执行任务的手来展开，更利于跨 embodiment。
- 控制执行：后端 whole-body controller 用 constrained QP / differential IK，优先保证双手 SE(3) 跟踪，同时加入 torso upright、CoM 支撑、碰撞避免、速度约束、human-like nominal posture 等项；异步执行链路把 10Hz policy、100Hz IK、500Hz robot control 串起来。

## 实验结论

- 论文在 3 个真实机器人长时程任务上验证：`Laundry`、`Delivery`、`Tablescape`，都要求双臂配合、移动底盘、头部主动感知，部分还涉及大范围导航。
- 成功率分别达到 `90% / 85% / 80%`，且都显著优于对比项。
- `Wrist-Only` 证明只靠 UMI 原始 wrist view 不够，缺少全局上下文、搜索能力和双手相对空间感。
- `RGB-Only` 证明“直接加 head RGB + 直接预测 6DoF head action”在跨 embodiment 下会明显 OOD，甚至在两项任务上接近完全失败。
- `Head-Only` 证明 egocentric 全局视角不能替代 wrist 局部接触信息，抓取精度和对位会出问题。
- `w/o Active Neck` 的退化说明主动 head control 不只是锦上添花，而是维持可观测性、搜索目标和精确放置的重要因素。

## 和我关注点的关系

- 和 UMI 非常相关：这基本可以看成 UMI 朝 mobile manipulation + active perception 的一次关键扩展，重点不在“换个 backbone”，而在“如何重新定义 observation / action / controller 来跨 embodiment 迁移”。
- 和 manipulation data 很相关：它说明只要表示和控制接口设计得对，纯人类示范并不一定必须再配 robot teleop data 才能落到真实机器人。
- 和 embodied foundation model / VLA 的关系在于：这篇虽然不是大模型路线，但它非常清楚地说明了 VLA 之前更底层的问题，即“什么 observation/action interface 对跨 embodiment 学习是对的”。这对以后做更大规模 mobile manipulation policy 很关键。
- 和视频生成/世界模型的直接关系不强，但其中“look-at point 表示主动感知意图”“把 head observation lift 到 3D 再做 embodiment-invariant reasoning”这两点，对未来把 video/world model 接到控制上是有启发的。

## 启发 / 可迁移点

- UMI 类系统要扩到 mobile manipulation，不能只加一个头部相机就完事，关键是重新定义 head observation 和 head action；否则 OOD 和不可执行约束会一起爆炸。
- “放松动作表示”这个思路很值得借鉴。head 这里用的是 `look-at point`，本质上是在保留任务相关意图、舍弃 embodiment-specific 细节。类似思路也许可以迁移到 torso、base、甚至 hand orientation 的部分维度控制上。
- 统一到 gripper-centric frame 很有价值。它不是单纯的数据预处理，而是在给策略一个更稳定、更任务相关的空间参考系。
- wrist + ego 双视角互补这一点很重要：全局视角负责找目标、理解场景、追踪任务进度，局部视角负责接触、对位和精抓。这比争论“到底该用头相机还是手腕相机”更有建设性。
- whole-body controller 不是后处理配角，而是整个 imitation interface 的一部分。数据表示、policy 输出和控制器约束必须协同设计。

## 局限与疑问

- 规模仍然不大。三项任务的数据量分别约为 115 到 200 条示范，证明了可行性，但离真正大规模、开放域 mobile manipulation 还有距离。
- 任务分布仍较窄，主要是结构化室内任务，尚未证明对更复杂障碍、动态场景、长尾物体类别的鲁棒性。
- 方法相当依赖较强的几何感知条件，包括 iPhone 深度/ARKit 对齐、机器人端 pointmap/stereo 深度、精心布置的多相机系统；部署成本并不低。
- whole-body controller 是强工程先验，泛化到其他机器人平台时仍需重新适配约束、姿态偏置和运动学结构，跨 embodiment 并不是“零成本迁移”。
- 论文证明了 head 表示和 action abstraction 很重要，但还没有真正回答这些设计在更大模型、更大数据规模下是否仍是最优接口。
- 推测：如果把这里的 hand-eye interface 接到更强的时序 world model 或 video model 上，可能能进一步增强长时程搜索与恢复能力，但本文本身没有验证。

## 值得细看部分

- `3D egocentric representation` 的具体构造方式：pointmap + DINOv3 patch feature + 3D positional encoding + arm masking，这部分是它解决 visual embodiment gap 的关键。
- `look-at point` 这个动作抽象：这是全文最可迁移的设计之一，体现了“模仿意图而不是模仿关节轨迹”。
- `gripper-centric frame` 的选择：很简单，但从学习接口角度看非常重要。
- whole-body controller 的目标项和约束项配置：如果你后面想做类似的 mobile manipulation imitation，这部分工程经验很有参考价值。

## 可做的后续实验

- 把 `look-at point` 扩展成更一般的 task-relevant latent target，测试是否能统一表示主动感知、导航关注点和操作关注点。
- 比较不同空间参考系：egocentric、base-centric、world-centric、gripper-centric，看看在更大数据规模下哪种最利于跨 embodiment。
- 测试更弱几何条件下的效果，比如只用单目深度估计或更 noisy 的 pointmap，评估这套表示对 3D 质量的敏感性。
- 在更大规模 human video / egocentric data 上预训练 head encoder，再用 HoMMI 任务微调，看看是否能进一步降低真实机器人示范需求。
