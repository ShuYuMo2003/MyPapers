# HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations

## 一句话结论

这篇论文的核心价值不只是“给 UMI 加一个头戴相机”，而是提出了一套能把机器人无关的人类 whole-body mobile manipulation 演示，稳定迁移到双臂移动机器人上的表示与控制接口；对 UMI 方向很值得细看。

## 核心创新
- 把 UMI 从 wrist-centric 操作扩展到带 egocentric 观察的 whole-body mobile manipulation 数据采集接口，用 3 个 iPhone 同步记录 wrist/head RGB、深度、6-DoF pose 和 gripper 宽度。
- 不直接拿 head RGB 和 human head 6-DoF pose 做模仿，而是显式处理 human-robot embodiment gap：用 embodiment-agnostic 的 3D visual representation 加 gripper-centric 坐标系，减少视角和外观偏差。
- 把 head action 从难以实现的 6-DoF 头部轨迹，改成更松弛的 3D look-at point 表示，把 active perception 迁移成机器人可执行的 gaze control。
- 用 constraint-aware whole-body controller 在 base、torso、head、双臂之间做协调，尽量保证手部轨迹精确、同时满足机器人本体约束。

## 方法概述
- 数据采集上，它延续 UMI 的 handheld gripper 思路，但额外加入头戴 iPhone，借助 ARKit multi-device collaboration 把三台设备对齐到统一坐标系。这样采到的数据天然同时包含局部接触视角和全局环境视角。
- 策略学习上，主体还是 Diffusion Policy 风格的 visuomotor policy，但关键不在 backbone，而在 observation/action interface 的设计。
- 观察表示上，作者认为 egocentric RGB 直接迁移会带来严重 OOD，因为人和机器人头高、手臂外观、相机视角都不同；所以他们转成 3D 表示，并尽量使用与 embodiment 无关的坐标系，同时掩掉人类身体/手臂等特征。
- 动作表示上，双手依然承担精细 manipulation，而 head 不再学 full pose imitation，只学“看向哪里”；这相当于把 active perception 从不可执行的姿态克隆，变成可在不同机器人上复用的感知意图。
- 执行层上，whole-body controller 负责把 hand-eye targets 转成受约束的整机运动。这一点很重要，因为论文不是单纯做 policy learning，而是明确承认 robot embodiment gap 里有一部分必须靠控制层接口来消化。

## 实验结论
- 论文做了 3 个长时程 mobile manipulation 任务：laundry、delivery、tablescape，分别覆盖主动搜索、长距离导航、双臂协调、whole-body 调整和精细放置。
- HoMMI 在三项任务上分别达到 90%、85%、80% 成功率，明显优于几个关键 baseline。
- `Wrist-Only(UMI)` 在需要全局上下文、主动搜索、导航和双臂对齐的任务上明显不够，说明原始 UMI 的 wrist-centric 观察对 mobile manipulation 不够。
- 直接把 head RGB 和 6-DoF head action 硬塞进 UMI 的 `RGB-Only(UMI+Ego)` 表现很差，甚至两项任务 0% 成功率，说明“多加个第一视角相机”本身并不能解决问题，反而会放大 embodiment mismatch。
- `Head-Only` 也不行，抓取和对位都差，说明 wrist camera 提供的局部接触信息仍然是 manipulation 成功的关键。
- 去掉主动头部控制的 ablation 也会掉点，说明 active perception 不是锦上添花，而是这类任务的必要能力。

## 和我关注点的关系
- 和 UMI 的关系非常直接：这基本可以看成 UMI 向 mobile manipulation 和 active perception 场景的系统化扩展。
- 和 manipulation/data 的关系也很强：它证明了 robot-free human demonstrations 不一定只能学固定基座、局部操作，经过合适的表示和控制接口设计，可以覆盖更长时程、更强 whole-body coordination 的任务。
- 和 cross-embodiment transfer 的关系尤其重要：论文最有价值的不是“又做了个 imitation learning policy”，而是把视觉 gap、运动学 gap、控制可实现性 gap 拆开分别处理。
- 和 VLA 的关系没那么直接，因为它不是大模型路线；但它对 embodied foundation model 很有启发：如果 observation/action tokenization 没有处理 embodiment mismatch，再大的模型也会学到脆弱映射。
- 和 video generation / world model 的直接联系较弱；但它提供了一个很清楚的 bridge：先把人类行为抽象成 embodiment-agnostic 的 hand-eye/world representation，再让生成模型只预测这些“可迁移 latent/action interface”，会比直接生成 robot joint trajectory 更合理。这里是推测。

## 启发 / 可迁移点
- 对 UMI 系统继续扩展时，最值得借鉴的是“不是盲目加传感器，而是重做跨 embodiment 的 observation/action interface”。
- `look-at point` 这个 head action abstraction 很值得迁移。很多 active perception 任务未必要学 camera pose imitation，而可以学更高层的感知目标。
- 局部 wrist 视角和全局 ego 视角的组合很关键。前者负责 contact-rich precision，后者负责 search / alignment / task progress，这种分工对未来 VLA observation design 也有参考价值。
- 论文把控制器作为 embodiment gap 的一部分来建模，而不是假设 policy 直接输出就能落地，这种 policy-interface-controller 分层很务实。
- 如果你后面想做更大规模 human manipulation data，HoMMI 说明可扩展数据采集接口本身就是研究对象，不只是“收数据前的工程准备”。

## 局限与疑问
- 任务数和场景复杂度仍然有限，虽然已经比固定基座更强，但距离开放环境 mobile manipulation 还差不少。
- 数据规模并不大，分别是 200、166、115 条 demonstrations，说明当前结果更多证明“接口设计有效”，还没有证明这种范式能大规模扩展到更广任务族。
- backbone 基本沿用 Diffusion Policy，论文的增益主要来自接口和控制设计，因此结论更像“system/interface contribution”，不是学习算法层面的突破。
- 论文主要在单一机器人平台上验证，跨平台泛化还没有被真正证明；“cross-embodiment”更多是 human-to-this-robot transfer，而不是 multi-robot transfer。
- 3D visual representation 的构建依赖 iPhone depth/ARKit 这类硬件与标定条件，换设备或更差传感质量后还能否保持效果，论文里没有充分展开。
- 没看到特别系统的 scaling law 式分析，比如数据量增加、表示模块替换、控制器简化后性能怎么变化，这对判断它能否成为更通用范式还不够。

## 值得细看部分
- `IV. HoMMI Data Collection Interface`：如果你关心 UMI 数据采集系统怎么扩到 mobile manipulation，这部分最值得看。
- `V. Cross-Embodiment Hand-Eye Policy`：这是整篇最核心的技术部分，尤其是 3D visual representation、gripper-centric frame、look-at point action。
- `VI. Robot System`：如果你想知道 policy 和 controller 怎样解耦协同，这部分应该细读。
- `VII. Evaluation` 里的 ablation 很有价值，因为它清楚回答了 wrist view、ego view、active neck 各自到底贡献了什么。

## 可做的后续实验
- 把 HoMMI 的 hand-eye abstraction 接到更强的 policy backbone 上，比如更大规模的 VLA / action diffusion / latent world model，测试接口设计是否仍然成立。
- 研究能否只用视频和更弱的姿态信号重建类似的 embodiment-agnostic 表示，减少对 ARKit/depth 的依赖。
- 把 `look-at point` 扩展成更一般的 active perception objective，例如 next-best-view token 或 task-relevant region token。
- 做真正的 multi-robot transfer：同一批 human demos 迁移到不同 head DoF、不同 base morphology、不同 arm span 的平台，检验这套接口是否真的具有更广泛的 embodiment invariance。
- 试试把这种 hand-eye/world abstraction 接到视频生成模型或 world model 上，让模型生成的是“可迁移中间表示”而不是像素或 joint，看看是否更适合 embodied planning。这里是推测。
