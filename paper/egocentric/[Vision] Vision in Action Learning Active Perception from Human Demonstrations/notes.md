# Notes

## 3-Minute Version

- **这篇文章在讲什么**
  - 作者想做的不是“加一个会动的摄像头”，而是让机器人学会 **active perception**，也就是为了完成任务，主动去 **找目标、减少遮挡、盯住关键区域**。

- **它为什么有意思**
  - 以前很多 imitation learning 默认“相机拍到什么就学什么”。
  - 这篇文章在说：**很多任务里，怎么看，本身就是任务的一部分。**

- **它怎么做**
  - 用一个 **6-DoF 机械臂当 neck**，让头相机真的能灵活移动。
  - 用 **3D 点云中间表征**，让 VR 用户先立刻看到新视角，机器人实体再异步慢慢跟上。
  - 用 **shared observation** 的方式采 demonstrations，尽量让“人怎么感知”和“机器人从什么输入学”对齐。

- **实验在证明什么**
  - active head camera 在遮挡场景里确实比 wrist / chest 视角更有用。
  - DINOv2 这种语义更强的表征也更适合这类任务。
  - point-cloud rendering 比直接 stereo RGB streaming 更不容易让人晕。

- **我最认同的点**
  - 它把 `active perception` 从一句空话，落成了一个完整系统：**硬件 + 遥操 + 学习 + 评测** 一起设计。

- **我保留的疑问**
  - 像 **Bag Task** 这种实验，更强地说明了 active head setup 很有价值；
  - 但还不算特别强地证明 policy 学到了很泛化的 **online search strategy**，也可能部分只是学到了任务里的固定套路。

## Paper

| Item | Content |
| --- | --- |
| **Title** | *Vision in Action: Learning Active Perception from Human Demonstrations* |
| **Folder** | `D:\Papers\paper\egocentric\[Vision] Vision in Action Learning Active Perception from Human Demonstrations` |
| **My quick feeling** | **很有意思。** 不是只讲“active camera 有用”，而是认真把 **active perception** 当成一个完整系统问题来做。 |

## Quick Take

- **这篇文章最打动我的点**：作者不是在说“多装一个会动的摄像头就行”，而是在说机器人要想真的学会 *active perception*，就得把 **硬件、VR 遥操、shared observation、policy learning、任务评测** 一起设计。
- **作者最想强调的事**：active perception 不是抽象口号，而是很具体的三件事：**找**、**避开遮挡**、**盯住关键区域**。
- **我目前的保留意见**：有些实验，尤其是 **Bag Task**，更像是在证明 active head setup 在遮挡场景里确实有帮助；但还没有特别强地证明 policy 学到了很泛化的 **online search strategy**。

## Motivation

- 作者一上来就讲得很直白：机器人不是“看得更清楚”就够了，而是要为了任务往前推，主动去获取信息。
- 在他们的 framing 里，active perception 主要做三件事：
  - **Search**：先把目标找出来。
  - **Reduce occlusion**：通过换视角或者交互，把遮挡降下去。
  - **Focus**：把注意力放到真正影响动作的局部区域上。
- 他们认为现在很多 imitation learning 系统的问题，不只是 camera 不够多，而是 **视角本身不对**：
  - `wrist camera` 会跟着手走，视角受 manipulation 约束；
  - `fixed camera` 经常看不到真正关键的位置；
  - 一到遮挡场景，就很容易丢掉 task-relevant information。
- 另一个作者特别在意的问题是 **observation mismatch**：
  - 人类示教时会自然转头、搜索、换视角；
  - 机器人学习时却常常不是从同样的 observation space 学；
  - 所以最后模型学到的，不一定是人真正用来完成任务的 perceptual strategy。

### 作者眼里的三个难点

| 难点 | 论文原意 | 我的直白理解 |
| --- | --- | --- |
| **Flexible hardware** | 机器人要能像人一样灵活调 gaze | **头得能真动起来**，而且不是装样子那种 |
| **Camera-gaze synchronization** | 人头怎么动、机器人视角怎么跟，要尽量同步 | **VR 遥操很容易晕**，因为机器人物理运动太慢 |
| **Scalable strategy learning** | 不想靠手工规则写 gaze heuristic | **最好别手写“先看哪再看哪”**，而是从示教里学 |

## Method

### 一句话版本

- 方法主线就是三件事：**6-DoF neck + 3D 中间表征 + shared-observation teleoperation**。

### 1. 6-DoF robot neck

- 作者没有做一个很复杂的 humanoid neck。
- 他们直接用一个 **`off-the-shelf 6-DoF arm`** 当作 robot neck。
- 这个选择挺巧的：
  - **好处**：工程上简单、自由度够大；
  - **目的**：让 head camera 的运动更接近人类上半身带来的视角变化。

### 2. 用 3D scene representation 解耦 VR 和机器人真实运动

- 这是整篇最核心的系统点。
- 作者不想走传统那条路：
  - 用户头一动；
  - 机器人相机立刻跟着动；
  - 再把最新 RGB 画面传回来。
- 因为这样很容易被 **控制延迟 + 视频流延迟** 拖垮，用户会不舒服，甚至 motion sickness。

#### 他们的做法

1. 先把头相机采到的 RGB-D 变成世界坐标系下的点云。
2. 用户头一动，先**直接从点云里渲染新视角**给 VR。
3. 机器人实体的头再**异步慢一点追上**。

### 3. shared observation

- 作者一直在强调：重点不是“人和机器人看到的每个像素都严格一样”。
- 更准确地说，是让人类操作员尽量和机器人处在同一个 **observation space** 里。
- 这样做的目的，是减少：
  - **人怎么感知**
  - 和 **机器人最后从什么输入学**
  - 这两者之间的落差。

### 4. policy 学什么

- 学习部分基于 **Diffusion Policy**。
- 输入包括：
  - active head camera 的 RGB 图像
  - neck、双臂、夹爪的 proprioception
- 输出包括：
  - neck 的未来动作
  - 双臂的未来动作
  - gripper width

- 所以这篇文章真正关键的一点是：
  - **不是**“加了个可动摄像头”
  - **而是**把 **head movement 本身** 放进了 policy 的 action space

## Teleoperation Flow

### 坐标系和流程，按我的理解可以这样看

| 变量 | 它是什么 | 我怎么理解 |
| --- | --- | --- |
| `W` | robot neck base 定义的 `world frame` | **全系统统一参考系** |
| `W T_H(t)` | 时刻 `t` 头相机在世界坐标系里的位姿 | **机器人真实头相机在哪** |
| `W X(t)` | 世界坐标系下的点云 | **当前 3D 场景缓存** |
| `W T_user(t+k)` | 用户最新头姿映射到世界坐标系后的位姿 | **VR 现在想从哪个视角看** |
| `W T_H(t+K)` | 更长时间后机器人真实更新到的头姿 | **机器人实体慢慢追上的结果** |

### 这套链路最关键的地方

- 遥操作其实有 **两条并行链路**：

| 链路 | 干什么 | 速度 |
| --- | --- | --- |
| **快链路** | 根据用户最新头姿实时渲染 VR 视角 | **快** |
| **慢链路** | 把聚合后的 head motion 发给机器人真实执行 | **慢** |

- 所以它不是：
  - “用户头一动，机器人立刻到位”
- 而是：
  - **“用户先马上看到对的视角，机器人实体随后异步跟上。”**

### `height offset` 到底是什么

- 这个地方原文没有展开公式。
- 更稳妥的理解是：
  - 它像是把 VR 头显位姿映射到机器人 `world frame` 时加上的一个**固定高度校准项**。
- 我觉得最好不要把它理解成：
  - “把头部动作变成相对动作”
- 更像是：
  - **相对头动已经测到了**
  - **然后再把渲染视点的高度对齐到机器人坐标系里**

## Experiments

### 实验总体想看什么

| 方向 | 他们在比什么 | 想回答什么 |
| --- | --- | --- |
| **Camera setup** | active head vs wrist/chest baselines | **主动视角到底值不值** |
| **Visual representation** | DINOv2 vs ResNet-DP vs DP3 | **什么表征更适合这个任务** |
| **Teleoperation interface** | point-cloud rendering vs stereo RGB streaming | **这种 VR 接口到底好不好用** |

### 三个任务

| Task | 论文定义 | 我自己的话 |
| --- | --- | --- |
| **Bag Task** | open bag, peek inside, take out object | **先把遮挡处理掉，再去看里面，再抓** |
| **Cup Task** | find cup, hand over, place on hidden saucer | **不同阶段要换视角，不然会被 shelf 挡住** |
| **Lime & Pot Task** | find lime, bimanual lift, precise align | **要一边搜、一边双臂配合，还要最后对准** |

### Camera setup 结果

- `ViA(active head only)` 比 `active head + wrist` 和 `chest + wrist` 都好。
- 这个结果其实挺有意思，因为它不是“camera 越多越好”。
- 作者的解释是：
  - head view 本身已经够用了，甚至可以算 **task-complete**
  - 额外 wrist views 可能只会带来冗余、噪声和遮挡
  - 在数据不多的时候，模型反而更容易被带偏

### Representation 结果

- 相同 active head 输入下，**DINOv2** 最好。
- 作者的解释是：
  - DINOv2 语义先验更强
  - policy 更容易先把目标“认出来”
  - 然后再做后面的 manipulation

### Teleoperation user study

- `point-cloud rendering` 相比 `stereo RGB streaming`：
  - 采集时间略长
  - 但 motion sickness 明显更低
  - 8 个参与者里有 6 个更喜欢作者这套系统

## Bag Task: 我觉得最值得单独记一下

- 这个 task 很贴合作者前面的 framing，因为它确实把 **遮挡** 和 **主动感知** 绑在一起了：
  - 先开包
  - 再看包里
  - 最后拿东西

- 但这里我会保留一个明显的问号：
  - 这项实验更像是在证明 **active head setup 在遮挡场景里有用**
  - 不一定足够强地证明 **policy 学会了强泛化的在线搜索策略**

### 为什么我会这么想

| 我关心的问题 | 目前正文里看到的情况 |
| --- | --- |
| **训练/测试物体有没有变** | **有**，作者明确写了 train/test objects |
| **bag 的位置/配置有没有像别的 task 那样明确随机化** | **没看见明确写** |
| **会不会只是学到一个固定探头套路** | **我觉得是可能的** |

- 所以对一个单任务 policy 来说，完全可能学成一种固定习惯，比如：
  - *“把包打开后，往某个方向探头看一下。”*
- 这不代表实验没价值。
- 只是说明它更强地支持：
  - **active head 对遮挡场景有帮助**
- 而不是特别强地支持：
  - **模型已经学到了可泛化的 online search policy**

## Overall Takeaway

- **这篇文章最有意思的地方**：它把 `active perception` 从一句容易说空的话，做成了一个**完整系统问题**。
- **它的强项**：
  - 问题 framing 很清楚
  - 系统整合做得完整
  - **`6-DoF neck + 3D rendering + shared observation`** 这条线很顺
- **它还没完全回答的地方**：
  - 有些实验还没彻底排除 **task-specific habit**
  - 尤其是 **Bag Task**，active head movement 也可能部分只是固定套路，不一定已经是强泛化搜索策略

## My Plain-Language Summary

- **如果让我用很口语的话概括这篇文章**：
  - 以前很多机器人方法默认“相机放那儿拍就行了”。
  - 这篇文章在说：**不对，很多任务里“怎么看”本身就是任务的一部分。**
  - 所以作者做了一个会动的头、一个不那么容易把人搞晕的 VR 遥操系统，再让机器人直接从人类示教里学“什么时候该看哪里”。
  - 这个想法我觉得**非常有意思**，而且系统上也真的做顺了。
