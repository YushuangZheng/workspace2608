# 闭环多流策略模块

本目录按照《新方法代码开发计划》组织关系—进度信念驱动方法。阶段一离线任务模型、阶段二关系—进度联合信念推断、阶段三闭环正常执行控制、阶段四对象中心入口守卫与事务提交，以及阶段五主动关系验证、统一关系恢复与任务重入均已实现。阶段六的环境无关顶层策略、序列化、逐周期诊断和 RLBench 适配已经接入，并已完成正常小样本门控和受控故障组件级 pilot；扩大任务、variation、seed 与样本数的论文正式评测仍待执行。

## 阶段一公式到代码的映射

- `state_index.py`：统一 `StateId=(skill_index, local_index)` 和前驱/后继状态图，mode 作为节点内部混合分量；
- `task_model.py`：`ClosedLoopTaskModel`、逐状态 `StateNode`、附加统计序列化及基础 DynaMAC 指纹绑定；
- `task_model_builder.py`：从同一批正常示范对齐构建关系先验、场景因子、边界模型和关系事件；
- `scene_factors.py`：实体构型与实体间相对构型的流形高斯分布，以及留一示范评分；
- `boundary_model.py`：本地完成、边界关系条件、边界场景条件和多臂事务元数据；
- `relation_events.py`：事件级 LINK 恢复锚点、未确认 LINK_PENDING 模板及 UNLINK 合法重入元数据；
- `query_adapter.py`：不推进基础策略时钟的指定状态动作查询。

逐状态关系先验使用：

```text
局部相对位姿样本
→ 复用 DynaMAC 流形高斯拟合
→ 保存跨示范 GMSD 连接分数并生成主关系先验
→ 有足够末端运动时，用逐示范相邻状态共动残差单向排除假 linked
→ 保存 P(external), P(linked)
```

跨示范协方差仍是离线关系先验的主判据。共动检查对每条示范计算末端相邻运动量和末端—参考系相对位姿残差；只有末端运动达到 `3e-7` 的加权平方量后，才把残差不超过 `1e-5` 且不超过末端运动量 `10%` 的示范计为共动支持。该证据只能降低 `P(linked)`，不能单独创建 LINK。动作激励不足时保留 GMSD 关系先验，但事件检测读取中性值 `0.5`，因此不会在无新证据时发起正式 LINK/UNLINK。若跨示范 GMSD 的 external→linked 假设、全部示范的闭合模板和 LODO 位置一致性都成立，但闭合后最短稳定窗口完全没有可观测运动，则保存 `LinkPendingCandidate`；它不是第三种关系状态，也不生成正式锚点或 `link_origin`，但会按正式 LINK 的同一口径保存一份尚未激活的事件前局部恢复轨迹模板。同一闭合事件在锚点窗口内稍后已获得正式 LINK 时，不重复保留 Pending。

每个 `StateNode` 保存全部 mode 的先验、流分布、关系先验、夹爪命令和场景因子分量，但 mode 不增加任务进度节点数量。场景因子在相同技能、相同模态内构建。离线构建从半径 `r=0` 开始，最多扩展到 `r=2`，并选择通过跨示范支持覆盖和稳定性检查的最小半径；邻域不跨技能边界。节点因子只读取实体自身附带的内部欧氏构型字段，普通刚体不强制建立节点因子；边因子只来自局部共同激活的任务参考系和任务接口提供的直接结构绑定，不包含虚拟技能帧或对侧末端，也不生成全场景对象完全图。若关节实体已经提供内部构型字段，该字段作为规范表示，不再同时生成与其结构主体等价的直接结构边。

双臂构建把 Eq.6 选中的对侧末端视为候选而非既成执行依赖。逐 `(skill, mode)` 只有两类证据可以保留该连续流：本臂活动 `link_origin` 与对臂正式 LINK/`LINK_PENDING` 指向同一共享或交接实体；或者该模态没有普通物理任务参考系、对侧末端是唯一非虚拟目标，并具有稳定非零定向相对几何。普通物体/目标流已经解释动作且不存在关系路径时，对侧末端只因示范同步被选中，便从闭环 sidecar 的 PoE、进度轨迹和本地完成目标中删除。冻结 V4 checkpoint 不改写，筛选记录保存在 `builder_config.peer_execution_dependencies`；schema v4 逐状态保存筛选后的 `selected_frames/mode_selected_frames`，加载时不会恢复成旧 Eq.6 全集。纯同步不产生动作依赖、边界依赖或事务；真实就绪仍由统一关系/场景守卫表达。

逐状态留一验证以归一化机器人轨迹作为局部状态识别基线，并叠加已经保留的场景因子，不加入名义时间先验或任何关系评分；通过前向选择比较候选加入前后的局部进度识别 margin，只保留至少 4/5 留一折正增益的因子。与机器人轨迹或关系变量不重复由场景因子的定义和候选排除保证。边界场景条件使用同一节点/边因子定义，根据下一技能参考系、边界关系实体及其直接结构实体限定边界专用范围，不要求先进入某个逐进度节点；硬条件必须覆盖全部正常留一示范，并保存 `[0,1]` 马氏兼容度阈值。双臂构建时，若一臂筛选后保留的本地完成目标以对侧末端为参考且正常边界稳定对应，同一规范化双臂末端边会作为对侧边界的普通场景候选；它只进入对应 `BoundaryModel.scene_conditions`，仍使用同一留一覆盖、阈值和 `EntryGuard`，不扩散为逐状态因子，也不新增协作守卫或对侧进度门控。跨臂因子直接从各臂 `RuntimeFeatures.ee_pose` 与对侧参考观测解析，不要求 benchmark 复制本臂末端别名。边界关系条件独立构建，不参与场景因子的留一筛选。

边界离线构建先生成关系事件，再以整个终止窗口、最终状态目标分布和当前技能由本臂直接 LINK/UNLINK 的关系目标分布构成 `LocalCompletionModel`。边界关系守卫候选只来自下一技能所有 affected arms 的 selected frames，必须同时通过总体支持度、观测可用率和全部正常留一折一致性；随后删除本臂及同一事务其他臂 LocalDone 已保证的关系。下一技能已选参考系即使在旧 DynaMAC 固定参与掩码中暂时 inactive，仍属于边界候选实体。阶段一只保存这些模型，不计算在线 `LocalDone`，也不标定本地完成阈值和连续确认周期。

离线 LINK/UNLINK 候选由同一 mode 的全部正常示范共同拟合的逐状态关系先验产生，并在完整任务联合先验序列上通过迟滞和稳定窗口检测，因此事件前后稳定窗口可以跨越技能边界。按示范留一只做稳定性检查：每折用其余 `N-1` 条示范重新拟合联合先验，要求至少 80% 留一折重现同类运动学事件。默认直接比较事件位置；若各折都已经检测到 LINK、但闭爪后的首段有效携带运动出现时间不同，则允许用同一锚点窗口内跨示范对齐的因果闭爪确认这些证据属于同一次事件。夹爪不能创建 LINK，正式位置仍取全部示范联合模型的运动学检测结果。通过后使用全部对应示范拟合最终锚点或脱离信息。夹爪命令不参与关系先验拟合，也不能单独创建事件；它只做模板完整性校验：LINK 锚点必须覆盖闭合命令，UNLINK 必须发生在最近一次打开命令之后。LINK 与 Pending 的轨迹最多保留16个对齐状态；若同一机械臂在窗口内更早存在其他参考系或同一参考系的 linked→external，则从最近一次 UNLINK 的下一状态开始截断且不向前回填，避免混入上一段抓取轨迹。轨迹可跨技能边界。正式 LINK 后至正式 UNLINK 前的部署关系期望由事件状态机保持连续，原始 GMSD 分数保留审计，但不会再用局部抖动清除 `link_origin`；写回状态节点的是 `[0.3,0.7]` / `[0.7,0.3]` 软先验而非 one-hot，在线证据仍可推翻本次关系。没有正式事件或 Pending 来源的孤立低协方差 linked 脉冲只保留在 `demo_relation_scores`，部署先验保持 soft external；Pending 候选区可以保留 soft linked 假设供后续验证请求判断，但不传播正式 `link_origin`，支持结束后即恢复 soft external。UNLINK 的合法重入状态和局部脱离目标也允许进入后继技能。双臂事务组要求全部正常示范中的边界稳定同步，并且双方各自参与同一 linked 实体、共同建立一个由全部示范确认且跨越该边界的 LINK，或共享同一硬场景条件。只观察对臂的定向守卫和单独的 Pending 不创建事务组。

`ClosedLoopTaskModel.save()` 使用 sidecar schema v4 保存筛选后的逐模态流集合以及关系、Pending 事件、场景、边界和恢复统计；schema v3 仍可只读加载。原有流均值、协方差和夹爪模型继续从绑定的 DynaMAC checkpoint 读取。

## 阶段二在线信念推断

阶段二以旁路方式建立固定的单周期状态估计链：

```text
RuntimeObservation
→ RuntimeFeatures
→ action-after progress prior
→ external/linked relation posterior
→ progress posterior
```

`RuntimeObservation` 统一提供末端位姿、任务参考系位姿、夹爪状态、上一命令目标、上一末端状态、可见性和跟踪可靠性。节点场景因子额外读取实体随观测附带的 `entity_configurations`；只有刚体真值位姿的平台可将该字段留空。RLBench 低维仿真真值未显式提供可见性和跟踪可靠性时，已有位姿默认使用 `visible=True`、`reliability=1`，字段和运行门控仍完整保留。

`RuntimeFeatureBuilder` 每个控制周期只计算一次实际末端运动、命令目标运动、参考系世界运动、末端—参考系相对位姿、相对运动残差、夹爪变化、动作激励和逐关系信息权重。只有当前与上一周期参考系均可见、跟踪可靠且机器人产生实际动作响应时，关系观测才具有非零信息权重；因此机器人与对象同时静止不会因为相对残差为零而被强判 linked。

`ProgressPriorBuilder` 将上一周期进度后验传播到未完成状态、上一周期实际发送的动作目标和稍微提前到达的直接后继，默认权重分别为 `0.20/0.65/0.15`，随后只保留名义状态附近的局部候选。`reference_state` 是 DynaMAC `query_state` 的目标状态，不能再额外偏移一个后继。跨技能边界默认禁止；阶段四事务提交时，跨界前进度质量与执行游标原子投影到下一技能入口，之后正常进度不得返回已提交边界之前。阶段二本身不评估守卫，也不自行提交任务时钟。

`RelationFilter` 对每个物理 arm-frame 关系仅维护 `[P(external), P(linked)]`。它依次组合关系持久转移、由名义进度先验加权的离线示范关系先验，以及动作条件化相对运动似然。`Unknown` 只是不可靠周期的决策状态，不进入二元概率向量。首次缺少动作激励、不可见、跟踪不可靠、后验熵过高或最大概率不足时输出 `Unknown`；已经由有效运动证据确认的 external/linked，在参考系继续可见、可靠且后验仍以足够置信度支持相同状态时，可跨越暂时缺少激励的静止周期保持判定。不可见、不可靠或后验冲突会使连续性缓存失效，参考系重现后不会在无新运动证据时“复活”旧判定。虚拟技能帧不进入关系滤波。

`StateEvaluator` 对每个候选状态分别保存机器人局部轨迹、稀疏场景构型和关系兼容度的 log-space 分项。可观测、可靠的物理流按 $q_t(\mathrm{external})$ 连续降权参与机器人轨迹匹配；离散决策为 `Unknown` 不会删除该连续位姿证据，但不可观测或不可靠时权重仍为0。关系兼容度则仍只以可靠的非 Unknown 关系后验计算。虚拟技能帧只按观测可靠性参与。连续高斯轨迹和场景因子用于在线进度校正的分数为峰值归一化支持度：

$$
\log\widetilde p(x\mid s)=-\frac12d_M^2(x,s).
$$

协方差仍决定马氏距离中的方向和尺度，但不同状态不再因 `logdet` 较小而凭绝对密度峰值获得额外优势。完整原始 `log p` 不参与进度后验，仍按每个帧/因子/mode 保存原始对数密度、马氏距离平方、协方差 `logdet`、维数和模态权重用于审计。离散 external/linked 关系的原始兼容度仍为 $q_t\cdot\pi_s^{\mathrm{demo}}$，并原样参与进度后验；用于 `NO_PLAUSIBLE_STATE` 绝对门限的 `normalized_explanation_score` 将其除以 $\max_z\pi_s^{\mathrm{demo}}(z)$。事件感知候选扩展和恢复后重入使用同一峰值尺度，并要求在线可靠离散决策与非持平离线先验具有相同 external/linked 方向；任一必要关系方向相反时该候选的 `relation_state_compatibility=0`。这是对原绝对关系阈值评分量的统一替换，不叠加新门控。原始、峰值归一及方向约束后的关系支持均保留审计。

多流机器人支持还使用动作 PoE 的联合可达峰值归一。事件感知候选扩展和恢复重入的机器人绝对阈值只读取 `robot_peak_normalized_compatibility`；逐流几何平均 `robot_compatibility` 仅供审计。候选扩展0.01和重入0.001均是 $[0,1]$ 联合可达峰值尺度，不能再解释成原始高斯密度或逐流边缘峰值阈值。

`ProgressFilter` 在局部候选上一次性计算：

$$
\beta_t(s)\propto
\bar\beta_t(s)
\widetilde L_t^{\mathrm{robot}}(s)
[\widetilde L_t^{\mathrm{state}}(s)]^{\lambda_x}
[C_t^{\mathrm{rel}}(s)]^{\lambda_r}.
$$

输出 `ALIGNED/FORWARD_REALIGNMENT/BACKWARD_REALIGNMENT/LOW_CONFIDENCE/NO_PLAUSIBLE_STATE`。没有候选达到最低支持度时，后验保留动作后名义先验并报告 `NO_PLAUSIBLE_STATE`，不把一组近零支持分数强行归一化成虚假的高置信结论。

`BeliefUpdater` 每个递增 tick 严格执行一次 `progress prior → relation posterior → progress posterior`，并返回含统一运行特征、关系估计、进度估计、关系变化、局部/扩展候选及全部 `CandidateScore` 的 `ClosedLoopBelief`。只有出现可靠二元关系变化且局部窗口无可解释状态时，才依次查询未来关系兼容状态段；未来候选还必须同时满足机器人、场景（若状态要求）和关系支持，且不能越过未放行边界。该机制允许正常动作提前完成前移，同时避免仅凭掉落产生的 external 关系跳到未来释放状态。

阶段二全部运行阈值集中在 `configs/closed_loop_belief.json`，并通过配置默认值往返测试防止代码和文件漂移。

## 阶段三动态角色与闭环正常执行

阶段三将 `ClosedLoopBelief` 接入动作查询与任务时间控制，但仍只处理 `TASK` 模式下的技能内正常执行：

```text
ClosedLoopBelief
→ 当前 mode 已选流的动态角色
→ 运行期精度权重
→ HOLD / REALIGN / ADVANCE
→ 指定 reference_state 的只读动作查询
```

`FrameRoleRouter` 为当前状态和当前 mode 中已通过离线任务参数选择的流分配动作角色；同时继续监控当前进度具有正式 `link_origin` 的物理关系，即使对应专家已不再被当前 mode 选择。虚拟技能帧继续作为固定执行参考系，不进入关系判断；物理流根据进度后验加权的期望关系、在线关系后验和跟踪可靠性分为：

- `EXECUTE`：期望和实际均为 external，权重为跟踪可靠性与 external 后验的乘积；
- `MONITOR`：期望和实际均为 linked，正常 PoE 权重为零，但阶段二关系滤波继续观测；
- `RECOVER`：期望和实际形成可靠二元失配，权重为零、阻断推进并输出恢复意图；
- `DEFER`：实际关系 Unknown 时通常阻断推进并沿用上一可信权重；仅对无历史、可见可靠且示范期望与二元后验都明确支持 external 的启动周期，允许使用 $q(\mathrm{external})$ 安全降权产生自然激励。进度加权的期望关系暂时模糊时同样保持 `DEFER`，但不把它冒充为实际 $q_t$ Unknown，不产生失配且不单独阻断进度。

关系滤波持续估计全部物理关系，因此路由器会缓存尚未被当前技能选中的物理流最近一次可靠 external/linked 权重。这样新流首次被选中但单帧激励不足时，可以沿用此前可信证据，而不会因为暂时 Unknown 突然失去所有伺服专家。普通未选流仍不分配角色；具有正式 `link_origin` 的未选关系只允许 `MONITOR/RECOVER/DEFER`，其 `selected_offline=False` 且不会进入 `execution_weights`，因此不会重新启用 Eq.6 已关闭的 PoE 专家。

`WeightedPoEExecutor` 只把非负运行权重传入现有 DynaMAC PoE：$w_{f,t}$ 只缩放专家精度，不缩放或修改专家均值。全 1 权重保持冻结基线语义，零权重等价于移除对应专家；所有当前执行权重均为零时返回不可用结果，不启用未选流或无权重回退。

正式 LINK 的本次执行确认允许使用阶段一为该 occurrence 学到的完整 `linked_entry_states` 自然运动区间，不再使用全局固定状态数。该区间只有从正式 LINK 的直接前驱沿拓扑进入时才激活，reset、恢复重入或任意跳转到 linked 状态不能重新取得宽限。事件直接前驱上的 pre-LINK external/Unknown 不能阻断建立关系所必需的后继命令；进入区间后，相应物理流保持零权重 `DEFER`，先提交抓取并执行有界自然携带，再由后续运动证据确认。当前查询状态可以比 beta 领先一个周期，因此区间内的实体即使不是当前 Eq.6 动作流也会被加入零权重关系监控，正式 occurrence 自身提供 linked 目标语义，但该实体不会重新进入动作 PoE。末入口状态在 beta 尚未到达时继续完成自然动作；beta 已确认末状态后再次查询同一状态才结束宽限，可靠 external 按正常失配进入恢复。一旦确认 linked，后续掉落也不能重新进入宽限。

`ClosedLoopExecutionController` 是 `nominal_state / estimated_state / reference_state` 的唯一正常执行提交入口：

- `HOLD` 保持旧 `reference_state` 并继续查询该状态进行闭环伺服；
- `REALIGN` 将引用索引直接改到可信的 `estimated_state`，不反向回放中间轨迹；
- `ADVANCE` 在可信进度已到达当前引用时查询同技能直接后继，或在观测已提前到达该直接后继时提交它；一个周期最多提交一条拓扑边。

`NO_PLAUSIBLE_STATE`、关键关系 Unknown、可靠关系失配、当前状态尚未提交的离散夹爪命令或跨技能候选都会 HOLD。当可信 `estimated_state` 仍落在当前 `reference_state` 之前时，通常视为当前目标未到达并继续伺服原目标；上一命令与末端的连续高斯相容度只保留为审计信息，不另立一个进度硬门控。若 beta 已确认当前引用，但观测中的夹爪状态还未体现该 `StateNode` 的命令，则保持同一引用并提交一次该离散动作，下一周期再按原进度逻辑判断。跨边界夹爪变化通常由提交周期执行；若正式 LINK/LINK_PENDING 证明目标入口闭爪是关系建立边动作，且引用和 beta MAP 已进入终止窗口、其他守卫已满足，边界层可保持源 StateId/LocalDone 不变而准备闭爪，准备期间离散完整性检查闭爪后状态。普通边界和释放没有该准备路径。该规则防止平滑 ADVANCE 跳过抓取/释放，不读取 IK 的 `reached/progressed/stopped`，也不建立第二套任务进度。对 `LOW_CONFIDENCE` 和高置信 `BACKWARD_REALIGNMENT`，执行器保留阶段二原始后验、MAP 与标签，只把相同技能内局部连续、动态角色与关系语义一致、夹爪一致且 PoE 位姿目标两两兼容的状态聚为控制等价类。包含 MAP 状态的类概率质量和全部类质量的归一化熵继续使用阶段二原有置信度/熵阈值；`LOW_CONFIDENCE` 通过时解除该项阻断；`BACKWARD_REALIGNMENT` 还要求 MAP 恰为当前引用的直接前驱且二者同属已接受等价类，才可把当前引用按控制等价完成。两者都只推进一个同技能直接后继。多状态落后、未通过等价检查的落后状态、关系阻断、无正权重流和入口守卫照常生效。等价类不跨技能，动作兼容度使用统一 `minimum_action_equivalence_compatibility`，不另加任务专用门限。

`MismatchTracker` 分别累计无正常候选、关系持续失配、持续 HOLD 和进度停滞，只在达到配置周期时输出可审计 `MismatchEvent`；阶段三不执行恢复动作。当当前事件命中 `LinkPendingCandidate`，后续正常状态或下一边界确实需要 linked，当前关系 Unknown，且当前与上一观测均可用、可靠但动作激励不足时，角色层输出 `RelationVerificationRequest`。本阶段只产生请求，不移动机械臂、不激活 Pending 模板，也不改变任务进度。

阶段三阈值集中在 `configs/closed_loop_execution.json`。正常 StackWine V4 只读回放覆盖动态路由和运行权重，确认无恢复误报、每个查询均有可执行动作，并覆盖新选流 Unknown 时沿用此前可信权重。

## 阶段四入口守卫与事务提交

阶段四在阶段二、三完成同一控制周期的信念更新和正常动作查询后，读取阶段一的 `BoundaryModel`，计算本臂局部完成度：

$$
R_{i,b,t}^{\mathrm{local}}
=P_{i,b,t}^{\mathrm{end}}
L_{i,b,t}^{\mathrm{goal}}
C_{i,b,t}^{\mathrm{own-rel}}.
$$

`P_end` 是本臂进度后验落在完整终止窗口的概率；`L_goal` 是当前 mode 全部最终目标流的联合峰值归一高斯支持；`C_own-rel` 是本臂必要关系后验与离线终止关系分布内积的最小值，空集合为1。任一本臂必要关系 Unknown、不可见或不可靠时，本周期 `LocalDone=False`，但不把它直接升级为恢复异常。

`EntryGuard` 再以逻辑与检查边界关系目标概率和稀疏场景条件。关系条件必须为非 Unknown、可靠且目标后验严格大于阈值；场景条件必须同时满足观测可靠性和边界支持兼容度。所有条件必须连续满足该边界的 `H` 个真实控制周期。必要 linked 关系仅因运动激励不足而 Unknown 且命中 `LinkPendingCandidate` 时，只生成 `RelationVerificationRequest` 并继续阻断边界；探测移动留给阶段五。

`MultiArmBoundaryController` 要求所有机械臂提供同一 tick 的 pre-action 信念快照，先计算全部 `TransitionRequest`，再交由 `TransitionTransactionCoordinator` 统一提交。独立边界可以异步提交；同一事务组只有在全部成员同周期放行时才原子提交。等待期间不改变已就绪机械臂的当前引用状态，阶段三仍继续用最新对象位姿查询当前状态动作并动态伺服。提交结果通过 `permitted_boundaries` 明确传回下一周期的阶段二候选传播。

运行参数按任务保存在 `configs/closed_loop_boundary/`。`theta_local` 和 `H` 只由五条正常示范的原始 RLBench 控制周期回放标定，并且只读取本地完成分数，不用守卫是否已经放行筛选本地正样本。对每个边界，`H` 取正常进度进入末端窗口前最长偶发就绪连续段加一，再由正常末端保持验证可持续性，不附加全局最短时长。当前30个边界使用 `H=1`、9个使用 `H=2`、WipeDesk `1→2` 使用 `H=3`。正式复核覆盖40个边界、200个边界×示范实例，边界前提前放行为0，正常末端保持后的200次决策均符合放行或定向等待语义；LiftTray `1→2` 的真实事务组在5/5条正常示范中双方共同就绪。当前40个边界统一使用 `evaluations/phase4_boundary_calibration/results/v5`；其中 `L_goal` 将当前 mode 的全部最终目标流变换到世界坐标系后，以精度加权 PoE 构造单一联合末端目标，再对真实末端位姿评分一次，不重复相乘多个坐标表达的边缘支持。

## 阶段五主动验证、恢复与重入

`VERIFY_LINK` 的固定反向探测还要求最近 TASK 实际末端轨迹能定义非零接近方向。若轨迹静止、样本不足或净位移近似为零，本周期保持 `TASK/HOLD`，不启动无方向的验证、不抛出策略异常。这是原探测模板的可定义前提，不是新状态或任务门控。

阶段五新增与 StateId、MiDiGaP/DynaMAC 轨迹 mode 相互独立的顶层 `ExecutionMode={TASK,VERIFY_LINK,RECOVERY}`，并为两种非 TASK 模式提供 `BeliefUpdater.update_frozen()`：正常进度后验、名义状态和引用状态保持不变，真实末端/参考系运动仍进入同一关系滤波器。阶段六顶层策略必须在非 TASK 周期使用该入口。多臂系统中任一臂进入非 TASK 后，所有正常进度提交和正常入口事务均冻结；其他机械臂只伺服各自已经提交的当前参考目标，辅助过程返回 TASK 后才恢复正常更新。

`VERIFY_LINK` 只接受阶段三或阶段四产生、且引用当前 `LinkPendingCandidate` 的必要 linked 请求。边界准备中的 Pending 可在源 StateId 尚未提交时按确切事件标识路由，但必须先从真实观测确认准备闭爪已发生，不允许一边开爪一边探测。验证期间 beta/StateId 仍冻结在源边界；仅对请求指定的 arm-frame，原关系滤波器改用该确切 Pending `candidate_state` 已学习的 soft 关系先验，其他关系仍按冻结 beta 混合。该上下文不修改进度、不提供物理证据，也不新增关系状态。其他入口条件仍是关系尚未可靠确认、参考系成对可用且跟踪可靠、信息权重不足。控制器保持入口朝向和夹爪命令，按最近 TASK 实际末端轨迹估计接近方向，以固定速度反向探测；刚进入模式时属于上一条 TASK 指令的运动不计为验证证据，只累计验证动作发出后下一周期观测到的末端—参考系成对响应。当前至少需要3个可靠动作响应和既有最小探测运动，并用窗口共动残差比 `sqrt(sum(||delta_p_f-delta_p_e||^2)/sum(||delta_p_e||^2))` 的统一0.85阈值给出响应方向；只有该方向与原二元关系滤波器当前非 Unknown、有效信息决策一致时，关系才稳定成立。窗口不创建第三种关系状态，也不替代原滤波器。稳定关系、探测时间上限或安全约束任一成立便停止外移；时间上限只计算探测阶段，之后仍按记录的实际路径反向返回，仅当返回路径本身不安全时输出结构化失败。返回后统一回到 TASK；若已稳定确认 linked/external，则把稳定周期的原始二元后验精确提交回同一个在线关系滤波状态，不做 one-hot 化或固定置信度增强，避免返回运动的短时残差抹掉有效确认。稳定验证为 linked 时只在当前 episode 激活 Pending 恢复模板并建立运行时来源，不修改离线任务模型；验证为 external/Unknown 时不激活。若边界准备后稳定判为 external，恢复意图保留同一事件标识并只读使用其未激活恢复模板，直到恢复获得 linked 证据才激活来源。同一事件只有关系状态、任务状态或抓取事件变化后才能再次验证。

`RECOVERY` 将可靠二元关系失配转换为事件级 `LINK(f)` 或 `UNLINK(f)` 目标。同一机械臂的所有 UNLINK 先于 LINK，以先释放夹爪资源。LINK 从当前正式 `link_origin`、当前 episode 已验证的 Pending 来源，或“已到达同一 Pending occurrence 且必要 linked 守卫稳定判为 external”的未激活修复模板解析锚点；第三种来源只有在恢复后获得有效 linked 证据时才转为 episode 运行时来源。跨臂定向守卫在对臂尚未到达 Pending 时仍只是等待，不会误触发恢复。执行 LINK 前先在参考系可观测时等待其连续稳定，再锁定本次恢复使用的单一世界位姿并据此实例化完整局部锚点；预抓取/开夹爪阶段若参考系相对锁定位姿显著移动，则废弃旧实例并重新等待稳定，夹爪闭合后则允许对象随末端运动。该规则避免追逐仍在滑动的掉落物体，不读取任务名或 StateId。恢复协方差统一加 `lambda_rec I`；UNLINK 统一执行打开夹爪、局部脱离和 external 验证。只有支持目标关系的有效运动证据才可完成关系目标，反方向或无信息证据不能替代确认。每个路点、关系验证、单目标尝试、整个恢复及重入等待均有明确周期上限和结构化失败。

可靠在线失配不自动创造示范中不存在的反向关系动作。运行时规划只接收能够解析到正式 LINK、允许的 Pending 模板或 UNLINK 元数据的目标；例如关系在正常任务中只建立且从不释放时，提前观察到 linked 不能凭空生成 UNLINK。该意图保留供诊断，动态角色继续安全阻断推进，执行模式保持 `TASK/HOLD`，让同一 `q_t/beta_t` 在下一观测周期继续消解进度—关系不同步；不得因此进入空目标 RECOVERY 或无约束全任务重入。严格规划接口仍对缺失元数据报错，避免开发期静默掩盖任务模型缺口。

关系目标完成后只在事件元数据提供的合法状态中比较当前完整机器人轨迹、稀疏场景和关系解释度；无关系目标的 `NO_PLAUSIBLE_STATE` 恢复可搜索全任务，但同技能之外只允许经入口守卫放行的相邻下一技能，禁止后退或跨多技能直接跳转。正常进度评分仍使用原始示范轨迹协方差；只有恢复后的重入机器人轨迹项复用既有 `lambda_rec`，按 `Sigma_reentry=Sigma+lambda_rec I` 评分，使恢复动作分布与其完成判定具有相同精度口径。重入机器人绝对准入读取联合可达峰值归一支持，逐流兼容度只作审计；重入关系项使用 `relation_state_compatibility`，将峰值归一的软先验支持度与 external/linked 方向一致性统一成单一分数。场景、候选范围、入口守卫和兼容度阈值不变。已通过场景/关系检查但尚未达到正式重入阈值的候选，可使用该候选自身的关系期望和流角色生成对齐动作；不能再用故障时冻结后验的旧关系语义否决该动作。这只是只读动作查询视图，真实 `beta/StateId/reference_state`、mode、角色历史和任务时钟仍冻结。重入选择、进度后验 one-hot 重置、执行 `reference_state` 设置和返回 TASK 只在完整阈值通过后由管理器原子完成，不恢复旧故障时钟。

阶段五配置集中于 `configs/closed_loop_recovery.json`。组件测试覆盖冻结更新、反向探测/原路返回、超时与安全、重复验证抑制、episode Pending、事件锚点实例化、UNLINK、目标排序、硬上限、完整状态重入及模式切换。真实 V4 元数据验收覆盖12个任务/机械臂模型、11个正式 LINK、7个 Pending、3个 UNLINK 和904个状态级 `link_origin`，结果位于 `evaluations/phase5_recovery_acceptance/results/v1`。受控行为 A/B 位于 `evaluations/phase5_behavior_ab/results/v5`：Pending 主动验证和统一关系恢复均从无动作对照的0%提高到100%，任务重入 StateId MAE 从37.0000降至0.6375，且进度冻结、反向探测、原路返回、重复抑制、有界失败和跨技能许可约束均通过。重入机器人兼容度只用240条正常回放从0.01标定为0.001，状态选择239→240、精确选择180→181、错误选择保持59→59；以上均为确定性理想执行器下的组件结果，不等同于完整 RLBench 在线故障恢复成功率，阶段六正在继续完整仿真验收。

## 阶段六顶层策略与 RLBench 适配

环境无关核心位于 `policy.py / config.py / state.py / diagnostics.py / serialization.py`。`ClosedLoopMultiStreamPolicy` 每个 pre-action tick 依次完成统一观测信念更新、动态角色与加权动作、边界事务、主动验证/恢复和状态重入，并以 `act → commit/abort` 保证环境拒绝动作时能够回滚本周期全部内部状态。阶段二至五的算法模块不依赖 RLBench。

RLBench 专用代码位于 `integrations/rlbench/rlbench_closed_loop/`，负责低维观测适配、双臂联合快照、策略进程协议和命令格式；benchmark 执行器位于 `integrations/rlbench/rlbench_dynamac/core/runtime.py` 与 `trac_ik.py`。现有 `direct_evaluate` 通过 `policy_type=closed_loop_multistream` 复用同一评测集和结果 schema，同时保留 `policy_type=dynamac` 的冻结 V4 执行器对照路径。跨技能事务提交周期仍执行源技能最终连续位姿目标，入口夹爪命令直接读取入口 `StateNode`；关系建立闭爪可以由核心边界层作为未提交边动作准备，RLBench 适配器只执行显式授权，不识别任务或 StateId。目标技能虚拟帧仍只在正式提交后的桥接动作下一次真实观测中捕获，下一周期进度质量同时原子投影到入口状态。

闭环策略的 RLBench 执行配置为 `stage6_hybrid_cartesian_executor_v19` / `rlbench-stage6-hybrid-cartesian-continuation-v23`：优先使用 CoppeliaSim 当前关节种子伪逆，再依次使用 bounded TRAC-IK Distance、有界 SE(3) 小步延续、碰撞感知优先且必要时放宽的采样 IK和碰撞感知线性路径。远目标可直接进入碰撞感知 RRTConnect；近目标首次求解不调用全局规划，只有同一策略目标被真实物理反馈确认停滞并耗尽本地求解层级后，才依次尝试碰撞感知 RRTConnect、碰撞放宽线性路径和最终的碰撞放宽 RRTConnect。路径生成成功后若真实末端仍无改善，也会跨周期跳过该路径族；任一备用解产生真实进展后重置层级并返回策略重新观测。最终非线性放宽层只在此前所有局部、碰撞感知和线性层均失败或物理停滞后启用，仍受统一 planner 时间/采样预算约束，不读取任务名或 StateId。每段移动后重新观测真实末端，内部子目标不提交为策略状态；bounded TRAC 请求和执行后物理完成使用 0.5 mm / 0.05°，外层策略目标接受使用 1 mm / 0.1°，只有原始目标进入外层包络才报告 `reached`。真实改善、物理停止和全链耗尽分别返回可区分结果；2 mm / 1° 仅是外部运动链 FK 对齐审计上限。已施加任务命令的引用和下一周期观测进入既有进度模型；`reached/progressed/stopped` 本身不修改进度先验、`StateId` 或 `reference_state`。`progressed/stopped` 不提交闭环绝对目标完成。动作事务只接受 `primary_action_status` 与 `primary_action_applied`：主动作未施加、改用关节保持时明确提交 `stopped/false`，其他字段由统一未知字段检查拒绝。规划路径若在有界物理预算内尚未结束，执行器会在返回闭环观测前释放其 RML 运动句柄，避免跨动作或跨 episode 保留已放弃路径的仿真资源；这只管理执行资源生命周期，不构成新的任务门控。TASK 夹爪按每臂的进度对齐、当前状态离散动作完整性和边界事务授权提交，固定笛卡尔包络不作为第二套任务完成判据；冻结 DynaMAC 对每个正式提交的动作事务保持原固定时钟推进并显式授权其夹爪值，执行器标签不回滚或延迟基线时序。只有没有任务级授权来源的非 TASK 辅助命令保留 `reached` 后提交的物理顺序。双臂先完成全部候选准备再推进共享物理时钟，但无关夹爪不强制捆绑。执行器不读取任务名或 StateId，可随 benchmark 替换而不改变阶段一至五算法。

逐周期诊断除原始进度后验、关系后验、角色、PoE、边界和恢复信息外，还保存 `ControlEquivalenceAssessment`：原始进度标签、控制等价状态集合、聚合置信度、类别熵、最小动作兼容度和是否接受。该记录不改写阶段二信念，并区分 `control_equivalent_progress_uncertainty` 与 `control_equivalent_backward_realignment`。

阶段六的当前结果入口唯一为 `evaluations/phase6_formal_evaluation/`。小样本正常回放、定向故障和机制修复运行只用于正式启动前的可行性门控；旧执行器产生的 pilot、诊断与预正式结果不作为当前交付保留，也不得与正式矩阵混合。论文成功率和统计结论只读取当前 v19/v23 协议完整生成的结果。

## 查询语义

`DynaMAC.query_state(observation, state_id, stream_weights, mode_index=...)` 是只读查询；`state_id` 严格只有 `(skill_index, local_index)` 两个进度分量，mode 只能通过独立的 `mode_index` 提供。不提供 `mode_index` 时读取 reset 已选择的该技能模态：

- `stream_weights=None` 使用冻结 baseline 的 Eq.5/Eq.6 参与掩码；
- 显式权重映射使用 `w * precision`，零权重等价于移除专家；
- 显式映射未列出的流权重为零；
- Eq.6 未选择的模态流不能被运行权重重新启用；
- 查询不会改变技能索引、时间索引、模态路径或虚拟帧状态。

查询尚未由 baseline 游标捕获的技能状态时，闭环控制器应通过 `DynaMACObservation.frames` 提供已经由其运行快照捕获的 `virtual_skill_*` 帧。
