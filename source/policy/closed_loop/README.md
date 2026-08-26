# 闭环多流策略模块

本目录按照《新方法代码开发计划》组织关系—进度信念驱动方法。阶段一离线任务模型、阶段二关系—进度联合信念推断和阶段三闭环正常执行控制已经实现；当前代码不包含阶段四入口守卫运行时或阶段五主动验证/恢复动作控制器。

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

逐状态留一验证以归一化机器人轨迹作为局部状态识别基线，并叠加已经保留的场景因子，不加入名义时间先验或任何关系评分；通过前向选择比较候选加入前后的局部进度识别 margin，只保留至少 4/5 留一折正增益的因子。与机器人轨迹或关系变量不重复由场景因子的定义和候选排除保证。边界场景条件使用同一节点/边因子定义，根据下一技能参考系、边界关系实体及其直接结构实体限定边界专用范围，不要求先进入某个逐进度节点；硬条件必须覆盖全部正常留一示范，并保存 `[0,1]` 马氏兼容度阈值。边界关系条件独立构建，不参与场景因子的留一筛选。

边界离线构建先生成关系事件，再以整个终止窗口、最终状态目标分布和当前技能由本臂直接 LINK/UNLINK 的关系目标分布构成 `LocalCompletionModel`。边界关系守卫候选只来自下一技能所有 affected arms 的 selected frames，必须同时通过总体支持度、观测可用率和全部正常留一折一致性；随后删除本臂及同一事务其他臂 LocalDone 已保证的关系。下一技能已选参考系即使在旧 DynaMAC 固定参与掩码中暂时 inactive，仍属于边界候选实体。阶段一只保存这些模型，不计算在线 `LocalDone`，也不标定本地完成阈值和连续确认周期。

离线 LINK/UNLINK 候选由同一 mode 的全部正常示范共同拟合的逐状态关系先验产生，并在完整任务联合先验序列上通过迟滞和稳定窗口检测，因此事件前后稳定窗口可以跨越技能边界。按示范留一只做稳定性检查：每折用其余 `N-1` 条示范重新拟合联合先验，要求至少 80% 留一折在位置容差内重现同类事件；通过后仍以全部示范联合模型确定事件位置，并使用全部对应示范拟合最终锚点或脱离信息。夹爪命令不参与关系先验拟合，也不能单独创建事件；它只做模板完整性校验：LINK 锚点必须覆盖闭合命令，UNLINK 必须发生在最近一次打开命令之后。LINK 与 Pending 的轨迹最多保留16个对齐状态；若同一机械臂在窗口内更早存在其他参考系或同一参考系的 linked→external，则从最近一次 UNLINK 的下一状态开始截断且不向前回填，避免混入上一段抓取轨迹。轨迹可跨技能边界。正式 LINK 后至正式 UNLINK 前的部署关系期望由事件状态机保持连续，原始 GMSD 分数保留审计，但不会再用局部抖动清除 `link_origin`；写回状态节点的是 `[0.3,0.7]` / `[0.7,0.3]` 软先验而非 one-hot，在线证据仍可推翻本次关系。没有正式事件或 Pending 来源的孤立低协方差 linked 脉冲只保留在 `demo_relation_scores`，部署先验保持 soft external；Pending 候选区可以保留 soft linked 假设供后续验证请求判断，但不传播正式 `link_origin`，支持结束后即恢复 soft external。UNLINK 的合法重入状态和局部脱离目标也允许进入后继技能。双臂事务组只为示范边界稳定同步且双方共享 linked 实体或同一硬场景条件的边界生成。

`ClosedLoopTaskModel.save()` 使用 sidecar schema v3 保存新增关系、Pending 事件、场景、边界和恢复统计。原有流均值、协方差和夹爪模型继续从绑定的 DynaMAC checkpoint 读取。

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

`ProgressPriorBuilder` 将上一周期进度后验传播到未完成状态、正常后继和稍微提前完成状态，默认权重分别为 `0.20/0.65/0.15`，随后只保留名义状态附近的局部候选。跨技能边界默认禁止；阶段四以后只有把已经放行的 `BoundaryId` 作为 `permitted_boundaries` 传入时，阶段二才允许先验或候选扩展进入下一技能首状态。阶段二本身不评估守卫，也不提交任务时钟。

`RelationFilter` 对每个物理 arm-frame 关系仅维护 `[P(external), P(linked)]`。它依次组合关系持久转移、由名义进度先验加权的离线示范关系先验，以及动作条件化相对运动似然。`Unknown` 只是不可靠周期的决策状态，不进入二元概率向量。首次缺少动作激励、不可见、跟踪不可靠、后验熵过高或最大概率不足时输出 `Unknown`；已经由有效运动证据确认的 external/linked，在参考系继续可见、可靠且后验仍以足够置信度支持相同状态时，可跨越暂时缺少激励的静止周期保持判定。不可见、不可靠或后验冲突会使连续性缓存失效，参考系重现后不会在无新运动证据时“复活”旧判定。虚拟技能帧不进入关系滤波。

`StateEvaluator` 对每个候选状态分别保存机器人局部轨迹、稀疏场景构型和关系兼容度的 log-space 分项。机器人轨迹只由可靠、非 Unknown 且按 $q_t(\mathrm{external})$ 加权的物理流参与；关系兼容度同样以可靠的非 Unknown 关系后验计算。虚拟技能帧只按观测可靠性参与。连续高斯轨迹和场景因子用于在线进度校正的分数为峰值归一化支持度：

$$
\log\widetilde p(x\mid s)=-\frac12d_M^2(x,s).
$$

协方差仍决定马氏距离中的方向和尺度，但不同状态不再因 `logdet` 较小而凭绝对密度峰值获得额外优势。完整原始 `log p` 不参与进度后验，仍按每个帧/因子/mode 保存原始对数密度、马氏距离平方、协方差 `logdet`、维数和模态权重用于审计。离散 external/linked 关系的原始兼容度仍为 $q_t\cdot\pi_s^{\mathrm{demo}}$，并原样参与进度后验；只有用于 `NO_PLAUSIBLE_STATE` 绝对门限的 `normalized_explanation_score` 将其除以 $\max_z\pi_s^{\mathrm{demo}}(z)$。这使软关系先验的最佳匹配上限统一为1，不改变原始关系区分比例、关系滤波、动态角色或 PoE。原始和峰值尺度归一化后的关系支持均保留审计。

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

`ClosedLoopExecutionController` 是 `nominal_state / estimated_state / reference_state` 的唯一正常执行提交入口：

- `HOLD` 保持旧 `reference_state` 并继续查询该状态进行闭环伺服；
- `REALIGN` 将引用索引直接改到可信的 `estimated_state`，不反向回放中间轨迹；
- `ADVANCE` 仅在 `nominal_state == estimated_state` 且它是旧引用的同技能直接后继时，把引用提交到该状态，一个周期最多一次。

低置信度、无可解释状态、关键关系 Unknown、可靠关系失配、后继未就绪或跨技能候选都会 HOLD。跨技能提交必须等待阶段四入口守卫，本阶段不会读取 `BoundaryModel` 放行边界。

`MismatchTracker` 分别累计无正常候选、关系持续失配、持续 HOLD 和进度停滞，只在达到配置周期时输出可审计 `MismatchEvent`；阶段三不执行恢复动作。当当前事件命中 `LinkPendingCandidate`，后续正常状态或下一边界确实需要 linked，当前关系 Unknown，且当前与上一观测均可用、可靠但动作激励不足时，角色层输出 `RelationVerificationRequest`。本阶段只产生请求，不移动机械臂、不激活 Pending 模板，也不改变任务进度。

阶段三阈值集中在 `configs/closed_loop_execution.json`。正常 StackWine V4 只读回放覆盖动态路由和运行权重，确认无恢复误报、每个查询均有可执行动作，并覆盖新选流 Unknown 时沿用此前可信权重。

## 查询语义

`DynaMAC.query_state(observation, state_id, stream_weights, mode_index=...)` 是只读查询；`state_id` 严格只有 `(skill_index, local_index)` 两个进度分量，mode 只能通过独立的 `mode_index` 提供。不提供 `mode_index` 时读取 reset 已选择的该技能模态：

- `stream_weights=None` 使用冻结 baseline 的 Eq.5/Eq.6 参与掩码；
- 显式权重映射使用 `w * precision`，零权重等价于移除专家；
- 显式映射未列出的流权重为零；
- Eq.6 未选择的模态流不能被运行权重重新启用；
- 查询不会改变技能索引、时间索引、模态路径或虚拟帧状态。

查询尚未由 baseline 游标捕获的技能状态时，闭环控制器应通过 `DynaMACObservation.frames` 提供已经由其运行快照捕获的 `virtual_skill_*` 帧。
