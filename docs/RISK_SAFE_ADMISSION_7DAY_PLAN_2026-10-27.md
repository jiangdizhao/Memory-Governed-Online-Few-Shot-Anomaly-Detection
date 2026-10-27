# Risk-Safe Memory Admission — 七天研究与实验执行计划（v1.1）

**日期：** 2026-10-27（周二）— 2026-11-02（周一）  
**状态：** 研究协议修订完成；**实验实现与执行尚未开始，必须等用户完成本地同步并明确授权**。  
**新研究分支：** \`risk-safe-admission-sprint-20261027\`（基于 V0.2 已归档提交 \`45c4a5faaac425a1a7c91be9e538fbb268782ea3\`）。  
**原始依据：** 项目文件 \`Risk-Safe Memory Admission-7DAY-PLANS.docx\` v1.0、\`Risk-Safe Memory Admission for v22: A Deployment-Oriented Research Strategy.pdf\`，以及 \`docs/Risk-Safe_Handoff_2026-10-27.docx\`。原始 docx 不在该 Git 分支的 \`docs/\` 下；本文件是其**入库、可审查的执行计划 v1.1**，保留原研究目标并重点修订人工查询协议。旧手稿、v22、V0.2 实验、结果和历史交接均不改动。

## 0. 科学目标、范围与本次修订

**主线一：** 验证 Risk-Safe Memory Admission：在写入已确认正常 patch 之前，能否利用当时可见的历史 verified-normal / verified-defect anchors 估计反事实正常收益和缺陷风险。CDARG 是主候选；SAEG 是低成本候选；Causal HITL-style Novelty 是必须实现的强对照。与 Static-only、Naive Append、Flat Greedy、Reservoir 比较，真正容量饱和时加入 ContCore-style adaptation。

**主线二：** 独立识别 Signed Correction（C）、Guarded Sidecar（T）和 AIF-inspired Query（Q）的贡献，以及 Admission（A）与 C/T/Q 的交互。不得把组件数量本身视为贡献。

**本版重大修订（原 v1.0 §2.1/§4/§5/§6 的查询内容以本版为准）：**

1. **撤销统一固定 3% 的先验要求。** 查询预算是设计变量。预算绝对数、实际图像查询率及被查询的物理对象比例必须并列报告；任何候选预算格点只作为 D1 数据/成本预检的参考，在读取新结果之前注册首轮最终格点。
2. **修复 V0.2 数量口径错误。** V0.2 每类 525 个物理对象 × 5 相机视角 = 2,625 张在线图像，采用按图像数计算的约 3% label-blind tape，即 **79 次查询 / 79 个互异物理对象**，约为物理对象的 15.05%；并非约 16 次查询。此口径已由 GitHub 的 \`final_decision_core.query_tape\` 与逐类 \`summary.json\` 核验。未来数据分组/流若不同，不能直接复用这些分母。
3. 区分 **A. Shared Random Tape（主因果归因）**、**B. Shared Frozen-Reference Informative Query（主动查询稳健性）**、**C. Native Full Closed-Loop（实际部署机制与交互）** 三种不同监督生成协议。A/B 各方法共享实际查询事件；C 允许策略之间的查询轨迹不同，但须有同等预算上限和明确的实际用量记录。A/B/C 的结论不可互相冒充。
4. 将 Query 与 Admission 分开：**没有 Query Controller 的 A 模式不是声称样本“值得询问”，而是使用标签盲随机化消除选择混杂**；B 用独立于被比较 Admission 状态的共享查询控制器；只有 C 才对原 v22 AIF-inspired controller 的部署价值提出主张。
5. 采用查询预算—性能曲线、已确认正常/缺陷标签构成、独立物理对象/缺陷覆盖、锚点激活时间与 \`risk_unavailable\` 统计；避免单点 AUROC 或仅按名义查询率得出结论。

## 1. 研究问题及预注册假设

- **RQ1 / H1（Admission）：** 正确的 verified-normal memory write 也可能降低独立缺陷 probe 的分数；历史 verified-defect anchors 的反事实分数侵蚀是否预测实际 probe 损害，并优于 Novelty、Margin 和随机拒绝？安全收益必须在 **matched-admission / matched-FPR-gain** 下成立。
- **RQ2 / H2（Correction）：** Signed Correction 能否在 raw-memory score 已被单独观察的前提下，修复 Admission 和阈值无法替代的局部 FP/FN？
- **RQ3 / H3（Query/Sidecar）：** 在相同 *预算上限* 下 AIF-inspired Query 是否比 random 更高效地获得有用的正常和缺陷证据？Sidecar 是否在真实标签充分且激活门槛满足的窗口改善 operating FPR/FNR？未激活应记录 \`underpowered/inactive\`，不是无效。
- **RQ4 / H4（Interaction）：** A×Q、A×C、A×T 的交互是否可重复，而非从不同标签样本、阈值或额外监督获得的伪增益？

## 2. 数据、视觉配置和查询计数的不可混淆边界

### 2.1 冻结的视觉与记忆条件

以 V0.2 的冻结 DINOv2 / VisionAD-style 多层特征、4-shot 静态正常支持、归一化余弦距离、\`knn=5\`、相同 patch / image aggregation、相同候选 patch 产生逻辑和相同 Greedy 替换机制为初始实现参照。V0.2 参考设置为 fixed normal bank 25,000 features（每张 support 上限 5,000）、support augmentation=4、feature fusion=concat、identity-only inference view、动态候选数 8/16/24、\`topk_patches=32\`、image aggregation=max。完整有效配置以当前代码实际读取并写入 manifest 为准，不凭简述静默更改。

严格区分：

1. **Physical object:** 一个工业产品实例；通常 Real-IAD 有多相机观测，测试独立性和分组泄漏以物理对象 ID 为单位。
2. **Camera image / observation:** 产品的某一相机原始图像；不同相机视角是相关的、但视觉证据不等价。
3. **Support augmentation:** 已有正常 support 的旋转/变换；不增加独立人工标签数量。
4. **Pseudo multi-view inference:** query/support 的匹配变换与异常图融合；不增加独立人工标签数量。V0.2 主协议保持 identity-only，除非以后独立注册视图消融。
5. **Human query:** 一次真实标注者确认的单个**图像级** OK/NG 标签机会；每个查询记录其对应物理对象 ID、相机视角 ID、图像路径及时间。当同一物理对象的另一相机视图没有被标注时，不得自动传播图像级缺陷标签或声称已获得像素级标注。若未来研究 object-level 审核工作流，需独立定义查询价格和权限。
6. **Query target/sample:** 在线图像是在数据流中被评分的单位；查询上限可按人工图像级标签**绝对次数**设定，同时约束同一物理对象在一个流中至多一次查询。若临床/生产流程要求再次复核，应另立协议，不能默默改变单位。

记录 \`N_images\`、\`N_objects\`、\`N_queries\`、\`N_queried_unique_objects\`、\`N_normal_labels\`、\`N_defect_labels\`、\`N_anchor_normal_objects\`、\`N_anchor_defect_objects\`。报告至少三项：
- image-query fraction = \`N_queries / N_images\`；
- unique-object audit fraction = \`N_queried_unique_objects / N_objects\`；
- absolute human-verification cost = \`N_queries\`（若验证成本因相机/标签不同而异，另报成本单位）。

**V0.2 例子，仅作为历史核对：** 525 physical objects、2,625 images、79 queried image labels、79 unique queried objects = 3.01% image-query fraction 和 15.05% unique-object audit fraction。原 v1.0 把“约 3%”误算成“约 16 次”不再适用。

### 2.2 Cold、Mature 和开发/确认分组

- **Cold 主条件：** 仅四个经验证的正常原始 support 对象用于初始静态 bank；不添加隐藏的 commissioning/history 标签。
- **Mature 辅助条件：** 原 V0.2 的额外 64 个已验证正常历史对象是单独的 history-assisted 监督条件，须加总其成本并独立呈现，不可称为 \`4-shot-total\`。
- **开发集：** Real-IAD \`audiojack, bottle_cap, eraser, plastic_plug, switch, rolled_strip_base\`；smoke 固定 \`audiojack, bottle_cap\`，不根据结果挑选。
- **确认集：** 未参与调参的其他 Real-IAD 类别与 MVTec AD；不得用确认集或 evaluation-only probe 反馈选择 \`lambda/tau\`、预算格点或阈值。

严格隔离 support / mature-history / online stream / fixed A/B probes 的**物理对象**。每个 probe 的标签仅用于离线诊断；标签知情排序只可用于预声明环境压力测试，不准作为在线 controller 的信息。

### 2.3 查询预算：不预设唯一 3%，但先固定可审计的比较规则

主要横轴优先用每类**绝对查询数** \`B\`，辅助标示所占两种分母的百分比。首轮 **候选（非强制）** 预算格点为 \`B ∈ {8,16,32,64,79,96,128}\` labels/category，均需满足对象互异和可查询上限。对历史 525-object/2,625-image 流，它们只是不同的标注强度；\`79\` 是 V0.2 可追溯参考，不是优先或最优预算。

**D1 只允许根据**各类别对象数、正常/缺陷基数的离线协议预检（仅用于研究环境设计、不得喂给在线策略）、每个 B 下标签盲抽样可覆盖范围、Sidecar/Admission 的最低历史证据要求、以及计算时间/内存预算，**在看任何新算法性能前**注册首轮执行格点、seed 和运行顺序，写入 \`protocol_v1.md\` 与 hash。若预算不可行，可预先删减格点并注明原因，不能根据好坏结果挑选最优 B。

以低预算—中预算—V0.2 参考预算组成第一周的**优先级子矩阵**，剩余高预算是预注册敏感性/后续确认，不要求七天穷尽所有格点与 16 因子全交叉。报告实际用掉的查询次数；某策略未用满 B，不能谎称所有策略查询数量相等。预算可以使用 nested priority prefixes（共同固定随机优先级），便于同 seed 对照，但每个 B 下必须独立按前缀重新回放状态，不得泄漏高预算轨迹。不同预算不仅改变 query count，也可能改变 label composition；不得以同比例缩放替代真实运行。

### 2.4 三种查询策略实验：不能合并结论

**Q-A：Shared Label-Blind Random Tape — Admission 的主归因实验**
- 在任何 query 标签揭示之前，用预注册 seed 产生同一 stream/seed/order/B 的对象互异、标签盲采样优先级及一个相机图像对应的 query time。可按时间区间分层以避免全部查询集中于末尾，但分层规则必须事先固定且**不可根据真实标签分层**。
- \`Static / Append / Reservoir / Greedy / HITL-Novelty / SAEG / CDARG\` 在同一实验条件下接收**完全相同的查询时间、图像和揭示标签**。在时刻 t 的预测完成后才可以看到查询标签。
- 随机 tape **不评估某张图片是否“值得问”**，是刻意消除 Query selection confounding。只可主张在该外生标注机制下的 Admission 效果，不可主张完整主动查询的部署性能。

**Q-B：Shared Frozen-Reference Informative Query — Informative feedback 的稳健性测试**
- 增设一个单独参考查询者：固定初始 support、固定 frozen-reference scorer、共享初始 \`theta0\`，只读**自己按时间先后实际取得的已查询标签/预算 ledger**，使用确定性 score/uncertainty/novelty 或经过严格裁剪且公开具体项的 **AIF-inspired** classify-vs-query 规则。
- **禁止**参考查询者读取或受到任一 Admission 方法的 STM、LTM、Correction、动态 Sidecar 或独立未来标签影响。原 v22 依赖动态记忆的 AIF utility 项若不能从共享参考状态计算，必须明确置为未启用并记录；因此 Q-B **不是**原始完整 v22 Query Controller 的性能复现。
- 对每个 stream/seed/order/B，参考查询者先独立以严格在线方式生成一次 query tape，并只在查询后揭示标签；该 tape 随后**原封不动**向所有 Admission 方法重放。所有方法仍获得相同标签。对照 Q-A 检查 CDARG 收益是否依赖查询分布。
- 若参考查询者低于预算上限，保存未花费预算并单独报告；不能为了看齐随机方法而使用未来标签补抽有利图像。

**Q-C：Native Full Closed-Loop Query — 独立部署/交互研究**
- 恢复各方法自身的 AIF-inspired Query、各自的 normal memory / correction / Sidecar 状态；允许不同方法由于不同状态而在**不同时间查询不同图像**。
- 实施相同的标签**预算上限** \`B\`、相同流、相同最初 support、相同任务可用信息和同等物理对象复查约束；报告 actual labels spent、normal/defect split、独立锚点及激活历程。若实际标签数量不同，称为 **budget-cap-matched** 而非 **count-matched**，并在累计已用标签成本轴给出曲线。
- \`Q-off\` 是 budget-cap-matched random controller（不是零查询）；必要时增加 count-matched 诊断，但必须说明如何保证不使用未来标签。在 A/C/T/Q 因子实验中，A-off=Greedy、C-off=无 Signed Correction（仍保留独立 defect Anchor Ledger）、T-off=固定 \`theta0\`、Q-off=同预算随机。
- Q-A/B 比的是固定获取的标签条件下的 Admission；Q-C 比的是完整决策策略的后果。**没有一个模式能自动代表另外两个模式。**

### 2.5 统一在线事件顺序及因果防火墙

对每个真实流事件 t：
1. 读取当前图像，利用不可变特征与 **t 之前**的 \`(M_t,C_t,theta_t)\` 得出 pre-query score/prediction。
2. 按 Q-A/Q-B/Q-C 对应策略、当时可见信息和剩余预算决定 \`q_t\`，不得先用当前真实标签判定其价值。
3. 只有 \`q_t=1\` 才揭示该**选中图像**的真实标签；之后才能记录 verified outcome。
4. 如果 verified-normal，生成统一候选 patch，利用 **t-1 及更早** 的 Anchor Ledger 进行 shadow admission/rejection；此时固定 Sidecar/Correction pre-update 状态，拒绝写入不得改变动态 bank。
5. 当前样本标签可在本次 Admission 决策后记入 Ledger，再更新 Correction、Sidecar 和 query-price state，供未来 t+1 使用；不得使当前新写入的 correction/Sidecar 信息反向批准当前 Admission。
6. Evaluation-only probe 只能在指标阶段被读取，不能为选 query、算风险门槛、校准阈值或确定后续 budget 使用。

当前 \`knn=5\` 和多层融合，不能把单最近邻距离的理想单调性当成全评分器的定理；CDARG 必须通过与实际生产 scorer 一致的 exact shadow rescore 验证。对于 \`N_defect_anchors=0\` 必须输出 \`risk_unavailable\`，采用预先定义的保守 fallback，**绝不**将风险默认为 0。只报告对已观测 anchors 的经验风险，不宣称未知缺陷的无条件安全保证。

### 2.6 预算相关输出与优先指标

保存每个 timestep 的 \`query_policy / budget_cap / actual_queries / image_view_id / physical_object_id / query_seed / selected_before_label / verified_label_if_queried / query_reason / pre_score / pre_theta / admission_state / anchor_counts / Sidecar_activation\`。Q-B 另存共享参考状态摘要/启用的 utility 项；Q-C 另存自身 query-price/阈值轨迹、预算未用尽原因。

关键图表：\`FNR-Degradation-AUC vs. actual label count\`、\`FPR improvement vs. actual label count\`、\`AUROC / AUPRC vs. label count\`、\`risk prediction quality vs. prior independent defect anchors\`、\`query normal/defect composition vs. time\`；同预算下另报告 admission-rate-matched / FPR-gain-matched Pareto、面积下的全程风险、最坏窗口、耗时与实际字节。

缺陷锚点未出现或 Sidecar 未激活的时间段视为 **unavailable/underpowered**；不能通过更改分母、主动跳过困难类别或零填充得到表面收益。单类多个 seeds 共享 probe 时按类别聚类做不确定性估计；seed/order 不是更多独立类别。

## 3. Admission 方法与公平竞争对照（承接 v1.0，算法未实施）

接口目标：
\`propose(state, verified_normal, fixed_candidates) → evaluate(pre_state, shadow_state, past_anchors) → decision(admit/reject/replace, risk_score, normal_utility, reason) → commit_if_accepted\`。

- **B0 Static-only:** 禁止动态正常写入。
- **B1 Naive verified-normal append:** 容量未满直接追加，满时按事先明定策略处理，不能偷偷超容量。
- **B2 Flat Greedy:** 复用 v02 \`greedy_slot\`，未满追加，满时真实的 redundant-pair replacement。
- **B3 Reservoir:** 同预算的随机保留/替换。
- **B4 Causal HITL-style Novelty:** novelty 阈值只由过去可见正常证据设定；不得复用原 HITL 文献依赖 oracle error pool / review normals 的非因果设置。
- **B5 SAEG:** 基于历史 verified-defect 图像高响应 patch 的排斥与正常覆盖效用；没有像素真值时不能假定所有高响应 patch 都是缺陷。
- **B6 CDARG:** 对 verified-normal 候选 \`C_t\` 建立临时 \`M_t ⊕ C_t\`，重算历史正常/缺陷 anchors 的变化。定义正常收益 \`U_N(C_t)\`、缺陷损害 \`R_D(C_t)\`，预注册 \`U_N - lambda R_D >= tau\` 和 defect-margin-crossing 安全边界；报告连续风险值、matched-admission 结果和零缺陷锚点 fallback。
- 容量饱和后增加 **ContCore-style** 同底座适配，标明不是官方原生结果。不得把 memory admission/correction 混作同一开关。

候选 patch 第一版按每图 8/16/24 个作为一个 **event-level** 提案，patch-level gate 属于独立扩展。必须有 shadow tensor non-alias、拒绝状态字节不变、确定性重复评估、capacity-hit/replacement>0（饱和 arm）等测试。优化顺序：exact event-level shadow → 固定历史锚点缓存/近邻预筛 → exact confirm。监控 p50/p95 延迟、GPU memory、实际成本，不允许通过改换 scorer 获得虚假加速。

## 4. 七天任务与验收（以下均是**将来计划**，本次 Git 操作不执行实验）

### D1 — 10/27：协议冻结与数据预检

- 在独立研究分支冻结新协议与 V0.2 SHA，核对类/对象/视角/标签计数的真实分母（特别验证 \`525×5=2625 → 79 queries\` 的旧协议复现）；检查每图标签与物理对象标签的关系。
- **在看新实验指标前**选择第一周预算子格点、random-tape seed、Q-B 参考选择规则、预算未用尽处理、每对象最多一次查询、不同策略标签可用权限；保存 \`query_policy_manifest.json\` 和版本 hash。计算各预算的理论最大 unique-object coverage、冷启动/Sidecar 所需标签数、流长/容量可行性。
- **待用户正式授权后**才实现只读 shadow bank、独立 Anchor Ledger、统一 runner/config、特征缓存、raw/operational score logging。只做 CPU/unit + 少量 GPU smoke；不得启动大矩阵。
- 交付：\`protocol_v1.md\`、\`dataset_preflight.json\`、\`query_budget_manifest.json\`、\`cache_manifest.json\`、\`baseline_runtime.json\`、单元测试及任务队列。若可验证标签分母不足，停止而非悄悄扩样。

### D2 — 10/28：主 Q-A 随机查询对照

- 实现 Static / Append / Greedy / Reservoir / Causal HITL Novelty 的共同候选、相同 Q-A query tape、相同 frozen \`theta0\`、单次 query 后 reveal 的顺序；实现多预算重复回放与 query ledger 一致性测试。
- 计算 probe AUROC/AUPRC、FNR erosion、FPR gain、admission、每类 normal/defect 标签数、实时/峰值开销。smoke 固定 audiojack/bottle_cap × 支持 seeds/顺序配对，首选 D1 注册的低/中预算子集。
- 交付：\`baseline_comparison_dev.csv\`、\`probe_snapshots.csv\`、\`query_budget_audit.csv\`、\`hitl_causal_adaptation_note.md\`。不得把 random tape 的查询称为“被控制器认为有价值”。

### D3 — 10/29：SAEG / CDARG 与影子评分

- 完成历史 verified-defect patch anchors、SAEG exclusion、CDARG 双锚点 exact counterfactual、阈值跨越、零缺陷 fallback、commit/reject/replace。
- 使用**相同 Q-A 查询序列**进行 paired shadow replay，记录提案事件、先验正常/缺陷锚点数量、风险评分与独立 probe 的事后损害。添加 exact-vs-fast 排序一致性测试。
- 交付：算法实现/因果单元测试、候选风险记录；不得让当前或未来 sample 的反馈参与当前风险批准。

### D4 — 10/30：风险预测和预算响应曲线

- 预先在开发数据上定义危险更新及分数损害指标；比较 CDARG / SAEG / novelty / boundary / random 的 ROC-AUC、PR-AUC、Spearman 和校准，覆盖预算子格点。
- 作 \`FNR erosion — normal adaptation\` Pareto，匹配 admission rate 和 normal FPR gain。**新增 Q-B 参考查询者的初步 smoke 与冻结 tape 复用审计**；按预算和锚点数分层展示 Q-A vs Q-B 的 CDARG 机制收益。
- 交付：\`risk_prediction_report\`、\`anchor_shuffle_ablation\`、\`pareto_dev_v1\`、\`query_A_vs_B_smoke\`、失败案例。缺陷标签不足时输出 \`insufficient_evidence\`。

### D5 — 10/31：扩展开发对照与真正容量饱和

- 优先扩展 Q-A 主归因矩阵到六开发类、3 supports seeds、Normal-first / Defect-first / Random / Bursty，原目标 72 个 class×seed×order 组合；条件数仍须乘方法和**实际选定**预算数。算力不足只运行预注册优先级的成对子矩阵，不挑有利类别/预算。
- 将 Q-B 扩展到能够配对比较的子矩阵；记录每个策略的 normal/defect anchor acquisition 而不仅是 Query rate。做 ContCore-style、真实 Greedy capacity hit/replacement、同实际内存上限与延迟审计。
- 交付：\`expanded_dev_results\`、\`shared_query_comparison\`、\`capacity_stress_report\`、\`contcore_adaptation_note\`。

### D6 — 11/01：真正 Q-C 闭环与 A/C/T/Q 因子实验

- 将原有 Signed Correction / Guarded Sidecar / Simple AIF Query 接回完整闭环。严格检查 query-before-label、单对象查询约束、Anchor Ledger 与 Sidecar 先后关系，Q-A/B 的归因代码不受污染。
- 在用户允许的 GPU 成本内，以 2 开发类 × 2 seeds × 2 顺序的小矩阵跑 \`2^4=16\` 因子**初步**实验；A-off=Greedy、C-off=关闭纠错但保留独立 defect anchor ledger、T-off=固定 \`theta0\`、Q-off=同预算上限 random。重点 A×Q、A×C、A×T，另报 full-minus-one。
- 必须明确：**同 budget cap 不等于同实际查询数**。比较实际标签成本轨迹、锚点组成和 controller activation。Sidecar inactive 的条件只能标注 \`underpowered\`，预注册较长流/较高预算敏感性实验，不准事后调低激活门槛。
- 交付：\`factorial_results\`、\`module_activation_audit\`、\`query_composition.csv\`、\`raw_vs_corrected_scores\`、交互图。

### D7 — 11/02：审计与 GO / CONDITIONAL / NO-GO

- 验证 seeds、物理对象隔离、每视角标签访问、Q-A/Q-B tape 恒等、Q-C 合法非恒等 query traces、超预算或未用满记录、代码 SHA、缓存/探针隔离、完成条件和失败重跑的不可变性。
- 输出带预算横轴的 risk-prediction、FNR-erosion/adaptation Pareto、normal/defect anchor acquisition、Q-A/Q-B/Q-C 结果以及四模块因子图；按类别聚类计算不确定性。记录真实容量饱和、时序、延迟和 GPU 峰值。
- 形成 RQ1–RQ4 证据矩阵：**GO** 要求风险预测增量及 matched-admission/matched-FPR 安全效益不能归因于更少写入；**CONDITIONAL** 如 SAEG 解释大部分 CDARG 收益则优先简化；**NO-GO** 如历史 anchors 不预测真实 probe 损害/有效风险事件几乎不存在/外部 novelty 等效则停止复杂化。
- 交付：\`week1_decision.md\`、图表、\`complete_manifest.json\`、\`limitations.md\`、\`next_week_confirmatory_plan.md\`。预注册但尚未运行的格点必须标注 pending，**不得补造性能数值**。

## 5. 指标、统计与闭环控制策略

- **固定阈值主指标**：在相同固定 \`theta0\` 下，\`FNR-Degradation-AUC = mean_k max(0, FNR_k-FNR_0)\`（预先固定的 probe 检查点），\`FPR-Improvement = FPR_0-FPR_K\`，AUROC/AUPRC、分数侵蚀 \`mean_d [s_before(d)-s_after(d)]_+\`。此 \`theta0\` 是 LOSO 诊断阈值，不宣称 FPR 保证。
- **部署指标**：保留自适应 \`theta_t\` 下的 operational FPR/FNR、纠错前后分数，并报告 query count 和 controller activation。不要用动态 threshold 掩盖不安全 raw-memory 写入。
- **预算指标**：所有方法绘制实际累计人工查询数曲线，且按独立 defect-anchor / normal-anchor 个数条件化风险预测表现。报告第一个 verified defect anchor 的到达时间、risk gate 首次可用时间、缺陷区域 coverage、多视角相关与 \`risk_unavailable\` 占比。仅有 image-level defect label 不表示标注了具体缺陷 patch。
- **公平归因**：Q-A 是 primary memory admission; Q-B 是 independent shared-policy robustness; Q-C 是 closed-loop strategy effect。需要在同一 query policy/B/stream/seed 内做配对；各阶段不同 query labels 不能进行伪因果归因。Q-C 的 query-policy 对照需报告各自计数、组成及累计成本曲线。
- **统计**：产品类别是主要独立抽样单位；seed/order 属于类别内重复；报告逐类差异、macro 和按类别 bootstrap 区间、最差类和异常案例。测试/探针不参与超参选择。 
- **组件判定**：A=CDARG vs Greedy；C=Signed Correction on/off；T=Sidecar on vs fixed theta0；Q=native AIF-inspired vs cap-matched random。预定义二阶交互 \`I_AQ=[J(1,1)-J(0,1)]-[J(1,0)-J(0,0)]\` 并保留 FNR/FPR/成本的分别呈现。无可识别性不能解释为“无效”；无独立收益、无交互收益而成本显著的组件应考虑删除。

## 6. 工程产物、文献边界和严格暂停点

**规划中的代码结构（尚未创建）：** \`experiments/risk_safe/{config.py,protocol.py,admission.py,anchors.py,counterfactual.py,baselines.py,metrics.py,runner.py,queue.py,audit.py}\` 与 \`tests/risk_safe/\` 对应因果 label access、shadow immutability、replacement saturation、score consistency、object isolation、query policy parity、Sidecar activation 测试。

**规划中的每次运行记录（尚未生成）：** \`outputs/risk_safe_sprint_20261027/<method>/<class>/<seed>/<order>/<query_policy>/<budget>/<run_id>/\`，保留 \`config.json\`、\`source_hashes.json\`、\`input_manifest.json\`、\`object_partitions.json\`、\`query_ledger.csv\`、\`anchor_ledger.csv\`、\`candidate_decisions.csv\`、\`probe_snapshots.csv\`、\`memory_audit.csv\`、\`timings.csv\`、\`summary.json\`、\`status.json\` 与 traceback。断点恢复须验证源码/输入 SHA；失败目录不自动覆盖；GPU 独占、按优先级排队，不根据结果改预算/标签抽样。

**文献边界：**
- VisionAD 是 frozen/static FSAD front-end：support augmentation/pseudo multi-view **不产生人工 query**。
- HITL Memory Correction（https://arxiv.org/abs/2608.17775）的原生 review-pool 纠错/阈值涉及提前知道错误的机制；主比较必须用 **causal** HITL-style Novelty，同一查询权限/特征/容量/候选，而非冒称原生论文完全复现。
- ContCore（https://github.com/jungyg/ContCore）与 FOADS/K-NG（https://github.com/Whishing/K-NG）是不同监督协议的竞争研究，官方 published native protocol 指标只能单列；同底座适配必须明示非官方复现。
- 其他预算相关外部文献说明人工查询预算并无统一 3% 定数；本文采用可审计的预算—风险曲线而不是直接沿用一篇论文的标注比例。

**本次文档提交即停止。** 用户完成本地 \`git fetch/switch/pull\` 并确认无误之后，才应创建实验代码、生成新特征缓存或运行真实 GPU 条件。不得因为七天计划写有 D1 日期就自动视为用户已授权执行。

## 7. 修订一致性与 D1 前审查清单

- [ ] GitHub 新分支 HEAD 与本地一致；v22/V0.2/旧论文/历史结果未修改。
- [ ] Real-IAD 525 physical objects 与 2,625 images（如沿用原 V0.2）被分别核实；历史 79 次查询不可误写成 16。
- [ ] 每次人工确认的图像级标签、相机视角和物理对象唯一性都有明确计费口径。
- [ ] Q-A random、Q-B shared informative、Q-C native full closed-loop 明确区分，budget cap 与实际 queries 不混写。
- [ ] 在读取任何新结果前冻结首轮预算格点与预算解释、随机种子、查询权重/规则、合法标签获取时序。
- [ ] 严格隔离 future labels 与 evaluation probes；\`N_defect_anchors=0\` 是 risk-unavailable。
- [ ] 主要算法对照接受相同原始图像、scorer、候选、阈值和持久内存预算；Q-A/B 得到完全相同的反馈事件。
- [ ] GO/CONDITIONAL/NO-GO 结果必须由真实实验给出，第一周未完成部分标注 pending。

**与原 v1.0 保持一致的其他原则继续有效：** 对照的容量、证据和 query 可见性匹配；真实 Greedy replacement；raw vs correction-adjusted score 双轨；阈值变化不能倒灌当前写入；开发与确认数据隔离；六类原始开发集；非 IID 只作声明的 synthetic stress；不使用未来标签或 oracle masks 进行在线决策；不可删改历史 negative results。