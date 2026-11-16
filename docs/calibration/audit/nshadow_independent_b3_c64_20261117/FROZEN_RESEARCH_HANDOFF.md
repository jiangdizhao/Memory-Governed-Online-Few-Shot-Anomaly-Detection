# 冻结研究交接 — N-shadow + Independent B3/C64（2026-11-17）

> **状态：FROZEN / EXPERIMENT COMPLETE / POST-HOC EXPLORATORY。** 本文是对已完成实验的证据冻结与后续对话交接，不代表新的独立复现实验，也不授权重新调参或开放新数据。原始 Stage A/B 工件不应被覆盖。

## 0. 一页接入：结论与证据身份

- 项目：Memory-Governed Online Few-Shot Anomaly Detection；分支 `calibration-stagea-consolidated-20261114`。
- **本次用户推送的源与结果归档提交**：`22919c588dad27e57290a16dea39322d59b96eff`。本文件是在该提交之后追加的独立冻结记录；不要把冻结文件的新提交误当成当时运行所用源码版本。
- **范围**：Real-IAD 30 类 × P0/P1/P2 = 90 条；VisA 12 类 × P0 = 12 条；共 **102/102 SEALED_COMPLETE，345,917 个图像事件**（Real-IAD 343,755；VisA 2,162）。
- **主要结果**：Real-IAD `BER(N-shadow+B3) − BER(F-B) = +0.0119410942`，类别配对 Bootstrap 95% CI `[+0.0077525630,+0.0163641482]`；VisA 相同差值 `+0.0070874587`，CI `[-0.0045709571,+0.0196054043]`。
- **最谨慎的结论**：在已实现的 F-B 查询与 Greedy 正常记忆轨迹上，允许缺陷侧 C1 反馈的 F-B 在 Real-IAD 取得更低 BER，主要表现为较低 FNR；VisA 平均同方向，但 12 类区间跨零。此补充对比不能把效果唯一归因于缺陷锚点本身，也不能证明复杂 Gaussian C1 优于简单 Signed kNN。
- **反证/不利结果保留**：Real-IAD 有 3/30 类 N-shadow+B3 的 BER 更低（`pcb`、`terminalblock`、`zipper`）；VisA 为 F-B 优势 5 类、N-shadow 优势 5 类、持平 2 类（`capsules`、`pipe_fryum`）。
- **不重复执行**：这 102 条补充轨迹与其 Stage B 已完成；后续工作先读本文件、原始实验合同与汇总，不应因为切换对话而重新启动 GPU。

## 1. 被回答的科学问题及 estimand

原 Gate 0 的 `N-shadow` 是 **F-B 共用 Greedy 正常参考 + 独立 Normal-only C1 Correction**；不是 Greedy-only、不是无 Correction，也不运行自己的 Query Policy。它继承 F-B 已执行的 Query，只把已验证正常样本写入自己的 C1；已验证缺陷被其 C1 忽略。原主实验只为 N-shadow 注册 AUROC/AP，未独立维护 B3，不能直接生成合法 BER/FPR/FNR。

补充实验为 N-shadow 增加**独立**、随其真实评分状态更新的 B3/C64，从而问：在相同 F-B 购得反馈与正常记忆条件下，包含缺陷侧反馈的完整 F-B 是否在操作点上优于独立校准的 Normal-only C1？

定义 `d = Metric(N-shadow+B3) - Metric(F-B)`。主要指标为：

```text
H_D_BER = BER(N-shadow+B3) - BER(F-B)
H_D_FNR = FNR(N-shadow+B3) - FNR(F-B)
H_D_FPR = FPR(N-shadow+B3) - FPR(F-B)
BER = (FPR + FNR) / 2; positive H_D_BER favors F-B.
```

`F-M` 仅供描述性参考，因为它使用自己的 Query/反馈/状态历史，不构成同轨迹因果对照。本实验是**看到原 H_D（AUROC）结果后提出的 post-hoc exploratory** 补充分析，绝不能倒写成 Gate 0 预注册 H_D 的 BER 部分。

## 2. 冻结方法：监督权限与在线因果时序

- **模型与支持信息**：沿用历史冻结 DINOv2/VisionAD 原生评分路径，四张部署 TRAIN-normal 支持图像及额外、互斥的 64 张受保护 TRAIN-normal C64 校准图像；此信息制度是 `4 Support + 64 commissioning normals`，**不是 strict-four-shot-total**。
- **N-shadow 状态**：`S_N,t = (F-B Greedy normal memory, independent normal-only C1)`；禁止 F-B 的 defect-only C1 状态流入 shadow。任何写入、反馈只影响以后的评分/阈值。
- **独立 B3**：复用原 `StateConsistentQuantile` 及 R7 `_initialize_b3 / _refresh_b3`；对本类本条件的 64 张 C64 在当前 shadow 完整状态下计算原生分数；`q=0.90`，`method='higher'`，正常锚点最低数 16，单次评分状态转移的阈值移动上限 `0.01`。两次真实状态改变可在一个反馈边界产生两次刷新；仅 F-B defect-side 更新而 shadow 状态未变化，不触发 shadow 刷新。
- **严格顺序**：每个 Real-IAD 五视角 Source Group（或 VisA 既定 atomic unit）在反馈之前统一使用旧状态与旧阈值评分、预测、写出并 fsync；仅此后读取 F-B 当时已经购买的合法标签；再更新 shadow 正常侧 C1、共享 F-B Greedy，必要时重算 C64 与 B3，供下一个原子单元使用。
- **实现效率**：原非 Query 事件复用已封存的 N-shadow 图像级分数；只有 F-B 已购 Query 事件为原生评分 parity 重编码，再按原合法反馈重建可变状态。状态 digest/version 每事件与旧封存结果比较；不重跑不相关的 F-M/K-B/K-M 完整评分流。
- **标签权限**：Stage A 只读取过去已付费购买的 F-B 标签；未购买 TEST outcome、Pixel Mask、未来真值不进入在线计算。102 条 Stage A 均通过后 Stage B 才展开官方图像级真值评价。无新增 Query、bootstrap 强制 Query、末尾 top-up 或 BEA 参数修改。
- **统计单元**：Real-IAD 的 P0/P1/P2 先在类别内平均，再对 30 类等权；VisA 12 类 P0 单独等权。CI 为 10,000 次类别层配对 percentile Bootstrap；Real-IAD 的 90 条轨迹**不是 90 个独立类别**，两个数据集不可合并推断。

## 3. Real-IAD 正式补充结果（30 类等权，三条件先类内平均）

| 方法 | AUROC ↑ | AP ↑ | BER ↓ | FPR ↓ | FNR ↓ | BA ↑ |
|---|---:|---:|---:|---:|---:|---:|
| F-B | 0.893481 | 0.884313 | 0.177941 | 0.134370 | 0.221512 | 0.822059 |
| N-shadow+B3 | 0.868060 | 0.849631 | 0.189882 | 0.132895 | 0.246870 | 0.810118 |
| F-M（描述性） | 0.881363 | 0.867313 | 0.187630 | 0.132818 | 0.242442 | 0.812370 |

| N-shadow+B3 减 F-B | 均值 | 配对类别 Bootstrap 95% CI |
|---|---:|---|
| BER | **+0.011941** | **[+0.007753,+0.016364]** |
| FNR | +0.025358 | [+0.017076,+0.034632] |
| FPR | -0.001476 | [-0.007463,+0.004107] |

- F-B 在 **27/30 类** BER 更低；N-shadow+B3 在 `pcb`、`terminalblock`、`zipper` 三类更低。不能隐藏这些类别级负结果。
- Real-IAD BER 改善约 **1.1941 个百分点**；F-B 的 FNR 比 shadow 低约 2.5358 pp，同时 FPR 比 shadow 略高约 0.1476 pp，后者 CI 跨零。不能将本结果误报为同时严格改善两类错误。
- 缺陷侧增量的操作价值是**在本次已实现反馈、状态及 B3 路径上的整体差异**；其效应可混合 C1 正负锚点占用/替换和校准变化，不能由此单独证明某个 Gaussian kernel 的结构必要性。

## 4. VisA 补充结果（12 类 P0）

| 方法 | AUROC ↑ | AP ↑ | BER ↓ | FPR ↓ | FNR ↓ | BA ↑ |
|---|---:|---:|---:|---:|---:|---:|
| F-B | 0.946476 | 0.948709 | 0.125069 | 0.089304 | 0.160833 | 0.874931 |
| N-shadow+B3 | 0.938917 | 0.939579 | 0.132156 | 0.095146 | 0.169167 | 0.867844 |
| F-M（描述性） | 0.935646 | 0.939517 | 0.138823 | 0.091812 | 0.185833 | 0.861177 |

| N-shadow+B3 减 F-B | 均值 | 配对类别 Bootstrap 95% CI |
|---|---:|---|
| BER | **+0.007087** | **[-0.004571,+0.019605]** |
| FNR | +0.008333 | [-0.006667,+0.025000] |
| FPR | +0.005842 | [-0.007475,+0.018333] |

- F-B BER 更低 5/12 类；N-shadow+B3 更低 5/12 类：`cashew`、`fryum`、`macaroni2`、`pcb3`、`pcb4`；`capsules`、`pipe_fryum` 持平。
- 虽然平均方向偏向 F-B，**BER、FNR、FPR 的 95% CI 均跨零**，不能宣称 VisA 已确认操作收益。
- VisA 是短流、10% Query 协议、只有 P0；其物理样本独立性尚未充分建立，不能与 Real-IAD 3% 协议合并成一个总体。

## 5. 工程完成情况与证据可靠性边界

据本次 Codex 执行记录和归档报告：

- **Stage A：102/102 SEALED_COMPLETE；Stage B：102/102 在全部 Stage A seal 后评价**。
- **重放检查**：102/102 replay-parity receipts PASS；覆盖 **345,917** 个事件状态 digest/version 对照及 **10,485** 次 Native Query-event Score 校验（绝对容差 `1e-5`）；原 N-shadow AUROC/AP、F-B/F-M 指标与 Query 账本保持一致；未读取 Mask。
- **运行耗时**：Stage A `10,087.2 s`；端到端（含 Stage B）`10,828.3 s`。
- **资源**：最大采样进程 RSS `7,160,397,824 B`；PyTorch GPU allocated/reserved 最高记录 `1,573,807,104 / 14,283,702,272 B`；本地输出树 `273,829,856 B`（约 274 MB）。这些数值为运行记录，不等同于对所有瞬时 GPU 进程/设备内存的全面测量。
- **测试**：Codex 报告 10/10 focused CPU tests PASS，Python 编译及 `git diff --check` PASS；三个历史 incomplete attempts 原样保留。
- **验证层级**：本冻结操作直接读取了已提交的科学报告、代码/合同、类别聚合 JSON 和 CSV，核实其汇总与方向一致；**没有从当前对话容器重新运行本地 GPU 或重新逐事件验算原始私有 Journal**。因此本记录应被引用为「已完成且归档、经 Codex parity 验证并经文件汇总复核」，而非一次额外独立的端到端复现。

## 6. 永久来源索引（GitHub 已归档；本地保留完整事件产物）

**已推送的九份源文件与结果**：

- [最终补充报告](FINAL_SUPPLEMENTARY_RESEARCH_REPORT.md)
- [Real-IAD 30 类对比 JSON](realiad_30_category_contrasts.json)；[Real-IAD 90 条明细 CSV](realiad_90_metrics.csv)
- [VisA 12 类对比 JSON](visa_12_category_contrasts.json)；[VisA 12 条明细 CSV](visa_12_metrics.csv)
- [资源与保真报告](RESOURCE_AND_FIDELITY_REPORT.md)
- [冻结方法合同](../../nshadow_independent_b3_c64/SUPPLEMENTARY_METHOD_CONTRACT.md)
- [补充运行源码](../../../../experiments/calibration/nshadow_independent_b3_c64.py)
- [补充测试](../../../../tests/calibration/test_nshadow_independent_b3_c64.py)

归档 Git 提交：[`22919c588dad27e57290a16dea39322d59b96eff`](https://github.com/jiangdizhao/Memory-Governed-Online-Few-Shot-Anomaly-Detection/commit/22919c588dad27e57290a16dea39322d59b96eff)。

**完整本地输出（GitHub 未包含 event-level 原始工件）**：

```text
E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection\outputs\calibration\nshadow_independent_b3_c64_20261117_v1_attempt_05\
```

该本地目录含 Stage A seal、shadow prediction journals、C64/B3 状态转移、Replay Parity receipts、Stage B 结果、进度/资源报告。未来独立逐事件复核必须使用它和历史私有反馈 / dataset 工件；不要把 GitHub 汇总文件误认为完整原始实验可复现包。

历史 Real-IAD 来源为 `realiad-final-20261111-r1-40d61c3`（36）和 `realiad-final-20261113-remaining54-a3688e87ac16`（54）；VisA 由原已封存 12 类 P0 轨迹恢复。源提交和运行的 `runner SHA-256` 如需精确追溯，以本地原始 run receipt 为准，不可将 Git 的 blob SHA 与文件 SHA-256 混为一谈。

## 7. 稿件主张、负结果及下一步决策

**目前可审慎使用的主张**：

> Under matched verified-feedback opportunities and a shared normal-reference trajectory, allowing defect-side signed correction yields a lower realized prequential balanced error than independently calibrated normal-only C1 correction on Real-IAD. The mean difference is 0.01194 (paired 30-category bootstrap 95% CI 0.00775 to 0.01636), with most of the improvement appearing in FNR. The corresponding VisA result is directionally positive but inconclusive.

**不能推出**：严格四标签学习、Gaussian C1 必须优于 Signed kNN、未查询样本的反事实价值、任意流分布下的 BEA 最优性/风险安全保证、正式 Bayesian/Active-Inference 最优性、外部所有工业场景的独立泛化。原 90 轨迹中 BEA 前期消耗查询预算和后段几乎没有 Query 的风险仍然存在；本补充实验没有更改或解决该问题。

**研究决策**：本次 N-shadow+B3 补充任务已经结束、冻结并归档；不需要为同一假设再次运行 102 条轨迹、不恢复 initial forced query 或末段 top-up、不进行看到 Stage B 后的阈值/纠错参数搜索。未来优先整合拒稿恢复主论文与最新系统基线证据，承认 Signed kNN 强简单对照，开展确有科研必要且明确预注册的独立确认，而非继续拆分多个微型消融。

**未来对话启动建议**：首先阅读本文件 → 最终补充报告 → 方法合同 → 两个 `category_contrasts.json` → 必要时查看 CSV 与源码；引用数据必须区分本次后验补充和早先 Gate 0 的预注册 H_D（AUROC）及 H_Q（BEA）结果。
