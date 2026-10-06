# ReliVE v2 汇报 PPT v3：变更记录与证据边界

- 生成日期：2026-10-06
- 输入演示稿：`ReliVE_v2_research_progress_20261006_final_v2.pptx`
- 输入 SHA-256：`5a14ada09208d30e4e7cd4fc2c8d2bba3e26259bbb8afc1f9e252f273c155c0e`
- 输出演示稿：`ReliVE_v2_research_progress_20261006_final_v3.pptx`
- 输出 SHA-256：`8eb4c7a26cc3f0d16d7074cd31cce0de93d66e5d15be3e87b90a25982bc384dc`
- PDF SHA-256：`bef96f2e412be72d00a49bab35c6e6ade8964f7bc6979dc6c75173894176ec24`
- Contact sheet SHA-256：`9a3c84a7150153146f9523e1fd3231dcf20a5803e7d7b69600dc10c24641f0d0`

## 完成内容

保留 V2 原文件不变，并以同一深海军蓝、青绿、琥珀与珊瑚配色制作 V3。主讲部分为 13 页，之后附 7 页 appendix，共 20 页。

| 页码 | 主题 | V3 增补 / 修订 |
|---|---|---|
| 1 | 封面 | 使用项目内历史候选帧；标注其仅为方法示意。 |
| 2 | 研究动机 | 用同一 candidate 的 ORIGINAL、KEEP_TARGET、DROP_TARGET、DROP_MATCHED_CONTROL 图像说明“正确答案可来自错误证据”。 |
| 3 | 研究问题 | 明确 temporal validity、sufficiency、specificity、operator executability、calibrated admission、fail-closed。 |
| 4 | ReliVE 框架 | Capture → Verify → Refine/Adapt → Re-verify → Certificate；单列 evaluation sidecar 与 runtime 的隔离。 |
| 5 | 类型化 evidence programs | SPATIAL_RELATION、CONTACT_ACTION、POSTCONDITION_PERSISTENCE。 |
| 6 | 干预变体 | ORIGINAL、KEEP_TARGET、KEEP_MATCHED_CONTROL[k]、DROP_TARGET、DROP_MATCHED_CONTROL[k]，总数为 `3 + 2K`。 |
| 7 | Differential policy | 固定展示 `g_keep` 与 `g_drop` 的计算式、calibration gate 与 oracle 拒绝规则。 |
| 8 | Adapt | 将 temporal reacquire、spatial recompose、residual localize、stop diagnostic 解释为单轮失败路由。 |
| 9 | 贡献 | 四项协议与审计层面的贡献。 |
| 10 | 实验设计 | protocol-dev、calibration、blind test 的分离、指标、baselines 与 ablations。 |
| 11 | 当前进度 | 以阶段状态区分“已完成”“进行中”“未开始”，不把工程产物当实验结论。 |
| 12 | Pilot 结果 | 75 reviewed anchors、5 R0 candidates、2 R1 routes、0 VERIFIED；`UNCERTAIN` 明确为当前实现的 terminal abstention 标签。 |
| 13 | 路线图 | 固定写明 9 个独立视频、18 个 matched claims（9 true + 9 false）；calibration 与 blind test 的规模待预注册。 |
| A–G | 附录 | timebase、grounding/human review、可复现机制、case schema、9-case readiness、R0/R1 trace 与数据缺口。 |

## 事实来源与边界

### 本地静态核验

`relive/protocol_dev/reviews/protocol_dev_readiness_report.json` 的当前受版本控制状态为：9 cases，三类 claim 各 3；`READY=0`、`PENDING=9`、`INVALID=0`。该审计同时记录 `model_calls_made=0`、`cache_writes=0`、`certificate_writes=0`、`new_verified_count=0`。因此本 PPT 不声称 protocol-dev 已可运行，只把 9/18 作为预注册 cohort 的结构化目标。

`relive-r1-pilot-closed` tag 仍指向 `66559522c91319eb93519fb8345d046176f41fdb`。未修改此 tag、pilot closure 或 R0/R1 历史 artifacts。

历史 closure 所记录的 R0/R1 数量、replay 零新增模型调用与 0 VERIFIED，用于说明已关闭的 pilot。服务器端 per-candidate trace 未存在于本地 checkout，故未在页面中补造它们。

### 视觉来源

所有临床图像均来自项目已有输出，未生成新图片：

- `outputs/model_comparison/qwen35_9b/h4/spatial_v1_discovery50/compat_pointer_v1/visualizations/discovery/7d5049cee1dbe796.jpg`
- `outputs/model_comparison/qwen35_9b/h4/spatial_v1_discovery50/compat_pointer_v1/interventions/65ce0bf7ca24b95b12/KEEP_ROI_V1/frame_0000.png`
- `outputs/model_comparison/qwen35_9b/h4/spatial_v1_discovery50/compat_pointer_v1/interventions/a229ace3428df141c6/DROP_ROI_V1/frame_0000.png`
- `outputs/model_comparison/qwen35_9b/h4/spatial_v1_discovery50/compat_pointer_v1/interventions/d214b8feed32d30764/DROP_CONTROL_V1/frame_0000.png`

这些图像仅说明干预设计和历史 pilot 观察，不被表述为新的 differential-policy 结果或 certificate 证据。

## 仍需补充的服务器素材 / 实验数据

1. R0/R1 的 per-candidate trace、四个 variants、pixel-audit 详情及 replay 对照。
2. Stage 3G human-overlay review packet / contact sheets。
3. protocol-dev 的人工 completion、oracle annotation、portable frame pattern 与 source-record binding；当前静态审计仍为 9 PENDING。
4. 连续分数 provider、冻结 calibration artifact、certificate threshold 与 control-tier admission 数据。
5. blind-test、TimeLens/E-VQA、主要 ablations、case-level bootstrap CI、成本与稳定性数据。

## 交付验证

- PPTX：20 slides；包完整性、布局几何与字体策略检查均为 0 findings / 0 warnings。
- 额外用 `@oai/artifact-tool` 重新导入 PPTX：PASS，确认 20 slides 与输出 SHA-256 一致。
- PDF：由 LibreOffice Impress 导出；20 pages，16:9（960.009 × 540 pt）。
- PDF 页面已光栅化检查封面、框架、pilot、roadmap 与 contact sheet，无裁切或页数错误。
