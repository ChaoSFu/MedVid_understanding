# ReliVE Stage 1 RAG：冻结、检索与 22 条配对对照

本实验只生成“需要观察什么”的 evidence requirements。输入是问题、任务类型和独立医学知识，不读取视频、不生成视觉事实，不运行 Stage 2、KEEP/DROP/CONTROL、calibration 或 certificate。

## 实验设计与边界

同一批 11 类任务，每类按源文件顺序选前 2 条，共 22 条。每条执行两组：

| 组 | 模型输入 | 输出 |
|---|---|---|
| baseline | question + qa_type + 空知识数组 | plan + 空 citations |
| rag | 相同 question、qa_type、模型配置和指令 + 检索知识 | 相同 plan schema + 知识引用 sidecar |

这是本实验的新配对 baseline。两组都使用统一的引用 envelope，不能将它当作与旧 Stage 1 输出逐字一致的重跑。旧 Stage 1、Stage 2 V3/V4 的冻结输入及结果不会被替换。

本地模型配置必须为固定、非采样生成。若使用已有 OpenAI-compatible adapter，必须显式配置 `temperature: 0`，不能继承服务默认值；API 的固定 temperature 仍不保证服务跨版本逐字复现，因此 exact replay 以保存的 raw cache 为准。

每组 plan 保留 v1.1 的三个字段：`planning_status`、`planning_reason`、`requirements`；每条 requirement 保留五个字段：`requirement_id`、`description`、`type`、`evidence_type`、`importance`。医学知识引用放在 `knowledge_citations`，并导出独立 `requirement_provenance.jsonl`。引用只能指向本次实际提供的卡片及已有 requirement。

知识库 1,121 条卡片仍是 `pending`。`--allow-pending-knowledge` 表示允许研究用途，不能解释为医学审核通过。冻结时校验原始主文件、来源及同目录 build summary 记录的所有 artifact hash，不读取 `excluded_knowledge.jsonl` 中的候选来扩充卡片。可选重新打开 PDF 检查原文引文，但这仍不能替代医学语义审核。

检索使用 SQLite FTS5/BM25，不调用 embedding 或 LLM。固定 top-4、每来源最多 2 条、相同分数按 chunk_id 排序、最多 6,000 UTF-8 **字节**的知识上下文。这是字节预算，不是声称精确的模型 token 预算。按完整字段装入，放不下时省略字段或整卡，不截断医学句子。索引与排序绑定 SQLite 版本和产物 hash；跨平台重建应作为新索引冻结。

仅允许 question + qa_type 进入规划器；GT answer、bbox、span、评分、媒体路径和原始 ID 都不进入 prompt。原始 JSON 必须反序列化以做白名单投影，数据文件字节用于身份哈希；不将禁止字段的值复制到运行时。重复原始 ID 通过 `dataset_sha256 + source_record_index` 区分。

Region Caption 的 box 指代缺少公开框时记录 `UNRESOLVED_TARGET_REFERENCE`，不使用 RC_info 猜目标、不让知识先验补出目标。本轮仍允许 target-independent Stage 1 规划，检索知识为空。这不是 Stage 2 目标已解决的声明。没有匹配卡片时也退回空知识上下文。

## 安装与本地无模型测试

在仓库根目录运行。Python >= 3.10，SQLite 必须支持 FTS5。继续使用原 Stage 1 已核验的 GPU 环境；不要为了本实验临时升级 transformers、torch 或修改 chat template。

```bash
cd /path/to/MedVid_understanding
python -m pip install -r RAG/requirements.txt -e ./relive
export PYTHONPATH="$PWD/relive/src${PYTHONPATH:+:$PYTHONPATH}"

python -m unittest discover -s RAG/stage1_rag/tests -v
python -m unittest discover -s RAG/tests

MOCK_OUT="$PWD/RAG/output/stage1_rag_mock_$(date +%Y%m%d_%H%M%S)"
python -m RAG.stage1_rag --mode mock-smoke --output-dir "$MOCK_OUT"
```

mock smoke 自动执行冻结、索引、22 对模拟请求、只读 replay 和 HTML 对照。模拟知识和结果始终带 `synthetic_test=true`；不构成医学实验，禁止用于真实模型输入。

## 服务器：恢复变量

下面示例路径来自现有服务器，须按实际仓库位置设置。知识 JSON **及其同目录原始 build artifacts** 不在 Git 仓库中，先保留或同步服务器完整构建目录。不要只移动 JSON 后混用另一轮 summary。

```bash
conda activate MedVidU-cu124
cd /home/huihui/codes/MedVid_understanding_rag
export PYTHONPATH="$PWD/relive/src${PYTHONPATH:+:$PYTHONPATH}"

# 使用已经在原 Stage 1 验证过的模型配置；若原实验是 27B，必须填同一份 27B 配置。
CONFIG="$PWD/relive/configs/qwen35_9b_claim_bank_v1.yaml"
DATASET=/home/huihui/codes/MedVid_understanding/data_json/init_datas/medvidu_eccv2026_trainval.json
KB_JSON="$PWD/RAG/output/server_full_text_v021_20261124_153002/full/relive_surgical_visual_knowledge_v0.1.json"
EXP="$PWD/RAG/output/stage1_rag_smoke_$(date +%Y%m%d_%H%M%S)"

test -f "$CONFIG"
test -f "$DATASET"
test -f "$KB_JSON"
mkdir -p "$EXP"
# 保存非秘密的实验路径；重开终端后 source 此文件恢复。
declare -p CONFIG DATASET KB_JSON EXP > "$EXP/paths.sh"
printf 'Experiment directory: %s\n' "$EXP"
```

`test -f` 成功时不输出文字；出错时不要继续。以下各命令使用新的独立子目录，已存在则拒绝覆盖。不要把输出目录设为历史 artifact、知识构建目录或缓存目录。

## 1. 冻结知识库

```bash
python -m RAG.stage1_rag --mode freeze \
  --knowledge-json "$KB_JSON" \
  --expected-sha256 1d71080a06ca8d6017ee700272411f73305ecc4ee70da23c959160fcb3a74594 \
  --expected-count 1121 --allow-pending-knowledge \
  --output-dir "$EXP/corpus"
```

如需重新逐份 PDF 校验原文引文，可在上述命令增加 `--references-root "$PWD/RAG/references" --parse-cache "$EXP/pdf_parse_cache"`。该步骤没有模型调用，会读取 PDF 并写独立解析缓存。未指定时 manifest 明确记录未重新执行 PDF 原文校验，不冒称医学审核已通过。

## 2. 构建索引与 GT-free 问题清单

```bash
python -m RAG.stage1_rag --mode index \
  --corpus-dir "$EXP/corpus" --output-dir "$EXP/index"

python -m RAG.stage1_rag --mode prepare \
  --dataset-json "$DATASET" --expected-count 6270 \
  --output-dir "$EXP/questions"

python -m RAG.stage1_rag --mode preflight \
  --runtime-manifest "$EXP/questions/question_runtime_manifest.jsonl" \
  --index-dir "$EXP/index" --config "$CONFIG" --per-qa-type 2 \
  --output-dir "$EXP/preflight22"

cat "$EXP/preflight22/manifest.json"
cat "$EXP/preflight22/gt_isolation_audit.json"
```

以上均不加载模型。检查 sample_count=22、arms=[baseline,rag]、合计 44 个 plan。检索卡片和排名在 `preflight22/retrieval.jsonl`，选样在 `selected_questions.jsonl`。源数据内容变化会改变 dataset hash 与 sample key，不可混用旧结果。

`preflight22/retrieval_review_queue.jsonl` 单独列出检索适用性审查，默认 PENDING。BM25 的命中不等于医学语义相关；同名词可能对应不同器械、解剖或操作场景。请检查实体与手术上下文，尤其是仅一个词重合的卡片。人工填写时复制到冻结目录外；不能为了让 22 条看起来更好而直接修改本次冻结 retrieval。若需改检索策略，应建立新版本和新 preflight。

默认预算：44 次初始生成，最多每个 plan 一次格式修复；无技术重试时最多 88 次调用。有技术重试时硬预算为 `22 × 2 × 2 × (max_retries+1)`，在 manifest 中冻结。有效 abstention 不做语义重试。格式修复不得修改已经合法的 requirements 列表。

## 3. 后台执行真实 22 条配对 smoke

不使用 RAG 构建的文本 API 模型偷偷替换原 planner。两组由同一进程加载一份冻结配置，交替执行 baseline/rag。

```bash
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  python -m RAG.stage1_rag --mode run \
  --preflight-dir "$EXP/preflight22" --index-dir "$EXP/index" \
  --cache-dir "$EXP/inference_cache" --output-dir "$EXP/run22" \
  > "$EXP/run22.log" 2>&1 &
echo $! > "$EXP/run22.pid"

tail -f "$EXP/run22.log"
```

进度以 `run22/checkpoints/*.json` 为准，完成数分母是 **44**。中断后不要改代码、配置、索引或 preflight；用相同参数加 `--resume` 继续。不加 `--resume` 会拒绝已有目录。已完成目录加 resume 仅验证并返回已有 summary，不加载模型。

```bash
python - "$EXP/run22/checkpoints" <<'PY'
from pathlib import Path
import sys
print('completed plans:', len(list(Path(sys.argv[1]).glob('*.json'))), '/ 44')
PY

# 只在上一个进程已退出时执行恢复，不能同时启动两个任务。
CUDA_VISIBLE_DEVICES=0 python -m RAG.stage1_rag --mode run \
  --preflight-dir "$EXP/preflight22" --index-dir "$EXP/index" \
  --cache-dir "$EXP/inference_cache" --output-dir "$EXP/run22" --resume
```

## 4. 零调用、零缓存写入 replay

```bash
python -m RAG.stage1_rag --mode replay \
  --preflight-dir "$EXP/preflight22" --index-dir "$EXP/index" \
  --cache-dir "$EXP/inference_cache" --run-dir "$EXP/run22" \
  --output-dir "$EXP/replay22"

cat "$EXP/run22/generation_summary.json"
cat "$EXP/replay22/generation_summary.json"
cat "$EXP/replay22/cache_replay_audit.json"
```

replay 不加载模型、不开网络、不创建缓存锁或补写缓存；缺失 raw cache 直接失败。它重新解析原始响应，要求 plans、requirements、provenance 三个 JSONL 的 SHA-256 与 fresh 完全一致。usage 和 execution trace 中的 cache_hit/latency 可以不同。replay source 及其缓存需保留原路径，不支持任意迁移 raw_response_ref 后静默修复。

## 5. 静态验收与人工对照报告

```bash
python -m RAG.stage1_rag --mode validate \
  --preflight-dir "$EXP/preflight22" --index-dir "$EXP/index" \
  --run-dir "$EXP/run22" --output-dir "$EXP/validation22"

python -m RAG.stage1_rag --mode compare \
  --preflight-dir "$EXP/preflight22" --index-dir "$EXP/index" \
  --run-dir "$EXP/run22" --output-dir "$EXP/comparison22"
```

重点查看：

- `run22/plans.jsonl`：两组完整 plan，包括失败记录；
- `run22/requirements.jsonl`：扁平 requirements；
- `run22/requirement_provenance.jsonl`：模型声明的知识引用和 PDF 页码，不是自动证明语义相符；
- `run22/execution_traces.jsonl`：请求、原始响应引用、每次格式/技术失败；
- `comparison22/comparison.html`：22 条 question、并排 requirements、检索卡片、来源文档；
- `comparison22/human_review_queue.jsonl`：两组相关性、覆盖、动作/临床语义、事件假设、类型、冗余、知识适用性审查。

对照目录也有不可变 manifest。人工填写前先复制队列到目录外，例如 `cp -n "$EXP/comparison22/human_review_queue.jsonl" "$EXP/human_review_answers.jsonl"`，填写副本，不改写原报告。

`status=COMPLETE` 只表示本轮写出了结果，**不是通过验收**。`engineering_gate=PASS` 要求全部 plan 合同有效，无 MODEL_OR_SCHEMA_FAILURE；有失败时保留全部记录并返回非零退出码。有效的 UNABLE_TO_PLAN 仍应人工检查理由。`validate.status=PASS` 表示数据完整性，不意味着医学语义正确。

完成 smoke 后先人工对照，不自动启动全量或覆盖旧 Stage 1。只有人工审核才能评价知识是否提高了规划质量，不能以 requirement 数量、引用数量或输出格式合法替代质量结论。新要求若需进入后续 Stage 2，应另建实验和输入绑定。

## 当前限制

- V1 为英文词法检索，不包含语义 embedding、reranker 或自动 query expansion；医学近义词和长问题可能召回不足。
- 知识仍待医学审核；自动来源/页码检查不能证明推导建议必然正确。
- 没有自动医学准确率；没有使用 GT 评分或修改 requirements。
- 此 22 条是既有开发 smoke，不是新的 blind test。
- 实验绑定代码、配置、知识、索引和选样；升级代码后需新 preflight/输出目录，不能继续旧冻结实验。
