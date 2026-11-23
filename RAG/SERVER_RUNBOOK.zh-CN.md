# RAG 服务器执行手册

当前分支：`codex/relive-surgical-visual-knowledge-v01`。
所有构建命令在 **仓库根目录**执行，不在 `relive/` 内执行。
本任务调用外部文本 API，不需要占用 A100，也不需要 MedVidU 图片或 QA 数据。

仓库已经包含工具、46 份 Git LFS PDF、SOP 和初始 inspect/mock 审计快照。
初始正式文件仍为 INSPECT_ONLY、0 卡片；mock 的 8 条不能作为正式医学知识。
不要覆盖这些快照。服务器的新结果放入 `RAG/output/server_<时间>/`。

## 1. 环境、代码和 PDF

首次创建专用环境；已有此环境时只需 activate。

```bash
source /home/huihui/miniconda3/etc/profile.d/conda.sh
conda create -n relive-rag-kb -c conda-forge python=3.12 git-lfs -y
conda activate relive-rag-kb

# 保留原 checkout 的分支和未提交内容，新建独立 worktree。
REPO=/home/huihui/codes/MedVid_understanding
WORKTREE=/home/huihui/codes/MedVid_understanding_rag
git -C "$REPO" fetch origin codex/relive-surgical-visual-knowledge-v01
git -C "$REPO" lfs install --local --skip-smudge
GIT_LFS_SKIP_SMUDGE=1 git -C "$REPO" worktree add --detach "$WORKTREE" FETCH_HEAD
cd "$WORKTREE"
git lfs pull origin
git lfs fsck
python -m pip install -r RAG/requirements-server.lock.txt
python -m unittest discover -s RAG/tests -v
```

如果 WORKTREE 已存在，先检查其中的 `git status` 和 `git rev-parse HEAD`，
不要再执行 worktree add，也不要 reset/clean。需要更新时创建另一个新的目录。
Git LFS 若报告已有自定义 hook 冲突，不使用 `--force`，先保留并报告现有配置。

确认下载的是完整 PDF，并逐个核对本地已扫描的 SHA-256：

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('RAG/output/source_manifest.json').read_text())
root = Path('RAG/references')
expected = {s['file_path']: s for s in manifest['sources']}
actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
          if p.is_file() and p.suffix.lower() == '.pdf'}
assert actual == set(expected), 'PDF 文件集合不一致'
for name, source in expected.items():
    path = root / name
    with path.open('rb') as f:
        assert f.read(5) == b'%PDF-', f'LFS 尚未下载真实 PDF: {name}'
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    assert h.hexdigest() == source['sha256'], f'内容哈希不匹配: {name}'
print('PASS: 全部', len(expected), '份 PDF 与本地扫描哈希一致')
PY
```

预期当前 46 份、子目录 29 份；算法不依赖固定数量。任何下载或哈希检查失败时停止。

## 2. 保存参数并重新 inspect

```bash
cd /home/huihui/codes/MedVid_understanding_rag
conda activate relive-rag-kb
RAG_RUN="$PWD/RAG/output/server_$(date +%Y%m%d_%H%M%S)"
PDF_ROOT="$PWD/RAG/references"
CACHE_ROOT="$PWD/RAG/cache"
SOURCE_REVIEW="$RAG_RUN/source_review.completed.jsonl"
mkdir -p "$RAG_RUN"
export RAG_RUN PDF_ROOT CACHE_ROOT SOURCE_REVIEW

python - "$RAG_RUN/server_variables.sh" <<'PY'
import os, shlex, sys
from pathlib import Path
names = ['RAG_RUN', 'PDF_ROOT', 'CACHE_ROOT', 'SOURCE_REVIEW']
Path(sys.argv[1]).write_text(''.join(
    f'export {n}={shlex.quote(os.environ[n])}\n' for n in names))
PY

python -m RAG.knowledge_builder.build --mode inspect \
  --input-dir "$PDF_ROOT" --cache-dir "$CACHE_ROOT" \
  --output "$RAG_RUN/inspection/relive_surgical_visual_knowledge_v0.1.json" \
  --verbose

cat "$RAG_RUN/inspection/build_summary.json"
```

预期 exit 0、INSPECT_ONLY、46 个文档、2,136 页、0 API 调用。
130 个稀疏文本/图片页属于待处理内容，不代表已完成视觉理解。
需要恢复变量时，source 当前 run 下的 `server_variables.sh`；其中没有 API Key。

## 3. 完成来源审核

```bash
cp -n "$RAG_RUN/inspection/source_review_template.jsonl" "$SOURCE_REVIEW"
nano "$SOURCE_REVIEW"
```

保持 source_id、file_path、title、sha256 不变。人工填写：

- decision：INCLUDE、EXCLUDE 或 PENDING；
- reviewer_id、rationale；
- independence_basis：说明独立教学来源依据，INCLUDE 时必填；
- allowed_pages：整份独立资料可用时为 []；仅接纳部分页面时填真实 PDF 页码。

不要因为想跑通程序而统一填写 INCLUDE。OpenSurgicalVision 含项目标签/mask
规则，建议本轮 EXCLUDE；不得默认接纳整个文件。

首次 smoke 至少要明确审查以下三份；只有 INCLUDE 的资料会发给 API：

1. `common_surgical_instruments_module.pdf`
2. `DualKnife_J_Professor_Naohisa_Yahagi.pdf`
3. `OpenSurgery_pdfs/003_handling-of-surgical-instruments.pdf`

准备“全量已审查来源”构建前，应明确处理全部来源：INCLUDE 或 EXCLUDE。
PENDING 会被跳过，不能将这种结果称为全部资料均已抽取。

## 4. 在服务器配置 API

通过终端输入，Key 不回显、不写入参数文件：

```bash
read -r -p 'API Base URL（包含 /v1，不含 /chat/completions）: ' LLM_BASE_URL
read -r -p '文本模型名称: ' LLM_TEXT_MODEL
read -r -s -p 'API Key（不回显）: ' LLM_API_KEY
printf '\n'
export LLM_BASE_URL LLM_TEXT_MODEL LLM_API_KEY
export LLM_TIMEOUT=120
export LLM_MAX_CONCURRENCY=3
export LLM_MAX_RETRIES=3
export LLM_REQUESTS_PER_MINUTE=20
export LLM_RESPONSE_FORMAT=auto
export ENABLE_VISION_EXTRACTION=false
```

使用你已开通的 OpenAI-compatible Chat Completions 服务及真实模型名。
不要把 Key 发到聊天或写进 Git。重开终端后需要重新安全注入 Key。
未配置价格时成本报告为 null，不会假装费用为 0；调用数和可获得的 token 用量仍记录。

## 5. 三份资料的真实 API smoke

```bash
python -m RAG.knowledge_builder.build --mode build \
  --input-dir "$PDF_ROOT" --cache-dir "$CACHE_ROOT" \
  --source-review "$SOURCE_REVIEW" \
  --include-file common_surgical_instruments_module.pdf \
  --include-file DualKnife_J_Professor_Naohisa_Yahagi.pdf \
  --include-file OpenSurgery_pdfs/003_handling-of-surgical-instruments.pdf \
  --max-pages 5 --max-api-calls 32 \
  --output "$RAG_RUN/api_smoke/relive_surgical_visual_knowledge_v0.1.json" \
  --verbose

python -m RAG.knowledge_builder.build --mode validate \
  --input-dir "$PDF_ROOT" --cache-dir "$CACHE_ROOT" \
  --output "$RAG_RUN/api_smoke/relive_surgical_visual_knowledge_v0.1.json"

cat "$RAG_RUN/api_smoke/build_summary.json"
cat "$RAG_RUN/api_smoke/quality_report.json"
```

先检查是否为 BUILT_PENDING_MEDICAL_REVIEW、synthetic_test=false、真实调用数大于
0、chunk_count 大于 0，且无未解决技术失败。32 是含重试及一致性校验的调用上限，
不保证一定完成。若 SOURCE_REVIEW_REQUIRED、API 配置错误、预算停止或卡片为 0，
先看报告，不进入全量构建。

人工核对文本、外观/运动特征、原文摘录、页码、derived guidance 和不足证据条件。
所有卡片必须保持 pending；结构校验通过不代表医学上正确。

## 6. 审查 smoke 后，后台执行全量文本抽取

在已完成来源审核且 smoke 可接受后执行。默认不启用视觉 API。

```bash
nohup bash -c '
  python -u -m RAG.knowledge_builder.build --mode build \
    --input-dir "$PDF_ROOT" --cache-dir "$CACHE_ROOT" \
    --source-review "$SOURCE_REVIEW" --max-api-calls 2000 \
    --output "$RAG_RUN/full/relive_surgical_visual_knowledge_v0.1.json" \
    --verbose
  code=$?
  printf "%s\n" "$code" > "$RAG_RUN/full.exitcode"
  exit "$code"
' > "$RAG_RUN/full.log" 2>&1 < /dev/null &
echo $! > "$RAG_RUN/full.pid"
echo "PID=$(cat "$RAG_RUN/full.pid")"
echo "RUN=$RAG_RUN"
```

查看状态：

```bash
tail -n 30 "$RAG_RUN/full.log"
if test -f "$RAG_RUN/full.exitcode"; then
  cat "$RAG_RUN/full.exitcode"
else
  ps -p "$(cat "$RAG_RUN/full.pid")" -o pid,etime,args
fi
```

2,000 是调用预算，可能不足以处理全部候选章节；检查最终 coverage、跳过来源、
失败列表和 budget stop，不能只看进程结束。若预算或失败结果已经封存，需要新输出
目录、同一 cache 继续；不要覆盖已封存报告。若进程中途被打断且尚未封存，使用相同
命令、相同输出与参数并加 `--resume`。已完成结果的 --resume 仅校验复用，不重新抽取。

## 7. 完成后离线验证及提交检查结果

```bash
python -m RAG.knowledge_builder.build --mode validate \
  --input-dir "$PDF_ROOT" --cache-dir "$CACHE_ROOT" \
  --output "$RAG_RUN/full/relive_surgical_visual_knowledge_v0.1.json"

cat "$RAG_RUN/full/build_summary.json"
cat "$RAG_RUN/full/quality_report.json"
```

应保留主知识 JSON、source_manifest、quality_report、accepted_candidate_audit、
excluded_knowledge、extraction_failures 和 API/cache 记录。将 summary、quality_report
及少量真实卡片发来复核；不要发送 Key。

Vision-assisted 提取应在文本抽取之后单独规划小范围真实 smoke：显式设置视觉模型、
`--enable-vision --max-vision-pages 3`，使用新输出目录。不要对 1,094 个候选页面直接全开。
本轮任何结果都不生成 certificate 或 VERIFIED，也不改变 ReliVE 已冻结实验。
