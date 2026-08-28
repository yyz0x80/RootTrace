# RootTrace

[English](README.en.md) | 简体中文

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Multi-Agent](https://img.shields.io/badge/architecture-multi--agent-orange.svg)
![Root Cause Analysis](https://img.shields.io/badge/purpose-root--cause--analysis-green.svg)

面向 GitHub Issue、CI 失败与回归问题的证据驱动多 Agent 根因分析系统。

它回答：**什么失败了、可能的根因在哪里、为什么发生、什么证据支持该结论**。

## 架构

```text
Issue / CI / Stack Trace / PR Context
                 ↓
             Lead Agent
                 ↓
   ┌─────────────┼─────────────┐
   ↓             ↓             ↓
Issue/CI       Code        Git History
Specialist   Specialist     Specialist
   └─────────────┼─────────────┘
                 ↓
           EvidenceGraph
                 ↑
      Historical RCA Retrieval
                 ↓
         Ranked Hypotheses
                 ↓
       Runtime/Test Verifier
       (ephemeral sandbox)
                 ↓
          Final RCA Report
```

## 核心特性

* 并行收集 Issue/CI、代码与 Git 历史的证据。
* 结构化 `EvidenceGraph`，支持来源追踪与矛盾检测。
* 可证伪的根因假设与运行时/测试验证。
* 对原始仓库只读分析；测试仅在一次性沙盒中运行。
* 可选的历史 RCA 检索，使用 TF-IDF、MiniBatchKMeans 与 Top-K 相似度。
* 输出 JSON/Markdown 报告、执行追踪、时序与提供商用量。

## 快速开始

```bash
git clone https://github.com/yyz0x80/RootTrace.git
cd RootTrace
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

编辑 `.env`，将 `your_api_key` 替换为有效密钥。根据你的提供商配置 `ROOTTRACE_BASE_URL` 与 `ROOTTRACE_MODEL`。

运行分析：

```bash
roottrace rca \
  --repo /path/to/repo \
  --issue /path/to/issue.md \
  --model glm-4.7-flash \
  --output-dir output/roottrace-demo
```

可选证据文件：`--stack-trace`、`--ci-log`、`--pr-diff`。输出包含 `rca_report.md`、`rca_report.json` 与 `evidence_graph.json`。

## Benchmark

基于 [SWE-bench Verified](https://github.com/princeton-nlp/SWE-bench) 派生的固定 50-case 开发集的结果：

| Model             | Top-1 File Localization | Any File Recall@5 |
| ----------------- | ----------------------: | ----------------: |
| DeepSeek-V4-Flash |                  73.47% |            80.59% |

**注意：** 这些结果来自受控的 50-case 开发子集，不应解释为官方 SWE-bench Verified 500-case 基准测试分数。该项目包括受控消融研究，涵盖智能体组成、历史检索和聚类策略。

## 适用范围

RootTrace 诊断问题并推荐修复范围，但在 RCA 模式下**不**编辑代码、生成/应用补丁、提交、推送、合并或创建 PR。

## 许可证

采用 Apache License 2.0 许可证。
