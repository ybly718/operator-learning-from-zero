# 贡献指南

## 文献

- 使用稳定 citation key。
- 优先正式出版版本，避免同一成果的 arXiv 与期刊/会议版本重复。
- 新增前沿论文时，同时说明它解决的问题、与基线的公平比较条件及复现入口。

## 实验

- 新模型必须复用公共数据划分与 metrics。
- 任何“优于”结论都要给出多随机种子结果、训练预算和硬件信息。
- 小规模 smoke 配置负责快速验证；full 配置才用于报告结果。

## 提交前

    python -m pytest
    ruff check .
    PYTHONPATH=src python scripts/run_experiment.py --config configs/smoke.json

提交结果文件时只保留轻量 JSON/图表；模型权重和大型数据应放在外部存储，并提供 manifest。
