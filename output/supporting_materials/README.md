# 支撑材料复现说明

本目录对应论文附录中的支撑材料文件清单，不含参赛者、学校或赛区身份信息。
赛题给出的 9 个观测点和固定参数已写入各程序，未重复提交赛题原始附件；论文未使用
自主查阅的外部数据集。

## 环境

- Python 3.10 或更高版本
- 依赖见 `requirements.txt`

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 运行顺序

各脚本应在其所在目录内运行，以保证结果文件写入对应目录。

```bash
cd Q1
python logistic_model.py

cd ../Q2
python advanced_model.py
python bootstrap_validation.py

cd ../Q3
python stability_analysis.py

cd ../Q4
python sensitivity_analysis.py
```

主要输出：

- `Q1/logistic_fit.png`
- `Q2/model_q2.png`
- `Q3/stability_analysis.png`
- `Q3/collapse_scenario.png`（补充情景图，正文未引用）
- `Q4/sensitivity_tornado.png`
- `Q4/sensitivity_envelope.png`

`results/` 中保存了论文正文实际引用的五幅复现结果图，便于直接核对。
Bootstrap 程序使用固定随机种子 `42`，因此在相同依赖版本下可复现论文报告的区间。

## AI 工具记录

`AI工具使用详情.pdf` 记录工具名称和版本、具体使用目的、关键交互，以及人工复核和
修改情况。若提交前使用了其他 AI 工具或新增关键交互，应如实更新该文件和论文参考文献。
