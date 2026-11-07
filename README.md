# MeshEngineer

MeshEngineer 面向静态三角网格的检查、修复、评价与批处理。当前提供 GLB/STL 只读导入，以及基于资产清单的批量导入检查。

资产可存放在本地、外置硬盘或挂载目录，通过独立存储配置定位。

## CLI

查看命令帮助：

```sh
meshengineer --help
meshengineer import-check --help
```

按清单批量检查资产并生成 JSONL 报告：

```sh
meshengineer import-check \
  --manifest assets.jsonl \
  --storage storage.json \
  --report import-report.jsonl
```

- `--manifest`：JSONL 资产清单。
- `--storage`：数据源标识到绝对根目录的 JSON 映射。
- `--report`：新建的 JSONL 报告文件；父目录须已存在。

退出码：`0` 全部导入成功；`1` 存在资产失败或批次中止；`2` 参数、配置或预检失败。
