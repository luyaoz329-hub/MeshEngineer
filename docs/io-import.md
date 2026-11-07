# IO 文件导入设计

状态：用户已批准整批工程初始化与 IO 方案，首版本地实现；远端执行仍未验证。职责沿用 [顶层架构](architecture.md)。

## 1. 接口与分层

| 所在层 | 入口/结构 | 契约 |
| --- | --- | --- |
| Core | `core/assets.py`：AssetRef、AssetSource、Asset、MeshData、NodeData、材质/资源结构 | 普通类型化 dataclass 与 NumPy，不暴露具体后端对象 |
| Core | `core/imports.py`：AssetImportError、ImportRecord、BatchSummary | 五类导入错误、小型 JSON 结果与批次完成状态 |
| IO | `io/loader.py`：`load_asset(ref: AssetRef, roots: Mapping[str, Path]) -> Asset` | 单资产只读加载；同一 bytes 计算哈希并解析 |
| IO | `io/stl.py`、`io/glb.py` | 自写 STL；pygltflib 读 GLB 容器/内嵌数据，NumPy 适配几何与已支持属性 |
| Pipeline | `pipeline/import_check.py`：`check_asset` | 调用 IO，形成单资产导入记录；不作质量判断或修复 |
| Curation | `curation/manifest.py`：`read_manifest`；`curation/batch.py`：`open_report`、`run_import_check` | 全清单预检、顺序调度、逐条报告和批次停止策略 |
| CLI | `cli/main.py`：`main` | `meshengineer import-check --manifest ... --storage ... --report ...` |

Core 提供定义，不承担路由；Curation 调用 Pipeline，Pipeline 调用 IO。其他检测/修复模块仍为骨架。本里程碑不实现导出、云下载、质量检测、修复算子或修复回退；保留后续模块职责。

## 2. 位置、身份和内容

存储 JSON 为 `source_id -> 绝对根目录`；清单 JSONL 每行只允许 `source_id`、`asset_id`、`relative_path`、可选 `expected_sha256`。身份由前两者组成，路径不承载身份，预期哈希可固定内容版本。哈希必须是 64 位十六进制；省略与 null 不等价，清单显式 null 非法。重复身份、未知/重复键、空行、注释、空清单和错误字段在读取任何资产前拒绝。

清单位置与资产位置独立，迁移只改存储配置。首版接受本地、外置硬盘和挂载目录；云端由用户先下载或挂载。根须为可访问绝对目录；相对路径使用 `/`，禁止绝对写法、反斜杠和 `..` 段。解析符号链接后仍须为根内普通文件，根内合法链接允许。配置错误用 ValueError/启动退出码 2，单资产越界用 invalid_path。

文件只打开一次读取为完整 bytes，读取前后 fstat 核对 size/mtime；同一 bytes 计算 SHA-256、校验预期并导入。单资产整体进入内存，不宣称对超大单文件流式解码；批次只按序保留一个资产，不一次加载整个数据集。未预期的程序异常不转成坏资产。

## 3. 统一内存模型

`Asset` 的 meshes 是独立 primitive 几何，nodes 保存层级和对几何的引用。V 为 float64 `(N,3)`，F 为 int64 `(M,3)`；提供给下游几何后端时按需传递二者，不在导入层增加 PyMeshLab/libigl 依赖。

MeshData 分别保存可选顶点法线/面法线、多组 UV/颜色、切线（含 w）、材质索引、自定义数值属性及 accessor 来源。已知属性转为 float64；自定义属性保留源 dtype、normalized 标记与 accessor/mesh/primitive 编号。缺失仍为 None/空映射，不补算。

NodeData 保存 parent、mesh_indices、原节点编号和 float64 局部 4×4 矩阵。实例共享几何，不烘焙世界变换、不改绕序。材质保存基础 PBR、alpha、doubleSided、emissive、unlit、五纹理槽与绑定/UV 变换；图像保存原编码 bytes/MIME，纹理/采样器按源引用复用，省略 filter 保留 None。

Source 保存引用、实际路径、输入哈希、格式、所选场景/STL 头、names/extras/camera 来源与保真说明。原生 GLB 后端对象不进入公共类型。

## 4. 首版格式边界

| 格式/行为 | 处理 |
| --- | --- |
| 入口分派 | 清单路径后缀 `.glb`/`.stl`，大小写不敏感；内容校验，不在格式间猜测 |
| GLB | glTF/GLB 2.0；静态、自包含、TRIANGLES；默认场景或唯一场景及其可达节点；多 primitive/实例/层级 |
| 数组 | 普通 accessor、offset/stride、normalized 整数 UV/颜色、合法零初值；索引 uint8/16/32；拒绝 sparse、量化和其他不支持形状 |
| 资源 | 内嵌 BIN/data URI；PNG/JPEG 原编码；KHR_texture_transform、KHR_materials_unlit 描述保留 |
| 压缩 | Draco、EXT/KHR meshopt 仅在非 required 且具有完整、有效普通 accessor 回退时读取回退；required 或无真实回退拒绝，不加载解码器 |
| 明确拒绝 | 实际动画/蒙皮/morph（包括未选场景）、非三角图元、未知或未支持 optional/required 扩展、外部资源 URI；未实际使用 skins 定义允许 |
| STL | 精确匹配 `84 + 50×面数` 的普通二进制优先，否则严格 ASCII 单 solid；保留面序/绕序/原面法线，float64 精确数值共享索引 |
| STL 收窄 | 不做近似焊接；颜色标记、非零属性、多 solid 拒绝；破损/截断/非有限数据报格式错误，无面不作为可用资产 |

导入器保留退化/重复/非流形/相交等质量缺陷供下游检测，不静默清理。GLB 图像仅检查 MIME/签名并保留字节，未作像素解码或视觉验证；普通回退可读也不证明与压缩分支等价。这些限制写入 source notes 和导入报告。

## 5. 错误、报告和资产保护

单一 AssetImportError 带 ref、code、英文 message 和可取得的实际 SHA-256：`invalid_path`、`read_error`、`hash_mismatch`、`invalid_asset`、`unsupported_asset`。失败不返回部分 Asset。

ImportRecord 不保存几何/图像，只保存来源、状态、实际哈希、几何/节点实例计数、场景、错误、说明和环境版本。顶点/面数按独立 primitive 计；instance_count 为引用网格的节点数，不把 primitive 摆放数量当实例数。

清单预检后，报告先检查与清单、配置、各输入的冲突，再独占新建，禁止覆盖；父目录必须已存在。每个资产结果写 JSONL 并 flush。普通坏文件继续后项；存储根失效记录当前 read_error 后停止，报告写/flush/close 失败停止。BatchSummary 区分 total、processed（已形成结果）、reported（成功刷入报告）、ok、failed、completed、abort_reason。未知程序错误保留栈。

输入只读，不覆写或移动原资产。首版无修复，不将“没有修改原件”冒充已验证修复回退；修复结果的复检与回退仍由后续 Pipeline/Evaluation 实现。

## 6. 验收与部署

工程固定 Python、uv 和直接依赖；strict mypy 覆盖源码。pygltflib 未提供类型标记，只有精确定位的 import-untyped 忽略，公共结构和适配逻辑继续检查，不启用全局忽略。

基础集用四个人工模型与独立临时变体验证清单、路径、哈希、几何/属性、资源、拒绝和批次故障；真实小集与完整登记集使用仓外模型及版本化独立预期。预期源于原始结构/编码值，不由被测导入器自动批准；模型哈希冻结，运行前后复核原件。缺数据和未跑完失败，不以 skip 冒充通过。

### 工程环境与基础检查

准备普通 CPython 3.14.6 和 uv 0.13.7，在项目根目录执行：

```sh
uv sync --locked --no-python-downloads
uv run --locked python -m unittest discover -s tests/base -t . -v
uv run --locked mypy src
```

版本以 `pyproject.toml`、`.python-version` 和 `uv.lock` 为准；项目不自动下载 Python。Hatchling 仅负责构建，隔离构建的传递依赖不宣称由运行锁完整覆盖。

### 分层回归与外置数据

| 层级 | 范围与触发 |
| --- | --- |
| 基础集 | `tests/base/`，不依赖外置盘和网络；独立实现改动后完整执行 |
| 真实小集 | `tests/manifests/real-small.jsonl`，当前3项；涉及IO、资产模型、清单或批次行为时执行 |
| 完整登记集 | `tests/manifests/registered.jsonl`，当前6项；里程碑交付前完整执行该版本全部条目 |

完整登记集指冻结清单的全部条目，不代表公开数据集全量。模型保留在仓外，仅身份、相对路径、哈希和独立预期入仓。参照 `configs/storage.example.json` 建立本机 `configs/storage.local.json` 并填写绝对根目录；更换机器只改根映射。

```sh
MESHENGINEER_TEST_MANIFEST=tests/manifests/real-small.jsonl \
MESHENGINEER_TEST_STORAGE=configs/storage.local.json \
MESHENGINEER_TEST_EXPECTATIONS=tests/expectations/real-small.json \
uv run --locked python -m unittest discover -s tests/external -t . -v
```

完整登记集将两处 `real-small` 替换为 `registered`。可选 `MESHENGINEER_TEST_ARTIFACT_DIR` 指定新结果目录，保存实际选集、预期、数量、JSONL报告和摘要；已有报告不可覆盖。测试方法数与实际资产数分别核对。

来源版本、许可和下载URL记录在 `tests/expectations/`；Thingi10K来源摘录见 `thingi-provenance.json`。本轮使用既有本地模型，未重新下载对照上游副本。`tests/prepare_external.py` 根据源结构制备新清单/预期，不调用产品导入器，不覆盖旧基线；日常回归使用已冻结清单。提议新测试集时，可指定新的 `--output-root`；更换STL样本时通过 `--thingi-provenance` 提供同结构来源JSON。

### 远端执行前提

`.github/workflows/regression.yml` 在PR运行基础集与类型检查；手动选择 `real-small` 或 `registered` 时，外置集使用带 `meshengineer-data` 标签的Linux x64自托管runner。需要先挂载模型，并将仓库变量 `MESHENGINEER_TEST_STORAGE` 设为runner上仓外存储配置的绝对路径，模型也放在checkout工作区之外。

模型不放入Actions cache；artifact保存选集、预期、版本、报告和日志14天。自托管runner需支持工作流所用的Node24 Actions。真实小集尚未配置为PR自动门槛，远端执行尚未验证。

未承诺的范围：全部公开数据集覆盖、超大单文件压力/峰值内存、像素/视觉保真、压缩解码与等价、下游修复后端联调、跨平台联合验证。远端 runner/数据部署与实际 Actions 结果须另取得证据；本地工作流配置不能代表远端通过。
