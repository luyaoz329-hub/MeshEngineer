# 测试素材来源

四个人工主夹具由本项目制备。关联测试常量的来源、许可和制备方式如下。

- `compression.json` 的普通 Box 与 Draco 数据取自 [Khronos Box](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Box)。Copyright 2017 Cesium，按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 使用；本项目抽取 JSON/二进制段并重新组装测试容器。
- `compression.json` 的 meshopt 常量由[官方 meshoptimizer](https://github.com/zeux/meshoptimizer) 编码器制备，并通过独立解码接口回读核验。具体来源 blob、制备与核验记录保存在该 JSON 中；编码器和解码器均非产品依赖。
- `unsupported-encodings.json` 保存 [three.js KTX2 测试图像](https://github.com/mrdoob/three.js/blob/dev/examples/textures/ktx2/2d_etc1s.ktx2) 的966字节原始BasisLZ数据及blob/SHA。MIT许可与版权声明见 [THREEJS-LICENSE.txt](THREEJS-LICENSE.txt)。

GLB1拒绝例根据 [Khronos旧版容器规范](https://github.com/KhronosGroup/glTF/tree/main/extensions/1.0/Khronos/KHR_binary_glTF) 及其schema独立构造。压缩边界测试不运行解码器，也不证明压缩分支与普通回退逐值等价。
