# ELRSlang Android smoke app

这个目录是首版 Android Vulkan 原生 smoke app 骨架。它的职责是验证移动端 host 能消费 PC 侧导出的 graph/shader/resource contract，而不是运行 Python 或 SlangPy。

## 资源导出

在仓库根目录执行：

```powershell
python -m elrslang.tools.export_mobile --graph slangpy_preview --out mobile/android/app/src/main/assets/elrslang
```

导出结果包含：

- `manifest.json`
- `graph.json`
- `shaders/*.slang`

## 构建边界

- 当前 native 层只创建 Vulkan instance 并返回 smoke 状态。
- 后续移动端实现应在 `native-lib.cpp` 中接入 Slang/GFX 或平台 shader 编译产物。
- Android 不嵌入 Python，不加载 SlangPy。
