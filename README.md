# ELRSlang

ELRSlang 是一个 SlangPy-first 的实时渲染器原型。PC 端用 Python/SlangPy 作为快速迭代 host，shader、graph 和 scene manifest 设计成可以被 Android Vulkan 原生 smoke app 复用的资源 contract。

## 当前能力

- `RenderPass` / `RenderGraph`：参考 Falcor 的 pass/graph 思路，支持 graph JSON、依赖编译、循环检测、资源边检查、graph settings、external input、输出枚举和基础 pass timing。
- `SlangFunctionPass`：直接调用 `.slang` 函数，使用 SlangPy 的 `_result`、`call_id()`、dict `_type` 等能力减少 CPU 侧 glue code。
- `HardwareRasterPass`：使用 SlangPy render pipeline 绘制完整 scene instances，支持 camera view/projection、材质基础色、简单 Lambert/unlit 混合和 `.pyscene` procedural mesh。
- `BuildAccelerationStructurePass` / `HardwareDXRPass`：把 scene instances 烘成 world-space geometry，构建 BLAS/TLAS，并从 camera 发射 primary rays 做硬件 ray tracing smoke。
- `hybrid_debug` graph：同时跑 raster 和 DXR，再用 Slang function pass 做 composite，验证混合渲染共享同一份 scene contract。
- 场景导入：支持 OBJ、glTF/GLB、FBX(`ufbx` 后端)、USD(`usd-core` 后端) 和 Falcor `.pyscene` 兼容层。
- Falcor `.pyscene`：支持常见 `sceneBuilder`、`Camera`、`Transform`、`TriangleMesh`、`Material`/`StandardMaterial`、`EnvMap`、light、`float3/float4` 等白名单 API；不执行任意 Python import/open/eval/exec。
- Viewer：支持 headless frames、交互窗口、F2 截图、W/A/S/D + mouse look 漫游、per-pass timing 输出。
- Android smoke：原生 Vulkan app 会检查 Vulkan instance，并尝试读取导出的 `assets/elrslang/manifest.json` 和 `graph.json`。

## 稳定安装

稳定方案固定为 Windows x64 + CPython 3.12 + `slangpy==0.40.1`。不要用 Python 3.14 作为可复现环境，因为当前 `slangpy 0.40.1` 没有可用的 cp314 wheel。

推荐安装 Python 3.12：

```powershell
winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
```

安装完成后，关闭并重新打开 PowerShell。然后在仓库根目录运行：

```powershell
cd G:\J_Pan\Code\Mine\ELRSlang
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
```

如果 Python 3.12 已经安装但不在 `PATH`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -PythonExe "C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe"
```

只安装依赖、不跑 GPU smoke：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -SkipSmoke
```

环境装乱后重建：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -Recreate
```

## 运行

Headless smoke：

```powershell
.\.venv\Scripts\python -m elrslang.viewer --frames 1 --graph slangpy_preview --backend automatic --width 32 --height 32
.\.venv\Scripts\python -m elrslang.viewer --frames 1 --graph raster_forward --backend automatic --width 32 --height 32
.\.venv\Scripts\python -m elrslang.viewer --frames 1 --graph dxr_pathtrace --backend automatic --width 32 --height 32
.\.venv\Scripts\python -m elrslang.viewer --frames 1 --graph hybrid_debug --backend automatic --width 32 --height 32
```

加载 Falcor `.pyscene`：

```powershell
.\.venv\Scripts\python -m elrslang.viewer --frames 1 --scene "G:\J_Pan\Code\Mine\ELRSlang\assets\scenes\falcor\falcor_pyscene\cornell_box.pyscene" --graph raster_forward --backend automatic --width 64 --height 64 --print-timings
```

打开交互窗口：

```powershell
.\.venv\Scripts\python -m elrslang.viewer --scene "G:\J_Pan\Code\Mine\ELRSlang\assets\scenes\falcor\falcor_pyscene\cornell_box.pyscene" --graph raster_forward --backend automatic --width 1280 --height 720
```

交互控制：

- `W/A/S/D`：前后左右移动
- `Q/E`：上下移动
- `Shift`：加速
- `Ctrl`：减速
- 右键拖拽：mouse look
- `F2`：截图
- `Esc`：关闭窗口

常见 warning：

```text
[WARN] Cannot enable D3D12 Agility SDK...
```

如果命令最后显示 `Rendered 1 frame(s)...` 并返回退出码 0，这个 warning 可以忽略。它通常来自 Python 安装盘符和 SlangPy/D3D12 Agility SDK 所在盘符不同。

## 测试

```powershell
.\.venv\Scripts\python -m compileall src tests
.\.venv\Scripts\python -m unittest discover -s tests
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\smoke.ps1
```

## 导出 Android 资源

```powershell
.\.venv\Scripts\python -m elrslang.tools.export_mobile --graph hybrid_debug --scene "G:\J_Pan\Code\Mine\ELRSlang\assets\scenes\falcor\falcor_pyscene\cornell_box.pyscene" --out mobile\android\app\src\main\assets\elrslang
```

导出结果包含：

- `manifest.json`
- `graph.json`
- `scene.json`
- `shaders/*.slang`

Android 工程位于 `mobile/android`。当前本机没有检测到 `gradle`，因此 Android assemble 尚未在本机验证。

## 设计边界

- SlangPy-first 不等于所有渲染都走软件模拟：全屏、compute、后处理和数据转换 pass 优先直接调用 Slang 函数；硬件 raster/DXR 仍走 SlangPy 暴露的底层 graphics/ray tracing API。
- `.pyscene` v1 是受控 Falcor 兼容层，目标是常见 scene construction 脚本和仓库内样例，不承诺执行任意 Python。
- Android 不嵌入 Python，也不运行 SlangPy host；它消费 PC 侧导出的 baked graph/shader/scene contract。
- 当前 raster/DXR 是基础闭环，不是完整 PBR renderer；后续可以在同一 scene contract 上扩展 GBuffer、material system、shadow、denoise、TAA 等功能。
