# ELRSlang

ELRSlang 是一个 SlangPy-first 的实时渲染器原型。首版目标是用 Python/SlangPy 快速搭建 PC 端可运行的 viewer，同时把 shader、资源 manifest 和 render graph 设计成可被 Android Vulkan 原生 smoke app 复用的形态。

## 当前能力

- `RenderPass` / `RenderGraph`：参考 Falcor 的 pass/graph 思路，支持 graph JSON、依赖编译、循环检测、资源边检查和顺序执行。
- `SlangFunctionPass`：默认 pass 类型，直接调用 `.slang` 函数，使用 SlangPy 的 `_result`、`call_id()`、dict `_type`、broadcasting 等能力减少 CPU 侧 glue code。
- `PipelinePass`：为必须使用固定管线的硬件光栅化和硬件光追保留扩展点。
- `HardwareRasterPass`：首版实现为 SlangPy render pipeline 绘制场景中的第一个 mesh，并自动居中缩放到屏幕。
- `BuildAccelerationStructurePass` / `HardwareDXRPass`：实现 BLAS/TLAS 构建和 SlangPy ray tracing 调用骨架，按设备 feature gate 执行。
- 场景导入：内置 OBJ 和最小 glTF JSON 解析；FBX、GLB、USD 识别为可选依赖路径。
- Android smoke：提供 Android Vulkan 原生工程骨架，不运行 Python/SlangPy host，复用导出的 graph/shader/resource contract。

## 稳定安装

稳定方案固定为 Windows x64 + CPython 3.12 + `slangpy==0.40.1`。不要用 Python 3.14 作为可复现安装环境，因为当前 `slangpy 0.40.1` 没有可用的 cp314 wheel。

先安装 CPython 3.12 x64。推荐用 Windows 自带的 `winget`：

```powershell
winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
```

安装完成后，关闭并重新打开 PowerShell。

如果你的机器没有 `winget`，请从 Python 官网安装 Windows x86-64 installer，并在安装时勾选 `Add python.exe to PATH`：

```text
https://www.python.org/downloads/release/python-31210/
```

然后在仓库根目录运行 bootstrap。它会创建 `.venv`、安装锁定依赖、检查 SlangPy 版本、运行单元测试和三条 renderer smoke：

```powershell
cd G:\J_Pan\Code\Mine\ELRSlang
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
```

如果 Python 3.12 已经安装，但不在 PATH，可以直接传 `python.exe` 的完整路径：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -PythonExe "C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe"
```

如果希望脚本尝试用 `winget` 安装 Python 3.12：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -InstallPython
```

如果只想安装依赖、不跑 GPU smoke：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -SkipSmoke
```

如果环境装乱了，可以重建：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -Recreate
```

## 测试

```powershell
.\.venv\Scripts\python -m unittest discover -s tests
.\.venv\Scripts\python -m compileall src tests
```

也可以直接运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\smoke.ps1
```

## 运行 viewer

离屏跑一帧 smoke：

```powershell
.\.venv\Scripts\python -m elrslang.viewer --frames 1 --graph slangpy_preview --backend automatic --width 32 --height 32
.\.venv\Scripts\python -m elrslang.viewer --frames 1 --graph raster_forward --backend automatic --width 32 --height 32
.\.venv\Scripts\python -m elrslang.viewer --frames 1 --graph dxr_pathtrace --backend automatic --width 32 --height 32
```

如果看到类似下面的日志，但命令最后显示 `Rendered 1 frame(s)...` 并返回退出码 0，就表示运行成功：

```text
[WARN] Cannot enable D3D12 Agility SDK...
[INFO] (rhi) layer: CreateDevice: Debug layer is enabled.
```

这个 warning 来自 SlangPy/D3D12 Agility SDK：当前 Python 安装在 `C:`，项目和 SlangPy wheel 在 `G:`，两者不在同一个盘符。它会影响 Agility SDK 加载提示，但不代表 renderer 失败。

打开交互式窗口：

```powershell
.\.venv\Scripts\python -m elrslang.viewer --graph slangpy_preview --backend automatic --width 1280 --height 720
```

加载场景：

```powershell
.\.venv\Scripts\python -m elrslang.viewer --scene path\to\scene.obj --graph raster_forward --backend automatic
```

可用 graph：

- `slangpy_preview`：用 Slang 函数直接生成 debug view，适合快速验证 SlangPy 数据绑定。
- `raster_forward`：使用硬件 render pipeline 画全屏 quad，并经过 tonemap/present。
- `dxr_pathtrace`：构建 acceleration structure 后调用 SlangPy ray tracing；如果设备缺少 `acceleration_structure` 或 `ray_tracing` feature，会给出明确错误。

## 导出 Android smoke 资源

```powershell
.\.venv\Scripts\python -m elrslang.tools.export_mobile --graph slangpy_preview --out mobile\android\app\src\main\assets\elrslang
```

Android 工程位于 `mobile/android`。首版只提供 Vulkan smoke app 骨架和资源 contract，不承诺与 PC renderer feature parity。

## Android 构建

需要本机安装 Android SDK、NDK 和 Gradle：

```powershell
cd mobile\android
gradle :app:assembleDebug
```

当前这台机器没有检测到 `gradle`、`ANDROID_HOME` 或 `ANDROID_SDK_ROOT`，所以 Android assemble 尚未在本机验证。

## 设计边界

- SlangPy-first 不等于所有渲染都走软件模拟：全屏、compute、后处理和数据转换 pass 优先直接调用 Slang 函数；硬件 raster/DXR 仍走 SlangPy 暴露的底层 graphics/ray tracing API。
- Android 首版不嵌入 Python，也不运行 SlangPy host。
- FBX、GLB、USD 的完整材质/动画兼容依赖后续导入器完善；当前首版先建立可扩展接口和明确错误。
