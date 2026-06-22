# Falcor 场景样例

这些文件从 `G:\J_Pan\Code\Mine\FalcorELR` 复制而来，用于 ELRSlang 的场景导入和渲染 smoke。

## 可直接尝试加载

- `meshes/bunny.obj`
- `meshes/teapot.obj`
- `tex_lod/floor.obj`
- `tex_lod/room.obj`
- `cesium_man/CesiumMan.gltf`
- `glb/robot_01.glb`
- `curves/one_curve.usda`
- `curves/two_curves.usda`

示例：

```powershell
.\.venv\Scripts\python -m elrslang.viewer --frames 1 --scene assets\scenes\falcor\meshes\bunny.obj --graph raster_forward --backend automatic
```

## Falcor 原生参考

`falcor_pyscene/` 和部分子目录里的 `.pyscene` 是 Falcor 自己的脚本化场景描述。ELRSlang 首版不会直接执行 `.pyscene`，但这些文件可用于参考 Falcor 的相机、材质和灯光组织方式。

`animated_cubes/animated_cubes.fbx` 也已经复制过来，但当前 Python 依赖里的 `trimesh` 不支持 FBX 解码。它保留给后续接入 Assimp/FBX importer 后验证动画和网格导入。

## 未复制的大场景

没有复制 Bistro、Lucy、dense bunny、材质 binary 数据等大体量资源。它们更适合后续导入器成熟后再按需引入。
