# luminal_hexagon

Direct Hexagon SDK/FastRPC backend for Snapdragon X Elite.

The first slice supports contiguous `F32` elementwise `Add` and `Mul`. The
DSP target is built from `device/` with Hexagon SDK 6.4 and v73 flags:

```powershell
cmake -S crates/luminal_hexagon/device -B target/luminal-hexagon-v73 `
  -DCMAKE_TOOLCHAIN_FILE=C:/Qualcomm/Hexagon_SDK/6.4.0.2/build/cmake/hexagon_toolchain.cmake `
  -DHEXAGON_SDK_ROOT=C:/Qualcomm/Hexagon_SDK/6.4.0.2 `
  -DDSP_VERSION=v73
cmake --build target/luminal-hexagon-v73 --config Release
```

Install/catalogue the resulting skel as required by the Windows FastRPC
deployment, then set `LUMINAL_HEXAGON_RPC_DLL` and
`LUMINAL_HEXAGON_SKEL_URI` before constructing `HexagonRuntime`.
