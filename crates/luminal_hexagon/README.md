# luminal_hexagon

Direct Hexagon SDK/FastRPC backend for Snapdragon X Elite.

The backend supports contiguous `F32` elementwise `Add`/`Mul` and a signed
`I8 × I8 → I32` 2D projection. The projection keeps weights in physical
`[out_features, in_features]` order, so no host-side transpose or intermediate
copy is needed. Egglog exposes scalar and HVX schedules as equivalent choices;
when inputs are loaded before compilation, Luminal profiles those choices on
the HTP and keeps the fastest one.

The current codegen boundary emits small SDK-compatible C/HVX kernels. The
installed SDK does not ship a public HexKL compiler, so `src/codegen.rs` is
deliberately isolated and can be replaced by a HexKL emitter later without
changing the dialect, search, or FastRPC ABI.

The DSP target is built from `device/` with Hexagon SDK 6.4 and v73 flags:

```powershell
cmake -S crates/luminal_hexagon/device `
  -B target/luminal-hexagon-v73 -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE=C:/Qualcomm/Hexagon_SDK/6.4.0.2/build/cmake/hexagon_toolchain.cmake `
  -DHEXAGON_SDK_ROOT=C:/Qualcomm/Hexagon_SDK/6.4.0.2 `
  -DHEXAGON_TOOLS_ROOT=C:/Qualcomm/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04 `
  -DPREBUILT_LIB_DIR=hexagon_toolv19_v73 `
  -DDSP_VERSION=v73
cmake --build target/luminal-hexagon-v73 --config Release
```

Install/catalogue the resulting skel as required by the Windows FastRPC
deployment, then set `LUMINAL_HEXAGON_RPC_DLL` and
`LUMINAL_HEXAGON_SKEL_URI` before constructing `HexagonRuntime`.

The smallest device example is the integer projection:

```powershell
cargo run --release -p luminal_hexagon --example integer
```

It validates the I32 logits against a CPU reference and reports token/s and
GMAC/s. It requires the FastRPC DLL and signed/catalogued skel to be visible
to the host process.
