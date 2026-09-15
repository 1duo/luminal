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

## Qwen3.5-0.8B Q4

The first resident model path targets the mixed-GGUF checkpoint
`unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_0.gguf`. The Qwen label is mixed:
the exported QWH9 file contains F32, Q4_0, Q4_1, Q5_K, Q6_K, and Q8_0 tensors,
plus a QHM4 HMX sidecar. This is the suitable artifact for the existing
Hexagon Q4 kernels; raw GGUF is rejected by the host before any large device
allocation.

The graph-facing API is deliberately one resident step:

```rust
use luminal::prelude::*;
use luminal_hexagon::{Qwen35Q4Config, Qwen35Runtime, qwen35_q4_step};

let mut graph = Graph::new();
let prompt = graph.tensor(8).as_dtype(DType::Int);
let token = qwen35_q4_step(prompt).output();
let mut runtime = Qwen35Runtime::try_initialize(Qwen35Q4Config::default())?;
runtime.set_data(prompt, vec![1_i32, 2, 3, 4, 5, 6, 7, 8]);
let mut runtime = graph.compile(runtime, CompileOptions::default());
let first = runtime.execute(&graph.dyn_map); // prefill
let next = runtime.execute(&graph.dyn_map);  // one decode step
```

The operation maps the QWH9 base store, QHM4 HMX sidecar, and one
`QWEN_RUNTIME_BYTES` shared buffer once. It then uses the ABI-compatible
resident `hexinfer` module: bind opcode 284, full prefill opcode 297, and one
full decode opcode 285 per generated token. The Luminal host path is therefore
zero-copy for weights, KV cache, and GatedDeltaNet state; only the small token
request/result descriptors cross the synchronous RPC boundary.

Set these variables before running the example:

```powershell
$env:LUMINAL_QWEN35_WEIGHTS = 'C:\path\to\qwen3_5_08B_Q4_0_weights.bin'
$env:LUMINAL_HEXAGON_RPC_DLL = 'C:\path\to\libcdsprpc.dll'
$env:LUMINAL_QWEN35_SKEL_URI = 'file:///libhexinfer-v73.so?hexinfer_skel_handle_invoke&_modver=1.0&_dom=cdsp'
cargo run --release -p luminal_hexagon --example qwen35
```

The current checked-in `device/` target remains the small generic Luminal
skel. Qwen execution intentionally uses the separately built and signed
resident `hexinfer` skel until its Qwen DSP translation unit (including the
HexKL HMX kernels) is vendored or selected through a CMake option. This keeps
the host ABI honest: a generic add/matmul skel cannot claim to execute a
hybrid recurrent transformer.
