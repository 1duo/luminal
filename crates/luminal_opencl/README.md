# luminal_opencl

OpenCL backend for Luminal, built first for the Qualcomm Adreno GPU in
Snapdragon X Elite systems.

The backend uses the system OpenCL ICD directly. On Windows ARM64 no separate
Qualcomm SDK is required: current X Elite graphics drivers provide the native
OpenCL 3.0 runtime and `cl_khr_fp16`.

```rust
use luminal::prelude::*;
use luminal_opencl::OpenClRuntime;

let mut graph = Graph::new();
let a = graph.tensor((2, 3));
let b = graph.tensor((3, 2));
let output = a.matmul(b).output();

let runtime = OpenClRuntime::try_initialize(0)?;
let mut runtime = graph.compile(runtime, CompileOptions::default());
runtime.set_data(a, vec![1.0; 6]);
runtime.set_data(b, vec![1.0; 6]);
runtime.execute(&graph.dyn_map);
println!("{:?}", runtime.get_f32(output));
# Ok::<(), String>(())
```

The runtime supports Luminal's 15 primitive operations with F32, F16, Int,
and Bool storage. Matmul is recognized by an egglog rewrite and emitted as one
kernel rather than materializing the broadcast multiply.

Device `0` is the preferred GPU. If both the Qualcomm native ICD and
Microsoft's OpenCL-on-D3D12 compatibility layer expose the same Adreno, the
native Qualcomm device is ordered first.

