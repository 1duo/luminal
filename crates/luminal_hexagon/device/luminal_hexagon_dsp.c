/* Keep this checked-in source identical in behavior to
 * luminal_hexagon::codegen::emit_dsp_source([AddF32, MulF32]).  The Rust
 * generator is used by graph tooling; this static form makes the SDK CMake
 * target usable without requiring Cargo on the Hexagon build host. */
#include <AEEStdDef.h>
#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "luminal_hexagon.h"

enum {
    LUMINAL_HEXAGON_ADD_F32 = 0,
    LUMINAL_HEXAGON_MUL_F32 = 1,
};

static inline void luminal_copy_tail(float *out, const float *a, const float *b,
                                     uint32_t begin, uint32_t n, int multiply) {
    for (uint32_t i = begin; i < n; ++i) {
        out[i] = multiply ? a[i] * b[i] : a[i] + b[i];
    }
}

static void luminal_add_f32(const float *a, const float *b, float *out, uint32_t n) {
    const uint32_t vectors = n & ~31u;
    for (uint32_t i = 0; i < vectors; i += 32) {
        const HVX_Vector va = *(const HVX_UVector *)(a + i);
        const HVX_Vector vb = *(const HVX_UVector *)(b + i);
        const HVX_Vector vc = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(va, vb));
        *(HVX_UVector *)(out + i) = vc;
    }
    luminal_copy_tail(out, a, b, vectors, n, 0);
}

static void luminal_mul_f32(const float *a, const float *b, float *out, uint32_t n) {
    const uint32_t vectors = n & ~31u;
    for (uint32_t i = 0; i < vectors; i += 32) {
        const HVX_Vector va = *(const HVX_UVector *)(a + i);
        const HVX_Vector vb = *(const HVX_UVector *)(b + i);
        const HVX_Vector vc = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(va, vb));
        *(HVX_UVector *)(out + i) = vc;
    }
    luminal_copy_tail(out, a, b, vectors, n, 1);
}

AEEResult luminal_hexagon_open(const char *uri, remote_handle64 *handle) {
    (void)uri;
    *handle = 1;
    return AEE_SUCCESS;
}

AEEResult luminal_hexagon_close(remote_handle64 handle) {
    (void)handle;
    return AEE_SUCCESS;
}

AEEResult luminal_hexagon_compute(remote_handle64 handle, uint32_t op, uint32_t n,
                                  const unsigned char *a, int a_len,
                                  const unsigned char *b, int b_len,
                                  unsigned char *out, int out_len) {
    (void)handle;
    if (!a || !b || !out || a_len < 0 || b_len < 0 || out_len < 0 ||
        (uint64_t)n * sizeof(float) > (uint64_t)a_len ||
        (uint64_t)n * sizeof(float) > (uint64_t)b_len ||
        (uint64_t)n * sizeof(float) > (uint64_t)out_len) {
        return AEE_EBADPARM;
    }
    switch (op) {
        case LUMINAL_HEXAGON_ADD_F32:
            luminal_add_f32((const float *)a, (const float *)b, (float *)out, n);
            break;
        case LUMINAL_HEXAGON_MUL_F32:
            luminal_mul_f32((const float *)a, (const float *)b, (float *)out, n);
            break;
        default:
            return AEE_EUNSUPPORTED;
    }
    return AEE_SUCCESS;
}
