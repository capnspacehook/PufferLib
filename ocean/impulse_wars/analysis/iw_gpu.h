// GPU inference for the analysis harness: the trainer's CUDA Impulse Wars
// net (src/algo.cu + impulse_wars.cu) behind a C API, so iw_analyze.c can
// run the policy forward on the GPU while the envs step on the CPU.
//
// A model is one checkpoint's weights on the GPU, shared by every thread.
// A net is a forward pass over a fixed batch with its own recurrent state,
// owned by one worker thread: create, run and free it from that thread, as
// each thread has its own cuBLAS handle and stream, and the forward is a
// CUDA graph captured on the first run. The obs, terminals and logits
// buffers are pinned host memory the harness reads and writes directly.
#ifndef IW_GPU_H
#define IW_GPU_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iwGpuModel iwGpuModel;
typedef struct iwGpuNet iwGpuNet;

// the encoder inputs of the last forward, for --weights-out, laid out as
// the CPU net's (ImpulseWarsNet) concat and record buffers
typedef struct iwGpuInputs {
    const float *concat;
    const float *fwRec;
    const int *fwActive;
} iwGpuInputs;

// "fp32" or "bf16", as the library was built
const char *iw_gpu_precision(void);
// sets up the device, returns false with the reason printed if there's no
// usable GPU
bool iw_gpu_init(int device);
// weights is the checkpoint as load_weights reads it, numFloats long
iwGpuModel *iw_gpu_load(const float *weights, long numFloats, int hidden, int layers);
// inputs: also copy the encoder inputs back after every forward
iwGpuNet *iw_gpu_net(iwGpuModel *model, int batch, bool inputs);
float *iw_gpu_obs(iwGpuNet *net);
float *iw_gpu_terminals(iwGpuNet *net);
// (batch, atn_sum + 1): every action head's logits, then the value
const float *iw_gpu_logits(iwGpuNet *net);
// starts a forward of the obs and terminals buffers; the rows of terminal
// games start from a zero recurrent state, like mingru_zero_term
void iw_gpu_launch(iwGpuNet *net);
// waits for every forward launched on this thread
void iw_gpu_wait(void);
iwGpuInputs iw_gpu_inputs(iwGpuNet *net);
void iw_gpu_free_net(iwGpuNet *net);

#ifdef __cplusplus
}
#endif

#endif
