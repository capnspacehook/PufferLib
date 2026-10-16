// Linked into static CUDA builds (build.sh) so a binary built against
// glibc 2.38 runs on glibc 2.35 hosts (Ubuntu 22.04). Prebuilt static archives
// (libstdc++, libgomp, nccl, raylib, cublasLt) reference symbols added after
// 2.35; definitions here win over libc.so and forward to the older ABI.
// Must not see _GNU_SOURCE, or the headers redirect strtol etc. back to
// __isoc23_* and these wrappers call themselves.
#undef _GNU_SOURCE
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/random.h>

// C23 variants only add 0b prefixes, which nothing here parses.
long __isoc23_strtol(const char* s, char** e, int b) { return strtol(s, e, b); }
long long __isoc23_strtoll(const char* s, char** e, int b) { return strtoll(s, e, b); }
unsigned long __isoc23_strtoul(const char* s, char** e, int b) { return strtoul(s, e, b); }
unsigned long long __isoc23_strtoull(const char* s, char** e, int b) { return strtoull(s, e, b); }

int __isoc23_sscanf(const char* s, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vsscanf(s, fmt, ap);
    va_end(ap);
    return r;
}

int __isoc23_fscanf(FILE* fp, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vfscanf(fp, fmt, ap);
    va_end(ap);
    return r;
}

// glibc 2.36. Used by libstdc++'s std::random_device.
uint32_t arc4random(void) {
    uint32_t v;
    while (getrandom(&v, sizeof(v), 0) != sizeof(v)) {
    }
    return v;
}

// glibc 2.38 re-versioned fmod/fmodf; bind to the original x86_64 versions.
double fmod_compat(double, double);
float fmodf_compat(float, float);
__asm__(".symver fmod_compat, fmod@GLIBC_2.2.5");
__asm__(".symver fmodf_compat, fmodf@GLIBC_2.2.5");
double fmod(double x, double y) { return fmod_compat(x, y); }
float fmodf(float x, float y) { return fmodf_compat(x, y); }
