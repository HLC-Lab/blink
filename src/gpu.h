#ifndef BLINK_GPU_H
#define BLINK_GPU_H

/* Vendor wrapper (CUDA / HIP): allocate, free, copy, memset.  Implemented in
 * gpu.c, the only file that includes the vendor runtime headers.             */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int         gpu_available(void);                          /* 1 if built with CUDA or HIP        */
const char *gpu_backend_name(void);                       /* "CUDA" | "HIP" | "none"            */
int         gpu_device_count(void);                       /* 0 if none visible / not built      */
void        gpu_set_device(int dev);
int         gpu_device_pci_id(int dev, char *buf, int len); /* 0 on success; identifies the physical device */
void       *gpu_malloc(size_t n);                         /* never NULL, fatal on failure; n==0 -> 1 byte */
void        gpu_free(void *p);
void        gpu_memset(void *dst, int value, size_t n);
void        gpu_memcpy_h2d(void *dst, const void *src, size_t n);
void        gpu_memcpy_d2h(void *dst, const void *src, size_t n);
void        gpu_memcpy_d2d(void *dst, const void *src, size_t n);
void        gpu_sync(void);

#ifdef __cplusplus
}
#endif

#endif /* BLINK_GPU_H */
