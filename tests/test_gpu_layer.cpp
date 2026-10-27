/* Unit test of the vendor wrapper (src/gpu.c).  Built only for GPU builds. */
#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include "gpu.h"

TEST(GpuLayer, BackendIsReported) {
    EXPECT_TRUE(gpu_available());
    EXPECT_STRNE(gpu_backend_name(), "none");
    ASSERT_GT(gpu_device_count(), 0);
    gpu_set_device(0);
}

TEST(GpuLayer, HostDeviceRoundTrip) {
    gpu_set_device(0);
    std::vector<unsigned char> in(4096), out(4096, 0);
    for (size_t i = 0; i < in.size(); i++) in[i] = (unsigned char)(i * 7);
    void *d = gpu_malloc(in.size());
    gpu_memcpy_h2d(d, in.data(), in.size());
    gpu_memcpy_d2h(out.data(), d, out.size());
    EXPECT_EQ(in, out);
    gpu_free(d);
}

TEST(GpuLayer, MemsetAndDeviceToDevice) {
    gpu_set_device(0);
    void *a = gpu_malloc(256), *b = gpu_malloc(256);
    gpu_memset(a, 0x5a, 256);
    gpu_memcpy_d2d(b, a, 256);
    gpu_sync();
    std::vector<unsigned char> out(256, 0);
    gpu_memcpy_d2h(out.data(), b, 256);
    for (unsigned char c : out) EXPECT_EQ(c, 0x5a);
    gpu_free(a); gpu_free(b);
}

TEST(GpuLayer, ZeroByteAllocationIsUsable) {
    gpu_set_device(0);
    void *p = gpu_malloc(0);
    EXPECT_NE(p, nullptr);
    gpu_free(p);
}

/* PCI id: how gpu_setup tells ranks that really share a device from ranks bound to different ones. */
TEST(GpuLayer, PciIdIsReported) {
    char id[32] = {0};
    ASSERT_EQ(gpu_device_pci_id(0, id, sizeof id), 0);
    EXPECT_GT(strlen(id), 0u);
}
