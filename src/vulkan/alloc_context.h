#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include <c10/util/ArrayRef.h>

// Thread-local allocation context for OutOfMemoryError diagnostics.
//
// The c10 allocator interface (`VulkanAllocator::allocate(size_t nbytes)`)
// only sees a byte count — the shape/dtype of the tensor being created lives
// in the TensorImpl, which aten builds above the allocator. To still answer
// "which tensor blew up the GPU", the tensor-factory ops of this backend
// (empty/empty_strided/resize_, tensor/allocation.cpp) push a small
// "pending shape/dtype" hint right before the storage allocation, and
// VulkanAllocator::allocate consumes it into a per-thread ring of the last
// large allocation requests. When an allocation fails, the OutOfMemoryError
// message (vulkan/allocator.cpp) prints that ring, so the runaway tensor is
// visible with its shape and dtype (e.g. the [B, H, T, T] fp32 SDPA scores of
// an untracked 50k-token sequence).
//
// Zero overhead in the default configuration: entries below
// kMinTrackedBytes are dropped without touching the ring, and the optional
// dispatcher op-name attribution (TORCHVULKAN_ALLOC_TRACE=1, see
// alloc_context.cpp) only installs its RecordFunction hook when explicitly
// enabled via the environment.

namespace torchvulkan::alloc_context {

struct AllocRecord {
    size_t nbytes = 0;
    std::string shape; // "[1, 16, 50000, 50000]", empty when unknown
    std::string dtype; // "float32", empty when unknown
    std::string op;    // dispatcher op name, empty unless TORCHVULKAN_ALLOC_TRACE=1
};

// ring of the last large allocation requests on the calling thread,
// oldest first
std::vector<AllocRecord> snapshot();

// forget the recorded context (python: torchvulkan.clear_allocation_context())
void clear();

// RAII hint attached to the NEXT allocate() on this thread by the tensor
// factory ops; the destructor drops the hint if the allocation never
// happened (e.g. an exception or a zero-byte early return)
struct Pending {
    Pending(std::string shape, std::string dtype);
    Pending(const Pending&) = delete;
    Pending& operator=(const Pending&) = delete;
    ~Pending();
};

// called by VulkanAllocator::allocate for every non-empty request BEFORE the
// buffer is obtained: consumes the pending shape/dtype (if any) and records
// the request into the ring — so the failing allocation itself is the last
// ring entry the OOM message prints
void record(size_t nbytes);

// multi-line block for the OutOfMemoryError message ("" never — always a
// usable diagnostic, with an explicit hint when the ring is empty)
std::string describe();

// "5.0 GiB" / "312.5 MiB" / "1024 B"
std::string format_bytes(size_t nbytes);

// "[1, 16, 50000, 50000]"
std::string format_shape(c10::IntArrayRef sizes);

} // namespace torchvulkan::alloc_context
