#include "alloc_context.h"

#include <torch/extension.h>
#include <ATen/record_function.h>
#include <c10/core/ScalarType.h>

#include <cstdio>
#include <cstdlib>

namespace torchvulkan {
namespace alloc_context {

namespace {

// diagnostics-only ring: the last N allocation requests of at least
// kMinTrackedBytes per thread (older entries evicted first). Small
// allocations are irrelevant next to a 4 GiB VRAM budget and would flood
// the ring between two layer-sized requests.
constexpr size_t kRingCapacity = 32;
constexpr size_t kMinTrackedBytes = 512u << 10;

struct Ring {
    std::vector<AllocRecord> items;
    size_t next = 0;
    bool full = false;

    void push(AllocRecord&& record)
    {
        if (!full)
        {
            items.push_back(std::move(record));
            if (items.size() == kRingCapacity)
            {
                full = true;
                next = 0;
            }
            return;
        }
        items[next] = std::move(record);
        next = (next + 1) % kRingCapacity;
    }

    std::vector<AllocRecord> snapshot() const
    {
        if (!full) return items;
        std::vector<AllocRecord> out;
        out.reserve(kRingCapacity);
        for (size_t i = 0; i < kRingCapacity; ++i) out.push_back(items[(next + i) % kRingCapacity]);
        return out;
    }
};

Ring& ring() { thread_local Ring r; return r; }
std::string& pending_shape() { thread_local std::string s; return s; }
std::string& pending_dtype() { thread_local std::string s; return s; }

// stack of the dispatcher ops currently running on this thread; the
// attribution of an allocation is the innermost enclosing op that is not one
// of the tensor factories themselves (an aten::empty inside aten::mm is
// attributed to aten::mm, an aten::empty called by the user to itself)
std::vector<std::string>& op_stack() { thread_local std::vector<std::string> s; return s; }

bool is_factory_name(const std::string& name)
{
    return name == "aten::empty.memory_format" || name == "aten::empty_strided" || name == "aten::resize_";
}

std::string current_op()
{
    const std::vector<std::string>& stack = op_stack();
    for (auto it = stack.rbegin(); it != stack.rend(); ++it)
        if (!is_factory_name(*it)) return *it;
    return stack.empty() ? std::string() : stack.front();
}

// op-name attribution costs a RecordFunction scope per dispatcher op, so it
// is strictly opt-in
bool trace_enabled()
{
    static const bool enabled = []() {
        const char* value = std::getenv("TORCHVULKAN_ALLOC_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void ensure_op_name_hook()
{
    static const bool installed = []() {
        at::addGlobalCallback(at::RecordFunctionCallback(
            [](const at::RecordFunction& rf) -> std::unique_ptr<at::ObserverContext> {
                std::string name = rf.name() != nullptr ? rf.name() : "";
                const char* overload = rf.overload_name();
                if (overload != nullptr && overload[0] != '\0') (name += ".") += overload;
                if (name.size() > 128) name.resize(128);
                op_stack().push_back(std::move(name));
                return nullptr;
            },
            [](const at::RecordFunction&, at::ObserverContext*) {
                if (!op_stack().empty()) op_stack().pop_back();
            }).needsInputs(false));
        return true;
    }();
    (void)installed;
}

void ensure_trace_on_this_thread()
{
    if (!trace_enabled()) return;
    ensure_op_name_hook();
    if (!at::isRecordFunctionEnabled()) at::enableRecordFunction(true);
}

} // namespace

Pending::Pending(std::string shape, std::string dtype)
{
    ensure_trace_on_this_thread();
    pending_shape() = std::move(shape);
    pending_dtype() = std::move(dtype);
}

Pending::~Pending()
{
    pending_shape().clear();
    pending_dtype().clear();
}

void record(size_t nbytes)
{
    // the pending hint belongs to exactly this allocation — drop it whether
    // or not the request itself is worth a ring entry
    std::string shape = std::move(pending_shape());
    std::string dtype = std::move(pending_dtype());
    pending_shape().clear();
    pending_dtype().clear();

    if (nbytes < kMinTrackedBytes) return;

    AllocRecord rec;
    rec.nbytes = nbytes;
    rec.shape = std::move(shape);
    rec.dtype = std::move(dtype);
    rec.op = current_op();

    if (trace_enabled())
        fprintf(stderr, "[torchvulkan][alloc] %s  shape=%s  dtype=%s  op=%s\n", format_bytes(nbytes).c_str(),
            rec.shape.c_str(), rec.dtype.c_str(), rec.op.c_str());

    ring().push(std::move(rec));
}

std::vector<AllocRecord> snapshot() { return ring().snapshot(); }

void clear()
{
    ring() = Ring{};
    pending_shape().clear();
    pending_dtype().clear();
}

std::string describe()
{
    std::vector<AllocRecord> items = ring().snapshot();
    if (items.empty())
        return "\nRecent large allocations: none recorded on this thread (the request bypassed the torchvulkan "
               "tensor factories — a raw aten/native allocation — or every recent allocation was small).";
    std::string out =
        "\nRecent large allocations on this thread (oldest first; the last one is the request that failed):\n";
    char index[16];
    for (size_t i = 0; i < items.size(); ++i)
    {
        snprintf(index, sizeof(index), "%3zu. ", i + 1);
        out += index;
        out += format_bytes(items[i].nbytes);
        if (!items[i].dtype.empty()) (out += "  ") += items[i].dtype;
        if (!items[i].shape.empty()) (out += "  ") += items[i].shape;
        if (!items[i].op.empty()) (out += "  via ") += items[i].op;
        out += "\n";
    }
    return out;
}

std::string format_bytes(size_t nbytes)
{
    char buf[32];
    if (nbytes >= (1ull << 30)) snprintf(buf, sizeof(buf), "%.1f GiB", static_cast<double>(nbytes) / static_cast<double>(1ull << 30));
    else if (nbytes >= (1ull << 20)) snprintf(buf, sizeof(buf), "%.1f MiB", static_cast<double>(nbytes) / static_cast<double>(1ull << 20));
    else if (nbytes >= (1ull << 10)) snprintf(buf, sizeof(buf), "%.1f KiB", static_cast<double>(nbytes) / static_cast<double>(1ull << 10));
    else snprintf(buf, sizeof(buf), "%zu B", nbytes);
    return buf;
}

std::string format_shape(c10::IntArrayRef sizes)
{
    std::string out = "[";
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        if (i > 0) out += ", ";
        out += std::to_string(sizes[i]);
    }
    out += "]";
    return out;
}

} // namespace alloc_context
} // namespace torchvulkan
