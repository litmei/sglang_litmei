// GraphOpRegistry dispatch for the TileLang GQA rows kernel.
//
// The kernel itself is compiled by TileLang at runtime (one .so per concrete
// shape) and its `extern "C" void call(uint8_t* ...)` entry is registered here
// from Python. Replay therefore stays a pure C++ call: the common graph plans
// and owns every buffer, and this module only looks the kernel up by shape.
#include <ATen/ATen.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace {

using ScalarArg = std::variant<bool, int64_t, double, std::string>;
// rows, slots, heads, head_dim, cache_tokens, capacity, scale bits
using ShapeKey = std::array<int64_t, 7>;
// Q, K, V, Table, Requests, Lengths, O
using TilelangCall = void (*)(uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                              uint8_t *, uint8_t *, uint8_t *);

struct ShapeKeyHash {
    std::size_t operator()(const ShapeKey &key) const {
        std::size_t hash = 1469598103934665603ull;
        for (int64_t value : key) {
            hash ^= static_cast<std::size_t>(value);
            hash *= 1099511628211ull;
        }
        return hash;
    }
};

std::mutex g_kernels_mutex;
std::unordered_map<ShapeKey, TilelangCall, ShapeKeyHash> g_kernels;

int64_t scale_bits(double scale) {
    int64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(scale), "unexpected double width");
    std::memcpy(&bits, &scale, sizeof(bits));
    return bits;
}

void check(const at::Tensor &t, at::ScalarType dtype) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == dtype && t.is_contiguous(),
                "TileLang GQA rows requires contiguous CPU tensors with exact dtype");
}

int64_t read_index(const at::Tensor &t, int64_t i) {
    if (t.scalar_type() == at::kInt)
        return static_cast<int64_t>(t.data_ptr<int32_t>()[i]);
    TORCH_CHECK(t.scalar_type() == at::kLong,
                "TileLang GQA rows metadata must be int32 or int64");
    return t.data_ptr<int64_t>()[i];
}

} // namespace

extern "C" void qwen_tilelang_register_gqa_rows(
    int64_t rows, int64_t slots, int64_t heads, int64_t head_dim,
    int64_t cache_tokens, int64_t capacity, double scale, void *call) {
    TORCH_CHECK(call != nullptr, "TileLang GQA rows kernel entry is null");
    std::lock_guard<std::mutex> lock(g_kernels_mutex);
    g_kernels[ShapeKey{rows, slots, heads, head_dim, cache_tokens, capacity,
                       scale_bits(scale)}] = reinterpret_cast<TilelangCall>(call);
}

extern "C" void qwen_graph_gqa_rows_tilelang_dispatch(
    std::vector<at::Tensor> &t, const std::vector<ScalarArg> &s) {
    // Q, K, V, Table, Requests, Lengths; output.
    TORCH_CHECK(t.size() == 7 && s.size() == 1,
                "TileLang GQA rows graph dispatch ABI mismatch");
    check(t[0], at::kBFloat16);
    check(t[1], at::kBFloat16);
    check(t[2], at::kBFloat16);
    check(t[3], at::kInt);
    check(t[6], at::kBFloat16);
    TORCH_CHECK(t[4].device().is_cpu() && t[5].device().is_cpu() &&
                    t[4].is_contiguous() && t[5].is_contiguous(),
                "TileLang GQA rows metadata must be contiguous CPU tensors");
    TORCH_CHECK(t[0].dim() == 3 && t[1].dim() == 3 && t[2].dim() == 3 &&
                    t[3].dim() == 2,
                "TileLang GQA rows ranks mismatch");
    const int64_t rows = t[0].size(0);
    const int64_t heads = t[0].size(1);
    const int64_t head_dim = t[0].size(2);
    const int64_t cache_tokens = t[1].size(0);
    const int64_t slots = t[3].size(0);
    const int64_t capacity = t[3].size(1);
    TORCH_CHECK(rows >= 1 && rows <= 256 && (heads == 2 || heads == 6) &&
                    head_dim == 256,
                "TileLang GQA rows unsupported geometry");
    TORCH_CHECK(t[1].size(1) == 1 && t[1].size(2) == head_dim &&
                    t[2].sizes() == t[1].sizes(),
                "TileLang GQA rows KV layout");
    TORCH_CHECK(t[4].numel() == rows && t[5].numel() == rows,
                "TileLang GQA rows metadata length");
    TORCH_CHECK(t[6].sizes() == t[0].sizes(),
                "TileLang GQA rows output shape");
    TORCH_CHECK(slots > 0 && capacity > 0 && cache_tokens > 0,
                "TileLang GQA rows table/cache geometry");

    // Marshal the int32/int64 metadata into int64 for the kernel's signature
    // and validate the request/length ranges before the kernel reads them.
    alignas(64) int64_t requests[256];
    alignas(64) int64_t lengths[256];
    for (int64_t i = 0; i < rows; ++i) {
        requests[i] = read_index(t[4], i);
        lengths[i] = read_index(t[5], i);
        TORCH_CHECK(requests[i] >= 0 && requests[i] < slots,
                    "TileLang GQA rows request slot out of bounds");
        TORCH_CHECK(lengths[i] >= 1 && lengths[i] <= capacity,
                    "TileLang GQA rows length out of bounds");
    }

    const double scale = std::get<double>(s[0]);
    const ShapeKey key{rows, slots, heads, head_dim, cache_tokens, capacity,
                       scale_bits(scale)};
    TilelangCall call = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_kernels_mutex);
        auto it = g_kernels.find(key);
        TORCH_CHECK(it != g_kernels.end(),
                    "TileLang GQA rows kernel is not compiled for rows=", rows,
                    " slots=", slots, " heads=", heads, " head_dim=", head_dim,
                    " cache_tokens=", cache_tokens, " capacity=", capacity);
        call = it->second;
    }
    call(static_cast<uint8_t *>(t[0].data_ptr()),
         static_cast<uint8_t *>(t[1].data_ptr()),
         static_cast<uint8_t *>(t[2].data_ptr()),
         static_cast<uint8_t *>(t[3].data_ptr()),
         reinterpret_cast<uint8_t *>(requests),
         reinterpret_cast<uint8_t *>(lengths),
         static_cast<uint8_t *>(t[6].data_ptr()));
}
