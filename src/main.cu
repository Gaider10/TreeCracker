#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cinttypes>
#include <chrono>
#include <vector>

#include "Random.cuh"
#include "trees.cuh"
#include "input_data.cuh"

#define PANIC(...) { \
    std::fprintf(stderr, __VA_ARGS__); \
    std::abort(); \
}

#define TRY_CUDA(expr) try_cuda(expr, __FILE__, __LINE__)

void try_cuda(cudaError_t error, const char *file, uint64_t line) {
    if (error == cudaSuccess) return;

    PANIC("CUDA error at %s:%" PRIu64 ": %s\n", file, line, cudaGetErrorString(error));
}

constexpr uint64_t threads_per_block = 256;

constexpr uint64_t max_results_1_len = 16 * 1024 * 1024;
__device__ uint64_t results_1[max_results_1_len];
__device__ uint64_t results_1_len;

constexpr uint64_t max_results_2_len = 1024 * 1024;
__device__ uint64_t results_2[max_results_2_len];
__device__ uint64_t results_2_len;

constexpr uint64_t max_results_3_len = 1024 * 1024;
__device__ uint64_t results_3[max_results_3_len];
__device__ uint64_t results_3_len;
__device__ uint32_t results_3_mask[max_results_3_len / 32];

constexpr uint64_t max_results_4_len = 1024 * 1024;
__device__ uint64_t results_4[max_results_4_len];
__device__ uint64_t results_4_len;

__device__ constexpr TreeChunk input_data = get_input_data();
constexpr int32_t max_calls = biome_max_calls(input_data.version, input_data.biome);
constexpr int32_t max_tree_count = biome_max_tree_count(input_data.version, input_data.biome);
constexpr bool collapse_nearby_seeds = input_data.version <= Version::v1_12_2;

constexpr uint64_t floor_pow2(uint64_t v) {
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v & ~(v >> 1);
}

constexpr uint64_t ceil_div(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

constexpr int64_t constexpr_floor(double v) {
    int64_t i = v;
    return i < v ? i : i - 1;
}

constexpr int64_t constexpr_ceil(double v) {
    int64_t i = v;
    return i > v ? i : i + 1;
}

constexpr int64_t constexpr_round(double v) {
    return constexpr_floor(v + 0.5);
}

constexpr auto constexpr_abs(auto a) {
    return a < 0 ? -a : a;
}

namespace LatticeData {
    // LLL[{{0,2^48},{1,25214903917}}] transposed
    constexpr int64_t basis[2][2] = {
        { 7847617, -18218081 },
        { 4824621, 24667315 },
    };
    constexpr int64_t basis_det = basis[0][0] * basis[1][1] - basis[0][1] * basis[1][0];

    constexpr double basis_inv[2][2] = {
        { (double) basis[1][1] / basis_det, (double)-basis[0][1] / basis_det },
        { (double)-basis[1][0] / basis_det, (double) basis[0][0] / basis_det },
    };

    constexpr double Si = basis_inv[0][0];
    constexpr double l = basis_inv[1][0] / Si;
    constexpr double Sj = basis_inv[1][1] - basis_inv[0][1] * l;
    constexpr double k = basis_inv[0][1] / Sj;

    constexpr int32_t kl_bits = 32;
    constexpr double max_i = Si * (UINT64_C(1) << 48);
    constexpr double max_j = Sj * (UINT64_C(1) << 48);
    constexpr double max_jk = max_j * constexpr_abs(k);
    constexpr double max_il = (max_i + max_jk) * constexpr_abs(l);
    static_assert(max_jk < (UINT64_C(1) << 62 - kl_bits), "Multiplying by k could overflow");
    static_assert(max_il < (UINT64_C(1) << 62 - kl_bits), "Multiplying by l could overflow");
    constexpr int64_t k_int = constexpr_round(k * (UINT64_C(1) << kl_bits));
    constexpr int64_t l_int = constexpr_round(l * (UINT64_C(1) << kl_bits));
    // 2^44 Rab
    constexpr int64_t Ri = constexpr_ceil(Si * (UINT64_C(1) << 44));
    constexpr int64_t Rj = constexpr_ceil(Sj * (UINT64_C(1) << 44));
    constexpr int32_t Mi = 2;
    constexpr int32_t Mj = 2;
    constexpr int64_t Ci = Ri + 1 + Mi * 2;
    constexpr int64_t Cj = Rj + 1 + Mj * 2;

    constexpr uint64_t max_threads_per_run = UINT64_C(1) << 32;
    constexpr uint64_t Cj_per_run = floor_pow2(max_threads_per_run / Ci);
    constexpr uint64_t threads_per_run = Ci * Cj_per_run;

    constexpr uint64_t total_threads = Ci * Cj;
    constexpr uint64_t total_runs = ceil_div(total_threads, threads_per_run);
};

__global__ __launch_bounds__(threads_per_block) void filter_1(int32_t i0, int32_t j0) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t i = i0 + index / LatticeData::Cj_per_run;
    int32_t j = j0 + index % LatticeData::Cj_per_run;
    i += j * LatticeData::k_int >> LatticeData::kl_bits;
    j += i * LatticeData::l_int >> LatticeData::kl_bits;
    uint64_t seed = (LatticeData::basis[0][0] * i + LatticeData::basis[0][1] * j) & LCG::MASK;

    Random tree_random = Random::withSeed(seed);
    if (!input_data.trees[0].test_full<0>(input_data.version, input_data.biome, tree_random)) return;

    uint64_t result_index = atomicAdd(reinterpret_cast<unsigned long long*>(&results_1_len), 1);
    if (result_index >= max_results_1_len) return;
    results_1[result_index] = seed;
}

__global__ __launch_bounds__(threads_per_block) void filter_2() {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= results_1_len) return;
    uint64_t seed = results_1[index];

    uint32_t found = 0;
    Random random = Random::withSeed(seed).skip<-max_calls>();
    for (int32_t i = -max_calls; i <= max_calls; i++) {
        #pragma unroll
        for (int32_t j = 0; j < input_data.trees_len; j++) {
            const Tree &tree = input_data.trees[j];

            Random tree_random(random);
            if (tree.test_pos_type(input_data.version, input_data.biome, tree_random)) {
                found |= 1 << j;
            }
        }

        random.skip<1>();
    }

    if (__popc(found) != input_data.trees_len) return;

    uint64_t result_index = atomicAdd(reinterpret_cast<unsigned long long*>(&results_2_len), 1);
    if (result_index >= max_results_2_len) return;
    results_2[result_index] = seed;
}

__global__ __launch_bounds__(threads_per_block) void filter_3() {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= results_2_len) return;
    uint64_t seed = results_2[index];

    uint32_t found = 0;
    Random random = Random::withSeed(seed).skip<-max_calls>();
    for (int32_t i = -max_calls; i <= max_calls; i++) {
        #pragma unroll
        for (int32_t j = 0; j < input_data.trees_len; j++) {
            const Tree &tree = input_data.trees[j];

            Random tree_random(random);
            if (tree.test_full(input_data.version, input_data.biome, tree_random)) {
                found |= 1 << j;
            }
        }

        random.skip<1>();
    }

    if (__popc(found) != input_data.trees_len) return;

    uint64_t result_index = atomicAdd(reinterpret_cast<unsigned long long*>(&results_3_len), 1);
    if (result_index >= max_results_3_len) return;
    results_3[result_index] = seed;
}

__global__ __launch_bounds__(threads_per_block) void filter_4() {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t calls = index % (max_calls + 1);
    index /= (max_calls + 1);

    uint32_t generated_mask = index % (1 << max_tree_count);
    index /= 1 << max_tree_count;

    if (__popc(generated_mask) < input_data.trees_len) return;

    if (index >= results_3_len) return;
    uint64_t seed = results_3[index];

    Random random = Random::withSeed(seed);
    if (calls & 256) random.skip<-256>();
    if (calls & 128) random.skip<-128>();
    if (calls & 64) random.skip<-64>();
    if (calls & 32) random.skip<-32>();
    if (calls & 16) random.skip<-16>();
    if (calls & 8) random.skip<-8>();
    if (calls & 4) random.skip<-4>();
    if (calls & 2) random.skip<-2>();
    if (calls & 1) random.skip<-1>();
    seed = random.seed;

    uint32_t tree_count = biome_tree_count(input_data.version, input_data.biome, random);
    if ((generated_mask >> tree_count) != 0) return;

    uint32_t found = 0;

    #pragma unroll
    for (int32_t i = 0; i < tree_count; i++) {
        bool generated = (generated_mask >> i) & 1;

        if (generated) {
            #pragma unroll
            for (int32_t j = 0; j < input_data.trees_len; j++) {
                const Tree &tree = input_data.trees[j];

                Random tree_random(random);
                if (tree.test_full(input_data.version, input_data.biome, tree_random)) {
                    found |= 1 << j;
                }
            }
        }

        Tree::skip(input_data.version, input_data.biome, random, generated);
    }

    if (__popc(found) != input_data.trees_len) return;

    if (collapse_nearby_seeds) {
        uint32_t seed_mask = UINT32_C(1) << (index & 31);
        if (atomicOr(&results_3_mask[index / 32], seed_mask) & seed_mask) return;
    }

    uint64_t result_index = atomicAdd(reinterpret_cast<unsigned long long*>(&results_4_len), 1);
    if (result_index >= max_results_4_len) return;
    results_4[result_index] = seed;
}

int main() {
    uint64_t host_results_1_len = 0;
    uint64_t host_results_2_len = 0;
    uint64_t host_results_3_len = 0;
    uint64_t host_results_4_len = 0;
    void *device_results_1_len;
    void *device_results_2_len;
    void *device_results_3_len;
    void *device_results_4_len;
    void *device_results_4;
    TRY_CUDA(cudaGetSymbolAddress(&device_results_1_len, results_1_len));
    TRY_CUDA(cudaGetSymbolAddress(&device_results_2_len, results_2_len));
    TRY_CUDA(cudaGetSymbolAddress(&device_results_3_len, results_3_len));
    TRY_CUDA(cudaGetSymbolAddress(&device_results_4_len, results_4_len));
    TRY_CUDA(cudaGetSymbolAddress(&device_results_4, results_4));
    TRY_CUDA(cudaMemsetAsync(device_results_1_len, 0, sizeof(results_1_len)));
    TRY_CUDA(cudaMemsetAsync(device_results_2_len, 0, sizeof(results_2_len)));
    TRY_CUDA(cudaMemsetAsync(device_results_3_len, 0, sizeof(results_3_len)));
    TRY_CUDA(cudaMemsetAsync(device_results_4_len, 0, sizeof(results_4_len)));
    std::vector<uint64_t> host_results_4(max_results_4_len);

    FILE *output_file = std::fopen("output.txt", "w");
    if (output_file == NULL) PANIC("Could not open output.txt\n");

    uint64_t parts = 1;
    uint64_t part = 0;

    uint64_t part_run_start = part * LatticeData::total_runs / parts;
    uint64_t part_run_end = (part + 1) * LatticeData::total_runs / parts;

    void *results_3_mask_ptr;
    TRY_CUDA(cudaGetSymbolAddress(&results_3_mask_ptr, results_3_mask));

    constexpr int32_t i0 = (int32_t)(((uint64_t)input_data.trees[0].x << 44) * LatticeData::Si + 0.5) - LatticeData::Mi;
    constexpr int32_t j0 = (int32_t)(((uint64_t)input_data.trees[0].z << 44) * LatticeData::Sj + 0.5) - LatticeData::Mj;

    constexpr uint64_t print_interval = 64;
    auto global_start_time = std::chrono::steady_clock::now();
    auto start_time = global_start_time;

    uint64_t total_filter_1_results = 0;
    uint64_t total_filter_2_results = 0;
    uint64_t total_filter_3_results = 0;
    uint64_t total_filter_4_results = 0;
    double total_filter_1_time = 0;
    double total_filter_2_time = 0;
    double total_filter_3_time = 0;
    double total_filter_4_time = 0;

    for (uint64_t run = part_run_start; run < part_run_end; run++) {
        if (collapse_nearby_seeds) {
            TRY_CUDA(cudaMemsetAsync(results_3_mask_ptr, 0, sizeof(results_3_mask)));
        }

        auto run_start_time = std::chrono::steady_clock::now();

        filter_1<<<ceil_div(LatticeData::threads_per_run, threads_per_block), threads_per_block>>>(i0, j0 + (int32_t)(run * LatticeData::Cj_per_run));
        TRY_CUDA(cudaGetLastError());
        TRY_CUDA(cudaMemcpyAsync(&host_results_1_len, device_results_1_len, sizeof(results_1_len), cudaMemcpyDeviceToHost));
        TRY_CUDA(cudaDeviceSynchronize());
        auto filter_1_time = std::chrono::steady_clock::now();

        if (host_results_1_len > max_results_1_len) {
            std::fprintf(stderr, "results_1_len > max_results_1_len, ignored %" PRIu64 "\n", host_results_1_len - max_results_1_len);
            host_results_1_len = max_results_1_len;
        }

        filter_2<<<host_results_1_len / threads_per_block + 1, threads_per_block>>>();
        TRY_CUDA(cudaGetLastError());
        TRY_CUDA(cudaMemcpyAsync(&host_results_2_len, device_results_2_len, sizeof(results_2_len), cudaMemcpyDeviceToHost));
        TRY_CUDA(cudaDeviceSynchronize());
        auto filter_2_time = std::chrono::steady_clock::now();

        if (host_results_2_len > max_results_2_len) {
            std::fprintf(stderr, "results_2_len > max_results_2_len, ignored %" PRIu64 "\n", host_results_2_len - max_results_2_len);
            host_results_2_len = max_results_2_len;
        }

        filter_3<<<host_results_2_len / threads_per_block + 1, threads_per_block>>>();
        TRY_CUDA(cudaGetLastError());
        TRY_CUDA(cudaMemcpyAsync(&host_results_3_len, device_results_3_len, sizeof(results_3_len), cudaMemcpyDeviceToHost));
        TRY_CUDA(cudaDeviceSynchronize());
        auto filter_3_time = std::chrono::steady_clock::now();

        if (host_results_3_len > max_results_3_len) {
            std::fprintf(stderr, "results_3_len > max_results_3_len, ignored %" PRIu64 "\n", host_results_3_len - max_results_3_len);
            host_results_3_len = max_results_3_len;
        }

        filter_4<<<host_results_3_len * (max_calls + 1) * (1 << max_tree_count) / threads_per_block + 1, threads_per_block>>>();
        TRY_CUDA(cudaGetLastError());
        TRY_CUDA(cudaMemcpyAsync(&host_results_4_len, device_results_4_len, sizeof(results_4_len), cudaMemcpyDeviceToHost));
        TRY_CUDA(cudaDeviceSynchronize());
        auto filter_4_time = std::chrono::steady_clock::now();

        if (host_results_4_len > max_results_4_len) {
            std::fprintf(stderr, "results_4_len > max_results_4_len, ignored %" PRIu64 "\n", host_results_4_len - max_results_4_len);
            host_results_4_len = max_results_4_len;
        }

        if (run == part_run_start) {
            std::fprintf(stderr, "Counts: %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", host_results_1_len, host_results_2_len, host_results_3_len, host_results_4_len);
        }
        total_filter_1_results += host_results_1_len;
        total_filter_2_results += host_results_2_len;
        total_filter_3_results += host_results_3_len;
        total_filter_4_results += host_results_4_len;
        total_filter_1_time += std::chrono::duration_cast<std::chrono::nanoseconds>(filter_1_time - run_start_time).count() * 1e-9;
        total_filter_2_time += std::chrono::duration_cast<std::chrono::nanoseconds>(filter_2_time - filter_1_time).count() * 1e-9;
        total_filter_3_time += std::chrono::duration_cast<std::chrono::nanoseconds>(filter_3_time - filter_2_time).count() * 1e-9;
        total_filter_4_time += std::chrono::duration_cast<std::chrono::nanoseconds>(filter_4_time - filter_3_time).count() * 1e-9;
        // for (uint64_t i = 0; i < results_3_len; i++) {
        //     std::printf("3: %" PRIu64 "\n", results_3[i]);
        // }
        TRY_CUDA(cudaMemcpy(host_results_4.data(), device_results_4, sizeof(*results_4) * host_results_4_len, cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < host_results_4_len; i++) {
            std::printf("%" PRIu64 "\n", host_results_4[i]);
            std::fprintf(output_file, "%" PRIu64 "\n", host_results_4[i]);
        }

        TRY_CUDA(cudaMemsetAsync(device_results_1_len, 0, sizeof(results_1_len)));
        TRY_CUDA(cudaMemsetAsync(device_results_2_len, 0, sizeof(results_2_len)));
        TRY_CUDA(cudaMemsetAsync(device_results_3_len, 0, sizeof(results_3_len)));
        TRY_CUDA(cudaMemsetAsync(device_results_4_len, 0, sizeof(results_4_len)));

        if ((run + 1 - part_run_start) % print_interval == 0) {
            auto end_time = std::chrono::steady_clock::now();
            double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / 1000000000.0;
            double rps = print_interval / time;
            double eta = (part_run_end - (run + 1)) / rps;
            double sps = rps * LatticeData::threads_per_run;

            std::fprintf(stderr, "%.3f s, %.3f Gsps, ETA: %f s\n", time, sps / 1000000000.0, eta);

            start_time = end_time;
        }
	}

    auto global_end_time = std::chrono::steady_clock::now();
    double global_time = std::chrono::duration_cast<std::chrono::nanoseconds>(global_end_time - global_start_time).count() / 1000000000.0;
    std::fprintf(stderr, "Finished in %.3f s\n", global_time);

    std::fprintf(stderr, "Filter Results: %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", total_filter_1_results, total_filter_2_results, total_filter_3_results, total_filter_4_results);
    std::fprintf(stderr, "Filter Times: %.3f s / %.3f s / %.3f s / %.3f s\n", total_filter_1_time, total_filter_2_time, total_filter_3_time, total_filter_4_time);

    std::fclose(output_file);

	return 0;
}