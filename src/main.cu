#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cinttypes>
#include <chrono>

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

constexpr uint64_t threads_per_run = UINT64_C(4) * 1024 * 1024 * 1024;
constexpr uint64_t threads_per_block = 256;
constexpr uint64_t blocks_per_run = threads_per_run / threads_per_block;

constexpr uint64_t max_results_1_len = 8 * 1024 * 1024;
__device__ uint64_t results_1[max_results_1_len];
__managed__ uint64_t results_1_len;

constexpr uint64_t max_results_2_len = 1024 * 1024;
__device__ uint64_t results_2[max_results_2_len];
__managed__ uint64_t results_2_len;

constexpr uint64_t max_results_3_len = 1024 * 1024;
__managed__ uint64_t results_3[max_results_3_len];
__managed__ uint64_t results_3_len;
__device__ uint32_t results_3_mask[max_results_3_len / 32];

constexpr uint64_t max_results_4_len = 1024 * 1024;
__managed__ uint64_t results_4[max_results_4_len];
__managed__ uint64_t results_4_len;

__device__ constexpr TreeChunk input_data = get_input_data();
constexpr int32_t max_calls = biome_max_calls(input_data.version, input_data.biome);
constexpr int32_t max_tree_count = biome_max_tree_count(input_data.version, input_data.biome);
constexpr bool collapse_nearby_seeds = input_data.version <= Version::v1_12_2;

__global__ __launch_bounds__(threads_per_block) void filter_1(uint64_t start) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t seed = (static_cast<uint64_t>(input_data.trees[0].x) << (48 - 4)) + start + index;

    Random tree_random = Random::withSeed(seed);
    if (!input_data.trees[0].test_full_shortcut(input_data.version, input_data.biome, tree_random)) return;

    uint64_t result_index = atomicAdd(reinterpret_cast<unsigned long long*>(&results_1_len), 1);
    if (result_index > max_results_1_len) return;
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
    if (result_index > max_results_2_len) return;
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
    if (result_index > max_results_3_len) return;
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
    if (result_index > max_results_4_len) return;
    results_4[result_index] = seed;
}

int main() {
    results_1_len = 0;
    results_2_len = 0;
    results_3_len = 0;
    results_4_len = 0;

    FILE *output_file = std::fopen("output.txt", "w");
    if (output_file == NULL) PANIC("Could not open output.txt\n");

    uint64_t parts = 1;
    uint64_t part = 0;
	
    uint64_t seeds_total = UINT64_C(1) << (48 - 4);
    uint64_t seeds_per_run = threads_per_run;

    uint64_t part_seed_start = part * seeds_total / parts;
    uint64_t part_seed_end = (part + 1) * seeds_total / parts;

    void *results_3_mask_ptr;
    TRY_CUDA(cudaGetSymbolAddress(&results_3_mask_ptr, results_3_mask));

    for (uint64_t run_seed_start = part_seed_start; run_seed_start < part_seed_end; run_seed_start += seeds_per_run) {
        auto start_time = std::chrono::steady_clock::now();

        if (collapse_nearby_seeds) {
            cudaMemsetAsync(results_3_mask_ptr, 0, sizeof(results_3_mask));
        }

        filter_1<<<blocks_per_run, threads_per_block>>>(run_seed_start);
        TRY_CUDA(cudaGetLastError());
        TRY_CUDA(cudaDeviceSynchronize());

        if (results_1_len > max_results_1_len) {
            std::fprintf(stderr, "results_1_len > max_results_1_len, ignored %" PRIu64 "\n", results_1_len - max_results_1_len);
            results_1_len = max_results_1_len;
        }

        filter_2<<<results_1_len / threads_per_block + 1, threads_per_block>>>();
        TRY_CUDA(cudaGetLastError());
        TRY_CUDA(cudaDeviceSynchronize());

        if (results_2_len > max_results_2_len) {
            std::fprintf(stderr, "results_2_len > max_results_2_len, ignored %" PRIu64 "\n", results_2_len - max_results_2_len);
            results_2_len = max_results_2_len;
        }

        filter_3<<<results_2_len / threads_per_block + 1, threads_per_block>>>();
        TRY_CUDA(cudaGetLastError());
        TRY_CUDA(cudaDeviceSynchronize());

        if (results_3_len > max_results_3_len) {
            std::fprintf(stderr, "results_3_len > max_results_3_len, ignored %" PRIu64 "\n", results_3_len - max_results_3_len);
            results_3_len = max_results_3_len;
        }

        filter_4<<<results_3_len * (max_calls + 1) * (1 << max_tree_count) / threads_per_block + 1, threads_per_block>>>();
        TRY_CUDA(cudaGetLastError());
        TRY_CUDA(cudaDeviceSynchronize());

        if (results_4_len > max_results_4_len) {
            std::fprintf(stderr, "results_4_len > max_results_4_len, ignored %" PRIu64 "\n", results_4_len - max_results_4_len);
            results_4_len = max_results_4_len;
        }

        if (run_seed_start == part_seed_start) {
            std::fprintf(stderr, "Counts: %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", results_1_len, results_2_len, results_3_len, results_4_len);
        }
        // for (uint64_t i = 0; i < results_3_len; i++) {
        //     std::printf("3: %" PRIu64 "\n", results_3[i]);
        // }
        for (uint64_t i = 0; i < results_4_len; i++) {
            std::printf("%" PRIu64 "\n", results_4[i]);
            std::fprintf(output_file, "%" PRIu64 "\n", results_4[i]);
        }

        results_1_len = 0;
        results_2_len = 0;
        results_3_len = 0;
        results_4_len = 0;

        if ((run_seed_start - part_seed_start) / seeds_per_run % 256 == 0) {
            auto end_time = std::chrono::steady_clock::now();
            double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / 1000000000.0;
            double sps = seeds_per_run / time;
            double eta = (part_seed_end - (run_seed_start + seeds_per_run)) / sps;

            std::fprintf(stderr, "%.3f s, %.3f Gsps, ETA: %f s\n", time, sps / 1000000000.0, eta);
        }
	}

    std::fclose(output_file);

	return 0;
}