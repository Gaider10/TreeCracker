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

constexpr uint64_t threads_per_run = UINT64_C(1) * 1024 * 1024 * 1024 * 4;
constexpr uint64_t threads_per_block = 256;
constexpr uint64_t blocks_per_run = threads_per_run / threads_per_block;

struct Result2 {
    uint64_t structure_seed_index;
    uint64_t tree_seed;
};

constexpr uint64_t max_results_0_len = 1024;
__device__ uint64_t results_0[max_results_0_len];
__device__ uint64_t results_0_len;
__device__ uint32_t results_0_mask[(max_results_0_len + 31) / 32];

constexpr uint64_t max_results_1_len = max_results_0_len;
__device__ Result2 results_1[max_results_1_len];
__device__ uint64_t results_1_len;

constexpr uint64_t max_results_2_len = 1024 * 1024;
__device__ Result2 results_2[max_results_2_len];
__device__ uint64_t results_2_len;

constexpr uint64_t max_results_3_len = 1024 * 1024;
__device__ Result2 results_3[max_results_3_len];
__device__ uint64_t results_3_len;
__device__ uint32_t results_3_mask[(max_results_3_len + 31) / 32];

constexpr uint64_t max_results_4_len = 1024 * 1024;
__device__ uint64_t results_4[max_results_4_len];
__device__ uint64_t results_4_len;

__device__ constexpr TreeChunk input_data = get_input_data();
constexpr int32_t max_calls = biome_max_calls(input_data.version, input_data.biome);
constexpr int32_t max_tree_count = biome_max_tree_count(input_data.version, input_data.biome);
constexpr int32_t min_skip = 0;
constexpr int32_t max_skip = input_data.version <= Version::v1_12_2 ? 10000 : max_calls;
constexpr int32_t skip_range = max_skip - min_skip + 1;
constexpr int32_t salt = biome_salt(input_data.version, input_data.biome);

__global__ __launch_bounds__(threads_per_block) void filter_1() {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= results_0_len) return;
    uint64_t structure_seed = results_0[index];

    Random random;
    if (input_data.version <= Version::v1_12_2) {
        random.setSeed(structure_seed);
        uint64_t a = static_cast<uint64_t>(static_cast<int64_t>(random.nextLong()) / 2 * 2 + 1);
        uint64_t b = static_cast<uint64_t>(static_cast<int64_t>(random.nextLong()) / 2 * 2 + 1);;
        uint64_t population_seed = static_cast<uint64_t>(input_data.chunk_x) * a + static_cast<uint64_t>(input_data.chunk_z) * b ^ structure_seed;
        random.setSeed(population_seed);
    } else {
        random.setSeed(structure_seed);
        uint64_t a = random.nextLong() | 1;
        uint64_t b = random.nextLong() | 1;
        uint64_t population_seed = static_cast<uint64_t>(input_data.chunk_x * 16) * a + static_cast<uint64_t>(input_data.chunk_z * 16) * b ^ structure_seed;
        uint64_t decoration_seed = population_seed + static_cast<uint64_t>(salt);
        random.setSeed(decoration_seed);
    }

    uint64_t result_index = atomicAdd(reinterpret_cast<unsigned long long*>(&results_1_len), 1);
    if (result_index >= max_results_1_len) return;
    results_1[result_index] = Result2 { index, random.seed };
}

__global__ __launch_bounds__(threads_per_block) void filter_2() {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t skip = index % (skip_range) + min_skip;
    index /= (skip_range);

    if (index >= results_1_len) return;
    uint64_t seed = results_1[index].tree_seed;

    Random random = Random::withSeed(seed);
    if (skip & 8192) random.skip<8192>();
    if (skip & 4096) random.skip<4096>();
    if (skip & 2048) random.skip<2048>();
    if (skip & 1024) random.skip<1024>();
    if (skip & 512) random.skip<512>();
    if (skip & 256) random.skip<256>();
    if (skip & 128) random.skip<128>();
    if (skip & 64) random.skip<64>();
    if (skip & 32) random.skip<32>();
    if (skip & 16) random.skip<16>();
    if (skip & 8) random.skip<8>();
    if (skip & 4) random.skip<4>();
    if (skip & 2) random.skip<2>();
    if (skip & 1) random.skip<1>();
    seed = random.seed;

    uint32_t found = 0;
    for (int32_t i = 0; i <= max_calls; i++) {
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
    results_2[result_index] = Result2 { results_1[index].structure_seed_index, seed };
}

__global__ __launch_bounds__(threads_per_block) void filter_3() {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= results_2_len) return;
    uint64_t seed = results_2[index].tree_seed;

    uint32_t found = 0;
    Random random = Random::withSeed(seed);
    for (int32_t i = 0; i <= max_calls; i++) {
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
    results_3[result_index] = Result2 { results_2[index].structure_seed_index, seed };
}

__global__ __launch_bounds__(threads_per_block) void filter_4() {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t generated_mask = index % (1 << max_tree_count);
    index /= 1 << max_tree_count;

    if (__popc(generated_mask) < input_data.trees_len) return;

    if (index >= results_3_len) return;
    uint64_t seed = results_3[index].tree_seed;

    Random random = Random::withSeed(seed);

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

    uint32_t results_3_index = index;
    uint32_t results_3_mask_mask = UINT32_C(1) << (results_3_index & 31);
    if (atomicOr(&results_3_mask[results_3_index / 32], results_3_mask_mask) & results_3_mask_mask) return;

    uint32_t results_0_index = results_3[index].structure_seed_index;
    uint32_t results_0_mask_mask = UINT32_C(1) << (results_0_index & 31);
    if (atomicOr(&results_0_mask[results_0_index / 32], results_0_mask_mask) & results_0_mask_mask) return;

    uint64_t result_index = atomicAdd(reinterpret_cast<unsigned long long*>(&results_4_len), 1);
    if (result_index >= max_results_4_len) return;
    results_4[result_index] = results_0[results_0_index];
}

int main() {
    uint64_t host_results_0_len = 0;
    uint64_t host_results_1_len = 0;
    uint64_t host_results_2_len = 0;
    uint64_t host_results_3_len = 0;
    uint64_t host_results_4_len = 0;
    void *device_results_0_len;
    void *device_results_1_len;
    void *device_results_2_len;
    void *device_results_3_len;
    void *device_results_4_len;
    void *device_results_0;
    void *device_results_4;
    TRY_CUDA(cudaGetSymbolAddress(&device_results_0_len, results_0_len));
    TRY_CUDA(cudaGetSymbolAddress(&device_results_1_len, results_1_len));
    TRY_CUDA(cudaGetSymbolAddress(&device_results_2_len, results_2_len));
    TRY_CUDA(cudaGetSymbolAddress(&device_results_3_len, results_3_len));
    TRY_CUDA(cudaGetSymbolAddress(&device_results_4_len, results_4_len));
    TRY_CUDA(cudaGetSymbolAddress(&device_results_0, results_0));
    TRY_CUDA(cudaGetSymbolAddress(&device_results_4, results_4));
    TRY_CUDA(cudaMemsetAsync(device_results_0_len, 0, sizeof(results_0_len)));
    TRY_CUDA(cudaMemsetAsync(device_results_1_len, 0, sizeof(results_1_len)));
    TRY_CUDA(cudaMemsetAsync(device_results_2_len, 0, sizeof(results_2_len)));
    TRY_CUDA(cudaMemsetAsync(device_results_3_len, 0, sizeof(results_3_len)));
    TRY_CUDA(cudaMemsetAsync(device_results_4_len, 0, sizeof(results_4_len)));
    std::vector<uint64_t> host_results_0;
    host_results_0.reserve(host_results_0_len);
    std::vector<uint64_t> host_results_4(max_results_4_len);

    FILE *input_file = std::fopen("input.txt", "r");
    if (input_file == NULL) PANIC("Could not open input file\n");

    FILE *output_file = std::fopen("output.txt", "w");
    if (output_file == NULL) PANIC("Could not open output file\n");

    void *results_0_mask_ptr;
    TRY_CUDA(cudaGetSymbolAddress(&results_0_mask_ptr, results_0_mask));
    TRY_CUDA(cudaMemsetAsync(results_0_mask_ptr, 0, sizeof(results_0_mask)));

    void *results_3_mask_ptr;
    TRY_CUDA(cudaGetSymbolAddress(&results_3_mask_ptr, results_3_mask));
    TRY_CUDA(cudaMemsetAsync(results_3_mask_ptr, 0, sizeof(results_3_mask)));

    bool input_eof = false;
    while (!input_eof) {
        while (host_results_0.size() < max_results_0_len) {
            uint64_t structure_seed;
            if (std::fscanf(input_file, "%" SCNu64 "\n", &structure_seed) != 1) {
                std::fclose(input_file);
                input_eof = true;
                break;
            }
            host_results_0.push_back(structure_seed);
        }
        host_results_0_len = host_results_0.size();
        TRY_CUDA(cudaMemcpyAsync(device_results_0_len, &host_results_0_len, sizeof(results_0_len), cudaMemcpyHostToDevice));
        TRY_CUDA(cudaMemcpyAsync(device_results_0, host_results_0.data(), sizeof(*results_0) * host_results_0_len, cudaMemcpyHostToDevice));

        filter_1<<<blocks_per_run, threads_per_block>>>();
        TRY_CUDA(cudaGetLastError());
        TRY_CUDA(cudaMemcpyAsync(&host_results_1_len, device_results_1_len, sizeof(results_1_len), cudaMemcpyDeviceToHost));
        TRY_CUDA(cudaDeviceSynchronize());

        if (host_results_1_len > max_results_1_len) {
            std::fprintf(stderr, "results_1_len > max_results_1_len, ignored %" PRIu64 "\n", host_results_1_len - max_results_1_len);
            host_results_1_len = max_results_1_len;
        }

        filter_2<<<host_results_1_len * skip_range / threads_per_block + 1, threads_per_block>>>();
        TRY_CUDA(cudaGetLastError());
        TRY_CUDA(cudaMemcpyAsync(&host_results_2_len, device_results_2_len, sizeof(results_2_len), cudaMemcpyDeviceToHost));
        TRY_CUDA(cudaDeviceSynchronize());

        if (host_results_2_len > max_results_2_len) {
            std::fprintf(stderr, "results_2_len > max_results_2_len, ignored %" PRIu64 "\n", host_results_2_len - max_results_2_len);
            host_results_2_len = max_results_2_len;
        }

        filter_3<<<host_results_2_len / threads_per_block + 1, threads_per_block>>>();
        TRY_CUDA(cudaGetLastError());
        TRY_CUDA(cudaMemcpyAsync(&host_results_3_len, device_results_3_len, sizeof(results_3_len), cudaMemcpyDeviceToHost));
        TRY_CUDA(cudaDeviceSynchronize());

        if (host_results_3_len > max_results_3_len) {
            std::fprintf(stderr, "results_3_len > max_results_3_len, ignored %" PRIu64 "\n", host_results_3_len - max_results_3_len);
            host_results_3_len = max_results_3_len;
        }

        filter_4<<<host_results_3_len * (1 << max_tree_count) / threads_per_block + 1, threads_per_block>>>();
        TRY_CUDA(cudaGetLastError());
        TRY_CUDA(cudaMemcpyAsync(&host_results_4_len, device_results_4_len, sizeof(results_4_len), cudaMemcpyDeviceToHost));
        TRY_CUDA(cudaDeviceSynchronize());

        if (host_results_4_len > max_results_4_len) {
            std::fprintf(stderr, "results_4_len > max_results_4_len, ignored %" PRIu64 "\n", host_results_4_len - max_results_4_len);
            host_results_4_len = max_results_4_len;
        }

        std::printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", host_results_0_len, host_results_1_len, host_results_2_len, host_results_3_len, host_results_4_len);
        // for (uint64_t i = 0; i < results_3_len; i++) {
        //     // std::printf("%" PRIu64 "\n", results_3[i].structure_seed);
        //     std::fprintf(output_file, "%" PRIu64 "\n", results_3[i].structure_seed);
        // }
        // for (uint64_t i = 0; i < results_3_len; i++) {
        //     std::printf("3: %" PRIu64 "\n", results_0[results_3[i].structure_seed_index]);
        // }
        TRY_CUDA(cudaMemcpy(host_results_4.data(), device_results_4, sizeof(*results_4) * host_results_4_len, cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < host_results_4_len; i++) {
            std::printf("%" PRIu64 "\n", host_results_4[i]);
            std::fprintf(output_file, "%" PRIu64 "\n", host_results_4[i]);
        }

        TRY_CUDA(cudaMemsetAsync(results_0_mask_ptr, 0, (host_results_0_len + 31) / 32 * 4));
        TRY_CUDA(cudaMemsetAsync(results_3_mask_ptr, 0, (host_results_3_len + 31) / 32 * 4));

        host_results_0.clear();
        TRY_CUDA(cudaMemsetAsync(device_results_1_len, 0, sizeof(results_1_len)));
        TRY_CUDA(cudaMemsetAsync(device_results_2_len, 0, sizeof(results_2_len)));
        TRY_CUDA(cudaMemsetAsync(device_results_3_len, 0, sizeof(results_3_len)));
        TRY_CUDA(cudaMemsetAsync(device_results_4_len, 0, sizeof(results_4_len)));
	}

    std::fclose(output_file);

	return 0;
}