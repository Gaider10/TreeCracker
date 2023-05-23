#pragma once

#include <cstdint>

struct LCG {
    static constexpr uint64_t MULTIPLIER = UINT64_C(0x5DEECE66D);
    static constexpr uint64_t ADDEND = UINT64_C(0xB);
    static constexpr uint64_t MASK = UINT64_C(0xFFFFFFFFFFFF);

    uint64_t multiplier;
    uint64_t addend;

    __host__ __device__ constexpr explicit LCG(uint64_t multiplier, uint64_t addend);

    __host__ __device__ constexpr void combine(const LCG &other);

    __host__ __device__ static constexpr LCG combine(int64_t n);
};

struct Random {
    uint64_t seed;

    __host__ __device__ explicit Random();
    __host__ __device__ Random(const Random &other);
    __host__ __device__ explicit Random(uint64_t seed);
    __host__ __device__ static Random withSeed(uint64_t seed);

    __host__ __device__ void setSeed(uint64_t seed);
    __host__ __device__ uint64_t setDecorationSeed(uint64_t worldSeed, int32_t x, int32_t z);
    __host__ __device__ void setFeatureSeed(uint64_t decorationSeed, uint32_t index, uint32_t step);
    __host__ __device__ void setLargeFeatureSeed(uint64_t worldSeed, int32_t chunkX, int32_t chunkZ);
    __host__ __device__ void setLargeFeatureWithSalt(uint64_t worldSeed, int32_t regionX, int32_t regionZ, int32_t salt);

    template<int64_t N = 1>
    __host__ __device__ Random &skip();

    template<int64_t N = 1>
    __host__ __device__ uint32_t next(uint32_t bits);

    template<int64_t N = 1>
    __host__ __device__ uint32_t nextInt();

    template<int64_t N = 1>
    __host__ __device__ uint32_t nextInt(uint32_t bound);

    template<int64_t N = 1>
    __host__ __device__ uint32_t nextIntFast(uint32_t bound);

    template<int64_t N = 1>
    __host__ __device__ uint64_t nextLong();

    template<int64_t N = 1>
    __host__ __device__ bool nextBoolean();

    template<int64_t N = 1>
    __host__ __device__ float nextFloat();

    template<int64_t N = 1>
    __host__ __device__ double nextDouble();
};

__host__ __device__ constexpr LCG::LCG(uint64_t multiplier, uint64_t addend) : multiplier(multiplier), addend(addend) {

}

__host__ __device__ constexpr void LCG::combine(const LCG &other) {
    uint64_t combined_multiplier = (multiplier * other.multiplier) & MASK;
    uint64_t combined_addend = (addend * other.multiplier + other.addend) & MASK;
    multiplier = combined_multiplier;
    addend = combined_addend;
}

__host__ __device__ constexpr LCG LCG::combine(int64_t n) {
    uint64_t skip = static_cast<uint64_t>(n) & MASK;

    LCG lcg(1, 0);
    LCG lcg_pow(MULTIPLIER, ADDEND);

    for (uint64_t i = 0; i < 48; i++) {
        if (skip & (UINT64_C(1) << i)) {
            lcg.combine(lcg_pow);
        }

        lcg_pow.combine(lcg_pow);
    }

    return lcg;
}

__host__ __device__ Random::Random() {

}

__host__ __device__ Random::Random(const Random &other) : seed(other.seed) {

}

__host__ __device__ Random::Random(uint64_t s) : seed((s ^ LCG::MULTIPLIER) & LCG::MASK) {

}

__host__ __device__ Random Random::withSeed(uint64_t seed) {
    Random random;
    random.seed = seed;
    return random;
}

template<int64_t N>
__host__ __device__ Random &Random::skip<N>() {
    constexpr LCG lcg = LCG::combine(N);
    this->seed = (this->seed * lcg.multiplier + lcg.addend) & LCG::MASK;
    return *this;
}

template<int64_t N>
__host__ __device__ uint32_t Random::next(uint32_t bits) {
    this->skip<N>();
    return static_cast<uint32_t>(this->seed >> (48 - bits));
}

template<int64_t N>
__host__ __device__ uint32_t Random::nextInt() {
    return this->next<N>(32);
}

template<int64_t N>
__host__ __device__ uint32_t Random::nextInt(uint32_t bound) {
    uint32_t r = this->next<N>(31);
    uint32_t m = bound - 1;
    if ((bound & m) == 0) {
        r = static_cast<uint32_t>((static_cast<uint64_t>(bound) * static_cast<uint64_t>(r)) >> 31);
    } else {
        for (uint32_t u = r;
                static_cast<int32_t>(u - (r = u % bound) + m) < 0;
                u = this->next<1>(31))
            ;
    }
    return r;
}

template<int64_t N>
__host__ __device__ uint32_t Random::nextIntFast(uint32_t bound) {
    uint32_t r = this->next<N>(31);
    uint32_t m = bound - 1;
    if ((bound & m) == 0) {
        r = static_cast<uint32_t>((static_cast<uint64_t>(bound) * static_cast<uint64_t>(r)) >> 31);
    } else {
        r = r % bound;
    }
    return r;
}

template<int64_t N>
__host__ __device__ uint64_t Random::nextLong() {
    uint32_t a = this->next<N>(32);
    uint32_t b = this->next<1>(32);
    return (static_cast<uint64_t>(a) << 32) + static_cast<uint64_t>(static_cast<int32_t>(b));
}

template<int64_t N>
__host__ __device__ bool Random::nextBoolean() {
    return this->next<N>(1) != 0;
}

template<int64_t N>
__host__ __device__ float Random::nextFloat() {
    return static_cast<float>(this->next<N>(24)) * 0x1.0p-24f;
}

template<int64_t N>
__host__ __device__ double Random::nextDouble() {
    uint32_t a = this->next<N>(26);
    uint32_t b = this->next<1>(27);
    return static_cast<double>((static_cast<uint64_t>(a) << 27) + static_cast<uint64_t>(b)) * 0x1.0p-53;
}

__host__ __device__ void Random::setSeed(uint64_t s) {
    this->seed = (s ^ LCG::MULTIPLIER) & LCG::MASK;
}

__host__ __device__ uint64_t Random::setDecorationSeed(uint64_t worldSeed, int32_t x, int32_t z) {
    this->setSeed(worldSeed);
    uint64_t a = this->nextLong() | 1;
    uint64_t b = this->nextLong() | 1;
    uint64_t s = (static_cast<uint64_t>(x) * a + static_cast<uint64_t>(z) * b) ^ worldSeed;
    this->setSeed(s);
    return s;
}

__host__ __device__ void Random::setFeatureSeed(uint64_t decorationSeed, uint32_t index, uint32_t step) {
    uint64_t s = decorationSeed + static_cast<uint64_t>(index) + static_cast<uint64_t>(10000 * step);
    this->setSeed(s);
}

__host__ __device__ void Random::setLargeFeatureSeed(uint64_t worldSeed, int32_t chunkX, int32_t chunkZ) {
    this->setSeed(worldSeed);
    uint64_t a = this->nextLong();
    uint64_t b = this->nextLong();
    uint64_t s = (static_cast<uint64_t>(chunkX) * a ^ static_cast<uint64_t>(chunkZ) * b) ^ worldSeed;
    this->setSeed(s);
}

__host__ __device__ void Random::setLargeFeatureWithSalt(uint64_t worldSeed, int32_t regionX, int32_t regionZ, int32_t salt) {
    uint64_t s = static_cast<uint64_t>(regionX) * UINT64_C(341873128712) + static_cast<uint64_t>(regionZ) * UINT64_C(132897987541) + worldSeed + static_cast<uint64_t>(salt);
    this->setSeed(s);
}