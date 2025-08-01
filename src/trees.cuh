#pragma once

#include <cstdint>
#include <cstdio>
#include <cassert>

#include "Random.cuh"

__device__ void a(const char *) {}

#define cassert(expr, msg) if (!(expr)) { a(msg); }

enum struct Version {
    v1_6_4,
    v1_8_9,
    v1_12_2,
    v1_14_4,
    v1_15_2,
    v1_16_1,
    v1_16_4,
};

__device__ constexpr int32_t pc(Version version, int32_t a) {
    if (version <= Version::v1_12_2) {
        return (a - 8) >> 4;
    } else {
        return a >> 4;
    }
}

__device__ constexpr uint32_t po(Version version, int32_t a) {
    if (version <= Version::v1_12_2) {
        return (a - 8) & 15;
    } else {
        return a & 15;
    }
}

enum struct Biome {
    Forest,
    BirchForest,
    Taiga,
    SnowyTundra,
};

enum struct TreeType {
    Unknown,
    Oak,
    FancyOak,
    Birch,
    Pine,
    Spruce,
};

struct TreeTypes {
    uint32_t mask;

    __device__ constexpr TreeTypes() : mask() {

    }

    __device__ constexpr TreeTypes(const TreeTypes &other) : mask(other.mask) {

    }

    __device__ constexpr bool is_empty() const {
        return this->mask == 0;
    }

    __device__ constexpr void add(TreeType type) {
        this->mask |= static_cast<uint32_t>(1) << static_cast<uint32_t>(type);
    }

    __device__ constexpr bool contains(TreeType type) const {
        return (this->mask & (static_cast<uint32_t>(1) << static_cast<uint32_t>(type))) != 0;
    }
};

__device__ constexpr int32_t biome_salt(Version version, Biome biome) {
    if (version <= Version::v1_12_2) return 0;

    switch (biome) {
        case Biome::Forest: {
            // 1.14.4 - 60001
            // 1.15.2 - 60001
            // 1.16   - 80001
            // 1.16.1 - 80001
            // 1.16.4 - 80001
            return version <= Version::v1_15_2 ? 60001 : 80001;
        };
        case Biome::BirchForest: {
            // 1.15.2 - 60001
            // 1.16   - 80001
            return version <= Version::v1_15_2 ? 60001 : 80001;
        };
        case Biome::Taiga: {
            // 1.15.2 - 60001
            // 1.16   - 80001
            return version <= Version::v1_15_2 ? 60001 : 80001;
        };
        case Biome::SnowyTundra: {
            // 1.15.2 - 60000
            // 1.16   - 80000
            return version <= Version::v1_15_2 ? 60000 : 80000;
        };
    }

    cassert(false, "Not implemented");
    return 0;
}

__device__ constexpr bool biome_has_tree_type(Version version, Biome biome, TreeType tree_type) {
    if (tree_type == TreeType::Unknown) return true;

    switch (biome) {
        case Biome::Forest: {
            if (version > Version::v1_6_4 && version <= Version::v1_8_9) {
                return
                    tree_type == TreeType::Oak ||
                    tree_type == TreeType::Birch;
            } else {
                return
                    tree_type == TreeType::Oak ||
                    tree_type == TreeType::FancyOak ||
                    tree_type == TreeType::Birch;
            }
        } break;
        case Biome::BirchForest: {
            return
                tree_type == TreeType::Birch;
        } break;
        case Biome::Taiga: {
            return
                tree_type == TreeType::Pine ||
                tree_type == TreeType::Spruce;
        } break;
        case Biome::SnowyTundra: {
            return tree_type == TreeType::Spruce;
        } break;
    }

    cassert(false, "Not implemented");
    return false;
}

__device__ TreeType biome_get_tree_type(Version version, Biome biome, Random &random) {
    switch (biome) {
        case Biome::Forest: {
            if (version <= Version::v1_6_4) {
                if (random.nextInt(5) == 0) return TreeType::Birch;
                if (random.nextInt(10) == 0) return TreeType::FancyOak;
                return TreeType::Oak;
            } else if (version <= Version::v1_8_9) {
                // Not sure exactly in which versions this happens
                if (random.nextInt(5) == 0) return TreeType::Birch;
                return TreeType::Oak;
            } else if (version <= Version::v1_12_2) {
                if (random.nextInt(5) == 0) return TreeType::Birch;
                if (random.nextInt(10) == 0) return TreeType::FancyOak;
                return TreeType::Oak;
            } else {
                if (random.nextFloat() < 0.2f) return TreeType::Birch;
                if (random.nextFloat() < 0.1f) return TreeType::FancyOak;
                return TreeType::Oak;
            }
        } break;
        case Biome::BirchForest: {
            return TreeType::Birch;
        } break;
        case Biome::Taiga: {
            if (version <= Version::v1_12_2) {
                if (random.nextInt(3) == 0) return TreeType::Pine;
                return TreeType::Spruce;
            } else {
                if (random.nextFloat() < 0.33333334f) return TreeType::Pine;
                return TreeType::Spruce;
            }
        } break;
        case Biome::SnowyTundra: {
            return TreeType::Spruce;
        } break;
    }

    cassert(false, "Not implemented");
    return TreeType::Unknown;
}

__device__ constexpr int32_t biome_min_tree_count(Version version, Biome biome) {
    switch (biome) {
        case Biome::Forest: {
            return 10;
        };
        case Biome::BirchForest: {
            return 10;
        };
        case Biome::Taiga: {
            return 10;
        };
        case Biome::SnowyTundra: {
            return 0;
        };
    }

    cassert(false, "Not implemented");
    return 0;
}

__device__ constexpr float biome_extra_tree_chance(Version version, Biome biome) {
    return 0.1f;
}

__device__ constexpr int32_t biome_max_tree_count(Version version, Biome biome) {
    int32_t min_tree_count = biome_min_tree_count(version, biome);
    // Probably not 1.8.9
    if (version <= Version::v1_8_9) {
        return min_tree_count + 1;
    } else {
        if (biome_extra_tree_chance(version, biome) != 0.0f) {
            return min_tree_count + 1;
        } else {
            return min_tree_count;
        }
    }
}

__device__ int32_t biome_tree_count(Version version, Biome biome, Random &random) {
    int32_t tree_count = biome_min_tree_count(version, biome);

    if (version <= Version::v1_8_9) {
        if (random.nextInt(10) == 0) {
            tree_count += 1;
        }
    } else {
        if (random.nextFloat() < biome_extra_tree_chance(version, biome)) {
            tree_count += 1;
        }
    }

    return tree_count;
}

__device__ constexpr int32_t biome_max_calls(Version version, Biome biome) {
    switch (biome) {
        case Biome::Forest: {
            return biome_max_tree_count(version, biome) * 21;
        } break;
        case Biome::BirchForest: {
            return biome_max_tree_count(version, biome) * 19;
        } break;
        case Biome::Taiga: {
            return biome_max_tree_count(version, biome) * 9;
        } break;
        case Biome::SnowyTundra: {
            return biome_max_tree_count(version, biome) * 8;
        } break;
    }

    cassert(false, "Not implemented");
    return 0;
}

struct BlobLeaves {
    uint32_t mask;

    __device__ constexpr BlobLeaves() : mask() {

    }

    __device__ constexpr BlobLeaves(const BlobLeaves &other) : mask(other.mask) {

    }

    __device__ constexpr BlobLeaves(uint32_t mask) : mask(mask) {

    }

    __device__ static constexpr BlobLeaves make(Version version, const char *str) {
        for (int32_t i = 0; i < 12; i++) {
            cassert(str[i] != '\0', "Leaves string too short");
            cassert(str[i] == '0' || str[i] == '1' || str[i] == '?', "Invalid leaves string character");
        }
        cassert(str[12] == '\0', "Leaves string too long");

        if (version <= Version::v1_14_4) {
            uint32_t mask = 0;
            for (int32_t y = 0; y < 3; y++) {
                for (int32_t xz = 0; xz < 4; xz++) {
                    int32_t i = y * 4 + xz;
                    int32_t j = i;
                    char c = str[i];
                    bool is_known = c == '0' || c == '1';
                    bool is_present = c == '1';
                    mask |= static_cast<uint32_t>(is_known) << (16 + j);
                    mask |= static_cast<uint32_t>(is_present) << (j);
                }
            }
            return BlobLeaves(mask);
        } else {
            uint32_t mask = 0;
            for (int32_t y = 0; y < 3; y++) {
                for (int32_t xz = 0; xz < 4; xz++) {
                    int32_t i = y * 4 + xz;
                    int32_t j = (3 - y) * 4 + xz;
                    char c = str[i];
                    bool is_known = c == '0' || c == '1';
                    bool is_present = c == '1';
                    mask |= static_cast<uint32_t>(is_known) << (16 + j);
                    mask |= static_cast<uint32_t>(is_present) << (j);
                }
            }
            return BlobLeaves(mask);
        }
    }

    __device__ bool test(Version version, Random &random) const {
        if ((mask >> (16 +  0) & 1) && (mask >>  0 & 1) != Random(random).nextInt< 1>(2)) return false;
        if ((mask >> (16 +  1) & 1) && (mask >>  1 & 1) != Random(random).nextInt< 2>(2)) return false;
        if ((mask >> (16 +  2) & 1) && (mask >>  2 & 1) != Random(random).nextInt< 3>(2)) return false;
        if ((mask >> (16 +  3) & 1) && (mask >>  3 & 1) != Random(random).nextInt< 4>(2)) return false;
        if ((mask >> (16 +  4) & 1) && (mask >>  4 & 1) != Random(random).nextInt< 5>(2)) return false;
        if ((mask >> (16 +  5) & 1) && (mask >>  5 & 1) != Random(random).nextInt< 6>(2)) return false;
        if ((mask >> (16 +  6) & 1) && (mask >>  6 & 1) != Random(random).nextInt< 7>(2)) return false;
        if ((mask >> (16 +  7) & 1) && (mask >>  7 & 1) != Random(random).nextInt< 8>(2)) return false;
        if ((mask >> (16 +  8) & 1) && (mask >>  8 & 1) != Random(random).nextInt< 9>(2)) return false;
        if ((mask >> (16 +  9) & 1) && (mask >>  9 & 1) != Random(random).nextInt<10>(2)) return false;
        if ((mask >> (16 + 10) & 1) && (mask >> 10 & 1) != Random(random).nextInt<11>(2)) return false;
        if ((mask >> (16 + 11) & 1) && (mask >> 11 & 1) != Random(random).nextInt<12>(2)) return false;
        if ((mask >> (16 + 12) & 1) && (mask >> 12 & 1) != Random(random).nextInt<13>(2)) return false;
        if ((mask >> (16 + 13) & 1) && (mask >> 13 & 1) != Random(random).nextInt<14>(2)) return false;
        if ((mask >> (16 + 14) & 1) && (mask >> 14 & 1) != Random(random).nextInt<15>(2)) return false;
        if ((mask >> (16 + 15) & 1) && (mask >> 15 & 1) != Random(random).nextInt<16>(2)) return false;
        random.skip<16>();
        return true;
    }
};

struct IntRange {
    int32_t min;
    int32_t max;

    __device__ constexpr IntRange() : min(), max() {

    }

    __device__ constexpr IntRange(const IntRange &other) : min(other.min), max(other.max) {

    }

    __device__ constexpr IntRange(int32_t val) : min(val), max(val) {

    }

    __device__ constexpr IntRange(int32_t min, int32_t max) : min(min), max(max) {

    }

    __device__ bool test(int32_t val) const {
        if (min != -1 && val < min) return false;
        if (max != -1 && val > max) return false;
        return true;
    }
};

struct TrunkHeight {
    __device__ static uint32_t get(uint32_t a, uint32_t b, Random &random) {
        return a + random.nextInt(b + 1);
    }

    __device__ static uint32_t get(uint32_t a, uint32_t b, uint32_t c, Random &random) {
        uint32_t height = a + random.nextInt(b + 1);
        return height + random.nextInt(c + 1);
    }
};

// height == trunk_height == total_height - 1
template<uint32_t trunk_height_a, uint32_t trunk_height_b>
struct NormalTreeData {
    IntRange height;
    BlobLeaves leaves;

    __device__ constexpr NormalTreeData() : height(), leaves() {

    }

    __device__ constexpr NormalTreeData(const NormalTreeData &other) : height(other.height), leaves(other.leaves) {

    }

    __device__ constexpr NormalTreeData(IntRange height, BlobLeaves leaves) : height(height), leaves(leaves) {

    }

    __device__ bool test(Version version, Random &random) const {
        if (version <= Version::v1_14_4) {
            if (!height.test(TrunkHeight::get(trunk_height_a, trunk_height_b, random))) return false;
        } else {
            if (!height.test(TrunkHeight::get(trunk_height_a, trunk_height_b, 0, random))) return false;
        }

        if (version <= Version::v1_14_4) {

        } else if (version <= Version::v1_15_2) {
            random.skip<2>(); // foliageHeight + foliageRadius
        } else if (version <= Version::v1_16_1) {
            random.skip<2>(); // foliageRadius + offset
        } else {

        }

        if (!leaves.test(version, random)) return false;

        return true;
    }

    __device__ static void skip(Version version, Random &random, bool generated) {
        if (version <= Version::v1_14_4) {
            TrunkHeight::get(trunk_height_a, trunk_height_b, random);
        } else {
            TrunkHeight::get(trunk_height_a, trunk_height_b, 0, random);
        }

        if (version <= Version::v1_14_4) {

        } else if (version <= Version::v1_15_2) {
            random.skip<2>(); // foliageHeight + foliageRadius
        } else if (version <= Version::v1_16_1) {
            random.skip<1>(); // foliageRadius
        } else {

        }

        if (generated) {
            if (version <= Version::v1_14_4) {
                random.skip<16>(); // leaves
            } else if (version <= Version::v1_15_2) {
                random.skip<18>(); // leaves + trunkTopOffset + beehive
            } else if (version <= Version::v1_16_1) {
                random.skip<18>(); // offset + leaves + beehive
            } else {
                random.skip<17>(); // leaves + beehive
            }
        }
    }
};

using OakTreeData = NormalTreeData<4, 2>;
using BirchTreeData = NormalTreeData<5, 2>;

// height == total_height
struct FancyOakTreeData {
    IntRange height;

    __device__ constexpr FancyOakTreeData() : height() {

    }

    __device__ constexpr FancyOakTreeData(const FancyOakTreeData &other) : height(other.height) {

    }

    __device__ constexpr FancyOakTreeData(IntRange height) : height(height) {

    }

    __device__ bool test(Version version, Random &random) const {
        if (version <= Version::v1_15_2) {
            Random tree_random(random.nextLong());

            if (!height.test(TrunkHeight::get(5, 11, tree_random))) return false;
        } else {
            if (!height.test(TrunkHeight::get(3, 11, 0, random))) return false;
        }

        return true;
    }

    __device__ static float treeShape(int32_t n, int32_t n2) {
        if ((float)n2 < (float)n * 0.3f) {
            return -1.0f;
        }
        float f = (float)n / 2.0f;
        float f2 = f - (float)n2;
        float f3 = sqrt(f * f - f2 * f2);
        if (f2 == 0.0f) {
            f3 = f;
        } else if (abs(f2) >= f) {
            return 0.0f;
        }
        return f3 * 0.5f;
    }

    __device__ static void skip(Version version, Random &random, bool generated) {
        if (version <= Version::v1_14_4) {
            random.skip<2>(); // tree seed nextLong
        } else if (version <= Version::v1_15_2) {
            random.skip<2>(); // tree seed nextLong

            if (generated) {
                random.skip<1>(); // beehive
            }
        } else if (version <= Version::v1_16_1) {
            uint32_t height = TrunkHeight::get(3, 11, 0, random);

            random.skip<1>(); // foliageRadius

            if (generated) {
                uint32_t branch_count = (0x765543321000 >> ((height - 3) * 4)) & 0xF;
                uint32_t blob_count = 1;
                for (uint32_t i = 0; i < branch_count; i++) {
                    int32_t n4 = height + 2;
                    int32_t n2 = n4 - 5 - i;
                    float f = treeShape(n4, n2);
                    double d4 = 1.0 * (double)f * ((double)random.nextFloat() + 0.328);
                    double d2 = (double)(random.nextFloat() * 2.0f) * 3.141592653589793;
                    int32_t ex = floor(d4 * sin(d2) + 0.5);
                    int32_t ez = floor(d4 * cos(d2) + 0.5);
                    int32_t ey = n2 - 1;
                    int32_t sy = ey - sqrt((double)(ex * ex + ez * ez)) * 0.381;
                    if (sy >= n4 * 0.2) {
                        blob_count++;
                    }
                }
                if (blob_count & 8) random.skip<8>();
                if (blob_count & 4) random.skip<4>();
                if (blob_count & 2) random.skip<2>();
                if (blob_count & 1) random.skip<1>();

                random.skip<1>(); // beehive
            }
        } else {
            uint32_t height = TrunkHeight::get(3, 11, 0, random);

            if (generated) {
                // Trunk 0 - 14
                uint32_t branch_count = ((0x765543321000 >> ((height - 3) * 4)) & 0xF);
                if (branch_count & 4) random.skip<8>();
                if (branch_count & 2) random.skip<4>();
                if (branch_count & 1) random.skip<2>();

                random.skip<1>(); // beehive
            }
        }
    }
};

// <= 1.12.2...?
// height: `total_height - 1` == `trunk_height + 1`
// leaves_height: `leaves_height - 1` BUT if the last 2 leaf radiuses are the same the last layer has radius 0, so visually it's just `leaves_height`
// leaves_radius: radius of the widest layer
// >= ?...1.16.4
// height: `trunk_height` == `total_height - 2`
// leaves_height: number of leaves layer
// leaves_radius: radius of the widest layer
struct PineTreeData {
    IntRange height;
    IntRange leaves_height;
    IntRange leaves_radius;

    __device__ constexpr PineTreeData() : height(), leaves_height(), leaves_radius() {

    }

    __device__ constexpr PineTreeData(const PineTreeData &other) : height(other.height), leaves_height(other.leaves_height), leaves_radius(other.leaves_radius) {

    }

    __device__ constexpr PineTreeData(IntRange height, IntRange leaves_height, IntRange leaves_radius) : height(height), leaves_height(leaves_height), leaves_radius(leaves_radius) {

    }

    __device__ bool test(Version version, Random &random) const {
        if (version <= Version::v1_14_4) {
            // 1.7...1.12.2
            if (!height.test(TrunkHeight::get(7, 4, random))) return false;
            uint32_t gen_leaves_height = 3 + random.nextInt(2);
            if (!leaves_height.test(gen_leaves_height)) return false;
            uint32_t gen_max_leaves_radius = 1 + random.nextInt(gen_leaves_height + 1);
            uint32_t gen_leaves_radius = gen_leaves_height - 1;
            if (gen_leaves_radius > gen_max_leaves_radius) gen_leaves_radius = gen_max_leaves_radius;
            if (!leaves_radius.test(gen_leaves_radius)) return false;
        } else {
            // 1.16.4
            uint32_t trunk_height = TrunkHeight::get(6, 4, 0, random);
            if (!height.test(trunk_height)) return false;

            uint32_t foliage_height = 3 + random.nextInt(2);

            uint32_t foliage_radius = 1 + random.nextInt(trunk_height - foliage_height + 1);

            uint32_t actual_leaves_height = foliage_height + 1 - (foliage_radius == 1);
            if (!leaves_height.test(actual_leaves_height)) return false;

            uint32_t actual_leaves_radius = foliage_radius;
            if (actual_leaves_radius > foliage_height - 1) actual_leaves_radius = foliage_height - 1;
            if (!leaves_radius.test(actual_leaves_radius)) return false;
        }

        return true;
    }

    __device__ static void skip(Version version, Random &random, bool generated) {
        if (version <= Version::v1_14_4) {
            // 1.7...1.12.2
            TrunkHeight::get(7, 4, random);
            uint32_t gen_leaves_height = 3 + random.nextInt(2);
            uint32_t gen_max_leaves_radius = 1 + random.nextInt(gen_leaves_height + 1);
        } else {
            // 1.16.4
            uint32_t trunk_height = TrunkHeight::get(6, 4, 0, random);

            uint32_t foliage_height = 3 + random.nextInt(2);

            uint32_t foliage_radius = 1 + random.nextInt(trunk_height - foliage_height);
        }
    }
};

// <= 1.12.2...? height: `total_height - 1`
// >= ?...1.16.4 height: `trunk_height`
// no_leaves_height: number of logs without leaves starting from the ground
// leaves_radius: radius of the widest layer, not the actual radius but the limit, so use -1 as max unless you know the radius was limited
// top_leaves_radius: radius of the top layer
// trunk_reduction: number of leaves above the trunk - 1
struct SpruceTreeData {
    IntRange height;
    IntRange no_leaves_height;
    IntRange leaves_radius;
    IntRange top_leaves_radius;
    IntRange trunk_reduction;

    __device__ constexpr SpruceTreeData() : height(), no_leaves_height(), leaves_radius(), top_leaves_radius(), trunk_reduction() {

    }

    __device__ constexpr SpruceTreeData(const SpruceTreeData &other) : height(other.height), no_leaves_height(other.no_leaves_height), leaves_radius(other.leaves_radius), top_leaves_radius(other.top_leaves_radius), trunk_reduction(other.trunk_reduction) {

    }

    __device__ constexpr SpruceTreeData(IntRange height, IntRange no_leaves_height, IntRange leaves_radius, IntRange top_leaves_radius, IntRange trunk_reduction) : height(height), no_leaves_height(no_leaves_height), leaves_radius(leaves_radius), top_leaves_radius(top_leaves_radius), trunk_reduction(trunk_reduction) {

    }

    __device__ bool test(Version version, Random &random) const {
        if (version <= Version::v1_14_4) {
            // 1.7...1.12.2
            uint32_t gen_height = TrunkHeight::get(6, 3, random);
            if (!height.test(gen_height)) return false;
            uint32_t gen_no_leaves_height = 1 + random.nextInt(2);
            if (!no_leaves_height.test(gen_no_leaves_height)) return false;
            uint32_t gen_max_leaves_radius = 2 + random.nextInt(2);
            uint32_t gen_top_leaves_radius = random.nextInt(2);
            if (!top_leaves_radius.test(gen_top_leaves_radius)) return false;
            uint32_t gen_leaves_radius = gen_height + 1 - gen_no_leaves_height + gen_top_leaves_radius >= 8 ? 3 : 2;
            if (gen_leaves_radius > gen_max_leaves_radius) gen_leaves_radius = gen_max_leaves_radius;
            if (!leaves_radius.test(gen_leaves_radius)) return false;
            uint32_t gen_trunk_reduction = random.nextInt(3);
            if (!trunk_reduction.test(gen_trunk_reduction)) return false;
        } else {
            // 1.16.4
            uint32_t trunk_height = TrunkHeight::get(5, 2, 1, random);
            if (!height.test(trunk_height)) return false;

            uint32_t foliage_trunk_height = 1 + random.nextInt(2);

            uint32_t foliage_height = trunk_height - foliage_trunk_height;
            if (foliage_height < 4) foliage_height = 4;

            uint32_t actual_no_leaves_height = trunk_height - foliage_height;
            if (!no_leaves_height.test(actual_no_leaves_height)) return false;

            uint32_t foliage_radius = 2 + random.nextInt(2);

            uint32_t foliage_offset = 0 + random.nextInt(3);
            if (!trunk_reduction.test(foliage_offset)) return false;

            uint32_t gen_top_leaves_radius = random.nextInt(2);
            if (!top_leaves_radius.test(gen_top_leaves_radius)) return false;

            uint32_t full_foliage_height = foliage_height + foliage_offset + 1;
            uint32_t actual_leaves_radius = full_foliage_height + gen_top_leaves_radius >= 8 ? foliage_radius : 2;
            if (!leaves_radius.test(actual_leaves_radius)) return false;
        }

        return true;
    }

    __device__ static void skip(Version version, Random &random, bool generated) {
        if (version <= Version::v1_14_4) {
            if (generated) {
                random.skip<4>();
                random.nextInt(3);
            } else {
                random.skip<3>();
            }
        } else {
            if (generated) {
                random.nextInt(3);
                random.nextInt<4>(3);
                random.skip();
            } else {
                random.nextInt(3);
                random.skip<3>();
            }
        }
    }
};

struct Tree {
    uint32_t x;
    uint32_t z;
    TreeTypes types;
    OakTreeData oak_tree_data;
    FancyOakTreeData fancy_oak_tree_data;
    BirchTreeData birch_tree_data;
    PineTreeData pine_tree_data;
    SpruceTreeData spruce_tree_data;

    __device__ constexpr Tree() : x(), z(), types() {

    }

    __device__ constexpr Tree(const Tree &other) : x(other.x), z(other.z), types(other.types), oak_tree_data(other.oak_tree_data), fancy_oak_tree_data(other.fancy_oak_tree_data), birch_tree_data(other.birch_tree_data), pine_tree_data(other.pine_tree_data), spruce_tree_data(other.spruce_tree_data) {

    }

    __device__ bool test_pos_type(Version version, Biome biome, Random &random) const {
        if (random.nextInt(16) != x) return false;
        if (random.nextInt(16) != z) return false;
        TreeType type = biome_get_tree_type(version, biome, random);
        if (!types.is_empty() && !types.contains(type)) return false;

        return true;
    }

    __device__ bool test_data(Version version, Biome biome, TreeType type, Random &random) const {
        if (type == TreeType::Oak && !oak_tree_data.test(version, random)) return false;
        if (type == TreeType::FancyOak && !fancy_oak_tree_data.test(version, random)) return false;
        if (type == TreeType::Birch && !birch_tree_data.test(version, random)) return false;
        if (type == TreeType::Pine && !pine_tree_data.test(version, random)) return false;
        if (type == TreeType::Spruce && !spruce_tree_data.test(version, random)) return false;

        return true;
    }

    template<int64_t N = 1>
    __device__ bool test_full(Version version, Biome biome, Random &random) const {
        if (random.nextInt<N>(16) != x) return false;
        if (random.nextInt(16) != z) return false;
        if (!types.is_empty()) {
            TreeType type = biome_get_tree_type(version, biome, random);
            if (!types.contains(type)) return false;

            if (!test_data(version, biome, type, random)) return false;
        }

        return true;
    }

    __device__ static void skip(Version version, Biome biome, Random &random, bool generated) {
        uint32_t x = random.nextInt(16);
        uint32_t z = random.nextInt(16);
        TreeType type = biome_get_tree_type(version, biome, random);

        if (type == TreeType::Oak) OakTreeData::skip(version, random, generated);
        if (type == TreeType::FancyOak) FancyOakTreeData::skip(version, random, generated);
        if (type == TreeType::Birch) BirchTreeData::skip(version, random, generated);
        if (type == TreeType::Pine) PineTreeData::skip(version, random, generated);
        if (type == TreeType::Spruce) SpruceTreeData::skip(version, random, generated);
    }
};

struct TreeChunk {
    Version version;
    Biome biome;
    int32_t chunk_x;
    int32_t chunk_z;
    uint32_t trees_len;
    Tree trees[16];

    __device__ constexpr TreeChunk() : version(), biome(), chunk_x(), chunk_z(), trees_len(), trees() {

    }
};

struct TreeChunkBuilder {
    Version version;
    Biome biome;
    int32_t chunk_x;
    int32_t chunk_z;
    uint32_t trees_len;
    Tree trees[16];

    __device__ constexpr TreeChunkBuilder(Version version, Biome biome) : version(version), biome(biome), chunk_x(), chunk_z(), trees_len(0), trees() {

    }

    __device__ constexpr Tree &new_tree(int32_t x, int32_t z, TreeType tree_type) {
        cassert(biome_has_tree_type(version, biome, tree_type), "Invalid tree type for the given biome");

        uint32_t tree_chunk_x = pc(version, x);
        uint32_t tree_chunk_z = pc(version, z);
        if (trees_len == 0) {
            chunk_x = tree_chunk_x;
            chunk_z = tree_chunk_z;
        } else {
            cassert(tree_chunk_x == chunk_x && tree_chunk_z == chunk_z, "Tree outside the main chunk");
        }

        uint32_t tree_offset_x = po(version, x);
        uint32_t tree_offset_z = po(version, z);

        for (uint32_t i = 0; i < trees_len; i++) {
            Tree &tree = trees[i];
            if (tree.x == tree_offset_x && tree.z == tree_offset_z) {
                cassert(tree_type != TreeType::Unknown, "Duplicate tree");
                cassert(!tree.types.contains(tree_type), "Duplicate tree");
                tree.types.add(tree_type);
                return tree;
            }
        }

        Tree &tree = trees[trees_len++];
        tree.x = tree_offset_x;
        tree.z = tree_offset_z;
        if (tree_type != TreeType::Unknown) tree.types.add(tree_type);
        return tree;
    }

    __device__ constexpr TreeChunkBuilder &tree_unknown(int32_t x, int32_t z) {
        Tree &tree = new_tree(x, z, TreeType::Unknown);

        return *this;
    }

    __device__ constexpr TreeChunkBuilder &tree_oak(int32_t x, int32_t z, IntRange height, const char *leaves) {
        Tree &tree = new_tree(x, z, TreeType::Oak);

        tree.oak_tree_data = OakTreeData(height, BlobLeaves::make(version, leaves));

        return *this;
    }

    __device__ constexpr TreeChunkBuilder &tree_fancy_oak(int32_t x, int32_t z, IntRange height) {
        Tree &tree = new_tree(x, z, TreeType::FancyOak);

        tree.fancy_oak_tree_data = FancyOakTreeData(height);

        return *this;
    }

    __device__ constexpr TreeChunkBuilder &tree_birch(int32_t x, int32_t z, IntRange height, const char *leaves) {
        Tree &tree = new_tree(x, z, TreeType::Birch);

        tree.birch_tree_data = BirchTreeData(height, BlobLeaves::make(version, leaves));

        return *this;
    }

    __device__ constexpr TreeChunkBuilder &tree_pine(int32_t x, int32_t z, IntRange height, IntRange leaves_height, IntRange leaves_radius) {
        Tree &tree = new_tree(x, z, TreeType::Pine);

        tree.pine_tree_data = PineTreeData(height, leaves_height, leaves_radius);

        return *this;
    }

    __device__ constexpr TreeChunkBuilder &tree_spruce(int32_t x, int32_t z, IntRange height, IntRange no_leaves_height, IntRange leaves_radius, IntRange top_leaves_radius, IntRange trunk_reduction) {
        Tree &tree = new_tree(x, z, TreeType::Spruce);

        tree.spruce_tree_data = SpruceTreeData(height, no_leaves_height, leaves_radius, top_leaves_radius, trunk_reduction);

        return *this;
    }

    __device__ constexpr TreeChunk build() {
        cassert(trees_len != 0, "Can't have 0 trees");

        TreeChunk tree_chunk;
        tree_chunk.version = version;
        tree_chunk.biome = biome;
        tree_chunk.chunk_x = chunk_x;
        tree_chunk.chunk_z = chunk_z;
        tree_chunk.trees_len = trees_len;
        for (uint32_t i = 0; i < trees_len; i++) {
            tree_chunk.trees[i] = trees[i];
        }

        return tree_chunk;
    }
};