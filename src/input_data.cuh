#pragma once

#include "trees.cuh"

// Only the version/biome combinations that have an example input were tested, anything else will probably not work

// 
__device__ constexpr TreeChunk get_input_data() {
    return TreeChunkBuilder(Version::v1_8_9, Biome::Taiga)
        .tree_unknown(0, 0)
        .build();
}

// TEST DATA

// 1.16.4 Forest ???
// chunk_trees[0] = TreeChunk(-4, -12, Biome::Forest);
// chunk_trees[0].add_tree(tree_oak(po(-64), po(-178), 5, 5, "0?1?""????""????"));
// chunk_trees[0].add_tree(tree_oak(po(-58), po(-181), 5, 5, "0?0?""0?1?""1?1?"));
// chunk_trees[0].add_tree(tree_birch(po(-56), po(-177), 0, 0, "????""????""????"));
// chunk_trees[0].add_tree(tree_oak(po(-53), po(-182), 5, 5, "1???""1???""1???"));
// chunk_trees[0].add_tree(tree_fancy_oak(po(-50), po(-188), 3, 0));

// chunk_trees[1] = TreeChunk(-3, -11, Biome::Forest);
// chunk_trees[1].add_tree(tree_birch(po(-47), po(-168), 0, 0, "????""????""????"));
// chunk_trees[1].add_tree(tree_oak(po(-43), po(-176), 5, 5, "????""????""????"));
// chunk_trees[1].add_tree(tree_oak(po(-38), po(-168), 0, 0, "????""????""????"));
// chunk_trees[1].add_tree(tree_birch(po(-34), po(-164), 0, 0, "????""????""????"));

// chunk_trees[2] = TreeChunk(-3, -12, Biome::Forest);
// chunk_trees[2].add_tree(tree_oak(po(-45), po(-188), 5, 5, "????""????""????"));
// chunk_trees[2].add_tree(tree_oak(po(-45), po(-184), 5, 5, "????""????""????"));
// chunk_trees[2].add_tree(tree_unknown(po(-40), po(-178)));

// chunk_trees[3] = TreeChunk(-5, -3, Biome::Forest);
// chunk_trees[3].add_tree(tree_oak(po(-75), po(-40), 6, 6, "0100""1111""0111"));
// chunk_trees[3].add_tree(tree_fancy_oak(po(-77), po(-37), 0, 0));
// chunk_trees[3].add_tree(tree_birch(po(-69), po(-45), 5, 5, "101?""0011""1011"));
// chunk_trees[3].add_tree(tree_oak(po(-74), po(-48), 4, 4, "????""????""0000"));
// chunk_trees[3].add_tree(tree_oak(po(-68), po(-38), 4, 4, "1?10""0010""0000"));
// chunk_trees[3].add_tree(tree_oak(po(-70), po(-48), 4, 4, "??11""?011""0?1?"));
// chunk_trees[3].add_tree(tree_oak(po(-65), po(-42), 5, 5, "????""?011""1110"));

// 1.3.1 Forest
// __device__ constexpr TreeChunk get_input_data() {
//     return TreeChunkBuilder(Version::v1_12_2, Biome::Forest)
//         .tree_oak(-14, 71, IntRange(5), "0000""1011""0011")
//         .tree_birch(-14, 63, IntRange(6), "1111""0011""0001")
//         .tree_oak(-20, 65, IntRange(4), "?100""?111""0011")
//         .tree_oak(-21, 62, IntRange(5), "010?""000?""1100")
//         .tree_oak(-9, 64, IntRange(6), "0011""0110""1110")
//         .build();
// }

// 1.8.9 Taiga
// __device__ constexpr TreeChunk get_input_data() {
//     return TreeChunkBuilder(Version::v1_12_2, Biome::Taiga)
//         .tree_spruce(553, 4879, IntRange(9), IntRange(1), IntRange(2), IntRange(0), IntRange(2))
//         .tree_spruce(556, 4884, IntRange(9), IntRange(1), IntRange(3, -1), IntRange(0), IntRange(1))
//         .tree_pine(564, 4873, IntRange(11), IntRange(4), IntRange(2))
//         .tree_pine(565, 4876, IntRange(11), IntRange(4), IntRange(3, -1))
//         .tree_pine(561, 4880, IntRange(9), IntRange(3), IntRange(2, -1))
//         .build();
// }

// 1.7 Taiga - rehorted
// __device__ constexpr TreeChunk get_input_data() {
//     return TreeChunkBuilder(Version::v1_12_2, Biome::Taiga)
//         .tree_spruce(85, 149, IntRange(8), IntRange(1), IntRange(3, -1), IntRange(1), IntRange(0))
//         .tree_spruce(76, 151, IntRange(6), IntRange(1), IntRange(2, -1), IntRange(0), IntRange(1))
//         .tree_spruce(84, 143, IntRange(9), IntRange(2), IntRange(3, -1), IntRange(1), IntRange(0))
//         .tree_pine(76, 146, IntRange(11), IntRange(3), IntRange(2, -1))
//         .tree_pine(79, 136, IntRange(8), IntRange(3), IntRange(2, -1))
//         .build();

//     return TreeChunkBuilder(Version::v1_12_2, Biome::Taiga)
//         .tree_spruce(-93, 336, IntRange(7), IntRange(1), IntRange(2), IntRange(0), IntRange(2))
//         .tree_spruce(-103, 338, IntRange(6, 7), IntRange(1, 2), IntRange(2, -1), IntRange(-1), IntRange(-1))
//         .tree_spruce(-101, 330, IntRange(8), IntRange(2), IntRange(2), IntRange(0), IntRange(0, 1))
//         .tree_spruce(-89, 338, IntRange(7), IntRange(2), IntRange(2, -1), IntRange(0), IntRange(0, 1))
//         .tree_pine(-92, 328, IntRange(9, 10), IntRange(3), IntRange(2, -1))
//         .tree_unknown(-97, 329)
//         .build();
    
//     return TreeChunkBuilder(Version::v1_12_2, Biome::Taiga)
//         .tree_pine(-80, 353, IntRange(7), IntRange(3), IntRange(2, -1))
//         .tree_pine(-82, 358, IntRange(8, -1), IntRange(4), IntRange(1))
//         .tree_pine(-87, 359, IntRange(7, -1), IntRange(4), IntRange(2))
//         .tree_pine(-87, 351, IntRange(11, -1), IntRange(4), IntRange(3, -1))
//         .build();
// }

// 1.12.2 Taiga/Forest - smash 7
// __device__ constexpr TreeChunk get_input_data() {
//     return TreeChunkBuilder(Version::v1_12_2, Biome::Taiga)
//         .tree_pine(-715, 138, IntRange(7), IntRange(3), IntRange(1))
//         .tree_pine(-726, 146, IntRange(8, -1), IntRange(3), IntRange(1))
//         .tree_spruce(-719, 141, IntRange(8, -1), IntRange(-1), IntRange(3, -1), IntRange(0), IntRange(1))
//         .tree_spruce(-728, 151, IntRange(7, -1), IntRange(-1), IntRange(3, -1), IntRange(1), IntRange(1))
//         .build();
    
//     return TreeChunkBuilder(Version::v1_12_2, Biome::Taiga)
//         .tree_spruce(-698, 147, IntRange(6), IntRange(2), IntRange(2, -1), IntRange(0), IntRange(1))
//         .tree_spruce(-703, 145, IntRange(9), IntRange(1), IntRange(2), IntRange(1), IntRange(1, -1))
//         .tree_spruce(-710, 146, IntRange(8), IntRange(2), IntRange(2, -1), IntRange(0), IntRange(-1))
//         .tree_pine(-706, 142, IntRange(7), IntRange(4), IntRange(3, -1))
//         .build();

//     return TreeChunkBuilder(Version::v1_12_2, Biome::Forest)
//         .tree_oak(-693, 99, IntRange(5, -1), "?111""?001""?011")
//         .tree_oak(-691, 95, IntRange(6, -1), "????""?1??""????")
//         .tree_oak(-684, 95, IntRange(4), "???0""???1""????")
//         .tree_birch(-686, 97, IntRange(6), "?1?0""????""????")
//         .build();
// }

// 1.14.4 Forest - seed 123 - tree seed 13108863711061
// __device__ constexpr TreeChunk get_input_data() {
//     return TreeChunkBuilder(Version::v1_14_4, Biome::Forest)
//         .tree_oak(-48, 99, IntRange(4), "111?""10??""?1?1")
//         .tree_birch(-47, 96, IntRange(6), "0?11""110?""110?")
//         .tree_birch(-44, 98, IntRange(7), "??1?""?100""0010")
//         .tree_oak(-46, 102, IntRange(4), "1000""?0?0""?0??")
//         .tree_oak(-43, 101, IntRange(5), "??00""??01""?011")
//         .tree_oak(-34, 100, IntRange(5), "0100""1001""1010")
//         .tree_oak(-37, 106, IntRange(5), "0100""0110""0011")
//         .tree_oak(-43, 110, IntRange(6), "110?""011?""0000")
//         .build();
// }

// 1.8.9 Forest - seed 123 - tree seed 241689593949439
// __device__ constexpr TreeChunk get_input_data() {
//     return TreeChunkBuilder(Version::v1_8_9, Biome::Forest)
//         .tree_oak(133, 493, IntRange(4), "0000""1000""1111")
//         .tree_oak(129, 501, IntRange(4), "0110""1011""010?")
//         .tree_oak(135, 499, IntRange(6), "1?0?""0?0?""0001")
//         .tree_oak(121, 499, IntRange(6), "0110""1100""0110")
//         .tree_oak(122, 492, IntRange(5), "??01""???1""?010")
//         .tree_birch(123, 489, IntRange(6), "0?01""??00""??11")
//         .tree_birch(132, 503, IntRange(6), "???0""???0""1000")
//         .build();
// }

// 1.8.9 Forest - seed -5141540374460396599
// __device__ constexpr TreeChunk get_input_data() {
//     // 69261861613140
//     return TreeChunkBuilder(Version::v1_8_9, Biome::Forest)
//         .tree_birch(256 + 11, 175, IntRange(7), "01??""00?0""10?0")
//         .tree_birch(256 + 12, 183, IntRange(6), "?1?0""?0?0""00?0")
//         .tree_birch(256 + 18, 169, IntRange(6, -1), "???1""1??0""???0")
//         .tree_oak(256 + 11, 180, IntRange(5), "?0??""10??""01??")
//         .tree_oak(256 + 21, 182, IntRange(6), "?1?0""?1?1""11?1")
//         .tree_oak(256 + 23, 173, IntRange(-1), "???1""?1?0""11?1")
//         .tree_oak(256 + 23, 168, IntRange(-1), "????""????""???1")
//         .build();

//     // 83751666894233
//     // return TreeChunkBuilder(Version::v1_8_9, Biome::Forest)
//     //     .tree_birch(204, 260, IntRange(7), "00?1""11?1""1??0")
//     //     .tree_birch(209, 253, IntRange(-1), "?1?1""?1?1""???0")
//     //     .tree_oak(205, 254, IntRange(4), "1??0""1??0""???1")
//     //     .tree_oak(212, 257, IntRange(-1), "????""1???""0??0")
//     //     .tree_oak(211, 262, IntRange(5), "?0?0""?0?1""01?0")
//     //     .build();
// }

// 1.16.1 Forest Oak + Fancy Oak - seed 123
// __device__ constexpr TreeChunk get_input_data() {
//     return TreeChunkBuilder(Version::v1_16_1, Biome::Forest)
//         .tree_oak(-15, 93, IntRange(4), "1111""0101""1111")
//         .tree_oak(-15, 86, IntRange(4), "11?0""01?0""0101")
//         .tree_oak(-13, 83, IntRange(4), "0?1?""0?1?""00??")
//         .tree_oak(-10, 84, IntRange(5), "?010""?110""0111")
//         .tree_oak(-5, 82, IntRange(5), "1101""1100""0110")
//         .tree_fancy_oak(-10, 93, IntRange(-1))
//         .build();
// }

// 1.14.4 Forest Oak - seed  - tree seed 137138837835894 - tree decorator seeds 275040572347288 92469715419341
// __device__ constexpr TreeChunk get_input_data() {
//     // return TreeChunkBuilder(Version::v1_14_4, Biome::Forest)
//     //     .tree_birch(167, -14, IntRange(-1), "00?1""1001""01?1")
//     //     .tree_oak(166, -10, IntRange(-1), "10?1""01?1""10?0")
//     //     .tree_oak(173, -15, IntRange(-1), "00?0""11?1""11?0")
//     //     .tree_oak(169, -3, IntRange(6), "00??""11??""00?0")
//     //     .tree_oak(173, -3, IntRange(-1), "???0""?1?0""00?0")
//     //     .build();

//     return TreeChunkBuilder(Version::v1_14_4, Biome::Forest)
//         .tree_oak(143, -5, IntRange(-1), "?1?1""0010""00?0")
//         .tree_oak(138, -8, IntRange(-1), "???0""01?0""01?0")
//         .tree_oak(139, -11, IntRange(-1), "????""????""????")
//         .tree_birch(140, -14, IntRange(-1), "????""????""????")
//         .tree_fancy_oak(143, -14, IntRange(-1))
//         .build();
// }

// 1.8.9 Taiga - some random map from minecraft story mode - seed 2234065947811606375
// __device__ constexpr TreeChunk get_input_data() {
//     // return TreeChunkBuilder(Version::v1_8_9, Biome::Taiga)
//     //     .tree_unknown(11, 21)
//     //     // .tree_pine(11, 21, IntRange(-1), IntRange(-1), IntRange(-1))
//     //     // .tree_pine(11, 21, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(11, 18)
//     //     // .tree_spruce(11, 18, IntRange(-1), IntRange(-1), IntRange(-1), IntRange(1), IntRange(-1))
//     //     // .tree_spruce(11, 18, IntRange(-1), IntRange(-1), IntRange(2), IntRange(1), IntRange(-1))
//     //     .tree_unknown(14, 15)
//     //         .tree_spruce(14, 15, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(14, 15, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(10, 12)
//     //         .tree_spruce(10, 12, IntRange(-1), IntRange(-1), IntRange(2), IntRange(0), IntRange(-1))
//     //         .tree_pine(10, 12, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(12, 9)
//     //         .tree_spruce(12, 9, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(12, 9, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(19, 15)
//     //         .tree_spruce(19, 15, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(19, 15, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(22, 14)
//     //         .tree_spruce(22, 14, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(22, 14, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(22, 19)
//     //         .tree_spruce(22, 19, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(22, 19, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .build();

//     // bx 360 bz 152 - cx 22 cz 9
//     // return TreeChunkBuilder(Version::v1_8_9, Biome::Taiga)
//     //     .tree_pine(29, 23, IntRange(-1), IntRange(-1), IntRange(1))
//     //     .tree_spruce(36, 17, IntRange(-1), IntRange(-1), IntRange(2), IntRange(1), IntRange(-1))
//     //     .tree_spruce(38, 12, IntRange(-1), IntRange(-1), IntRange(2), IntRange(1), IntRange(-1))
//     //     .tree_unknown(39, 21)
//     //         .tree_spruce(39, 21, IntRange(-1), IntRange(-1), IntRange(3), IntRange(-1), IntRange(-1))
//     //         .tree_pine(39, 21, IntRange(-1), IntRange(-1), IntRange(3))
//     //     .tree_unknown(31, 19)
//     //         .tree_spruce(31, 19, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(31, 19, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(28, 15)
//     //         .tree_spruce(28, 15, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(28, 15, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(34, 11)
//     //         .tree_spruce(34, 11, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(34, 11, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .build();

//     // return TreeChunkBuilder(Version::v1_8_9, Biome::Taiga)
//     //     .tree_unknown(21, 54)
//     //         .tree_spruce(21, 54, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(21, 54, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(23, 50)
//     //         .tree_spruce(23, 50, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(23, 50, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(23, 45)
//     //         .tree_spruce(23, 45, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(23, 45, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(22, 40)
//     //         .tree_spruce(22, 40, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(22, 40, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(17, 46)
//     //         .tree_spruce(17, 46, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(17, 46, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(12, 43)
//     //         .tree_spruce(12, 43, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(12, 43, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .tree_unknown(8, 49)
//     //         .tree_spruce(8, 49, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//     //         .tree_pine(8, 49, IntRange(-1), IntRange(-1), IntRange(2))
//     //     .build();

//     // bx 360 bz 200 - cx 22 cz 12
//     return TreeChunkBuilder(Version::v1_8_9, Biome::Taiga)
//         .tree_pine(43, 48, IntRange(-1), IntRange(-1), IntRange(1))
//         .tree_unknown(54, 41)
//             .tree_spruce(54, 41, IntRange(-1), IntRange(-1), IntRange(3), IntRange(-1), IntRange(-1))
//             .tree_pine(54, 41, IntRange(-1), IntRange(-1), IntRange(3))
//         .tree_unknown(49, 42)
//             .tree_spruce(49, 42, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//             .tree_pine(49, 42, IntRange(-1), IntRange(-1), IntRange(2))
//         .tree_unknown(43, 43)
//             .tree_spruce(43, 43, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//             .tree_pine(43, 43, IntRange(-1), IntRange(-1), IntRange(2))
//         .tree_unknown(40, 45)
//             .tree_spruce(40, 45, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//             .tree_pine(40, 45, IntRange(-1), IntRange(-1), IntRange(2))
//         .tree_unknown(50, 49)
//             .tree_spruce(50, 49, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//             .tree_pine(50, 49, IntRange(-1), IntRange(-1), IntRange(2))
//         .tree_unknown(53, 55)
//             .tree_spruce(53, 55, IntRange(-1), IntRange(-1), IntRange(2), IntRange(-1), IntRange(-1))
//             .tree_pine(53, 55, IntRange(-1), IntRange(-1), IntRange(2))
//         .build();
// }