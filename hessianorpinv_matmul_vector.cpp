#include <torch/extension.h>
#include <emmintrin.h>
#include <immintrin.h>

using namespace std;

// 海赛矩阵或其逆阵乘以一个列向量。
// 单层网络中，每个weight的海塞矩阵比较稀疏，但用稀疏矩阵存储仍超出内存使用限制，只有对角上的block有值，且这些小方阵值完全相同，其他位置均为0.
// 其伪逆阵也同样如此。
torch::Tensor hessianorpinv_matmul_vector_naive(const torch::Tensor& subblock, const torch::Tensor& vec, int num_block) {
    assert(subblock.dim() == 2 && vec.dim() == 2 && subblock.size(1) * num_block == vec.size(0) && vec.size(1) == 1);
    cout << "-O3" << endl;
    auto ret = torch::zeros_like(vec, torch::kFloat32);
    int64_t num_substride = subblock.size(0);
    for (int64_t i = 0; i < num_block; ++i) {
        // Print progress (optional)
        cout << i << "/" << num_block << std::endl;
        for (int64_t j = 0; j < num_substride; ++j) {
            for (int64_t k = 0; k < num_substride; ++k) {
                ret[i * num_substride + j][0] += subblock[j][k] * vec[i * num_substride + k][0];
            }
        }
    }
    return ret;
}

// avx2 simd优化
torch::Tensor hessianorpinv_matmul_vector_avx2(const torch::Tensor& subblock, const torch::Tensor& vec, int num_block) {
    assert(subblock.dim() == 2 && vec.dim() == 2 && subblock.size(1) * num_block == vec.size(0) && vec.size(1) == 1);
    int64_t num_substride = subblock.size(0);
    assert((num_substride % 16) == 0);
    auto ret = torch::zeros_like(vec, torch::kFloat32);
    float* pa = (float*)subblock.data_ptr(), *cur_pa = pa;
    float* pb = (float*)vec.data_ptr(), *cur_pb = pb;
    float* pc = (float*)ret.data_ptr(), *cur_pc = pc;
    for (int64_t i = 0; i < num_block; ++i) {
        // Print progress (optional)
        for (int64_t j = 0; j < num_substride; ++j) {
            for (int64_t k = 0; k < num_substride; ++k) {
                pc[i * num_substride + j] += pa[j * num_substride + k] * pb[i * num_substride + k];
            }
        }
    }
    return ret;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("hessianorpinv_matmul_vector", &hessianorpinv_matmul_vector_avx2, "Custom implementation of sparse diagonal hessian matrix(or its pseduo-inverse matrix) matmul a vector");
}
