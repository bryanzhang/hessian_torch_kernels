#include <torch/extension.h>
#include <emmintrin.h>
#include <immintrin.h>

using namespace std;
namespace py = pybind11;

// 海赛矩阵或其逆阵乘以一个列向量。
// 单层网络中，每个weight的海塞矩阵比较稀疏，但用稀疏矩阵存储仍超出内存使用限制，只有对角上的block有值，且这些小方阵值完全相同，其他位置均为0.
// 其伪逆阵也同样如此。
// 必须确保所有的tensor都是连续的
torch::Tensor hessianorpinv_matmul_vector_naive(const torch::Tensor& subblock, const torch::Tensor& vec, int num_block) {
    assert(subblock.dim() == 2 && vec.dim() == 2 && subblock.size(1) * num_block == vec.size(0) && vec.size(1) == 1);
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

// avx simd优化
// 必须确保所有的tensor都是连续的
torch::Tensor hessianorpinv_matmul_vector_avx2(const torch::Tensor& subblock, const torch::Tensor& vec, int num_block) {
    assert(subblock.dim() == 2 && vec.dim() == 2 && subblock.size(1) * num_block == vec.size(0) && vec.size(1) == 1);
    int64_t num_substride = subblock.size(0);
    assert((num_substride % 16) == 0);
    auto ret = torch::zeros_like(vec, torch::kFloat32);
    float* pa = (float*)subblock.data_ptr();
    float* pb = (float*)vec.data_ptr();
    float* pc = (float*)ret.data_ptr();
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

// 必须确保使用的tensor都是连续的
torch::Tensor hessianorpinv_matmul_vector_v1(const py::list& py_list, const torch::Tensor& vec) {
  vector<torch::Tensor> tensors;
  for (auto item : py_list) {
	  tensors.push_back(py::cast<torch::Tensor>(item));
  }
  auto ret = torch::zeros_like(vec, torch::kFloat32);
  float* pc = (float*)ret.data_ptr();
  int num_block = tensors.size();
  int row_base = 0;
  float* pb = (float*)vec.data_ptr();
  int num_substride = 0;
  for (int64_t i = 0; i < num_block; ++i, row_base += num_substride) {
	  auto& tensor = tensors.at(i);
	  num_substride = tensor.size(0);
	  assert(tensor.dim() == 2 && tensor.size(1) == num_substride);
  	  float* pa = (float*)tensor.data_ptr();
	  for (int64_t j = 0; j < num_substride; ++j) {
		  for (int k = 0; k < num_substride; ++k) {
			  pc[row_base + j] += pa[j * num_substride + k] * pb[row_base + k];
		  }
	  }
  }
  assert(vec.dim() == 2 && row_base == vec.size(0));
  return ret;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("hessianorpinv_matmul_vector", &hessianorpinv_matmul_vector_avx2, "Custom implementation of sparse diagonal hessian matrix(or its pseduo-inverse matrix) matmul a vector");
	m.def("hessianorpinv_matmul_vector_v1", &hessianorpinv_matmul_vector_v1, "Custom implementation of sparse diagonal hessian matrix(or its pseduo-inverse matrix) matmul a vector");
}
