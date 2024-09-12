from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
        name='hessianorpinv_matmul_vector',
        ext_modules=[
            CppExtension('hessianorpinv_matmul_vector', ['hessianorpinv_matmul_vector.cpp'], extra_compile_args=['-std=c++17', '-O3', '-mavx2']),
        ],
        cmdclass={
            'build_ext' : BuildExtension
        },
        # 定义 TORCH_EXTENSION_NAME 宏
        define_macros = [('TORCH_EXTENSION_NAME', 'hessianorpinv_matmul_vector')]
)
