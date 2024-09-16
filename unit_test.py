#! /usr/bin/python3

import torch
import hessianorpinv_matmul_vector as custom_kernels
import unittest
import time
import random

class TestKernel(unittest.TestCase):
    def test_simple(self):
        subblock = torch.tensor([[1,2,3],[2,5,4],[3,4,6]], dtype=torch.float)
        vector = torch.tensor([[0],[1],[0],[0],[2],[0],[-1],[4],[0],[3],[0],[-3]], dtype=torch.float)
        actual = custom_kernels.hessianorpinv_matmul_vector(subblock, vector, 4)
        print("Actual result: ", actual)
        zeroblock = torch.zeros_like(subblock, dtype=torch.float)
        dense = torch.cat((torch.cat((subblock, zeroblock, zeroblock, zeroblock), dim=1), torch.cat((zeroblock, subblock, zeroblock, zeroblock), dim=1), torch.cat((zeroblock, zeroblock, subblock, zeroblock), dim=1), torch.cat((zeroblock, zeroblock, zeroblock, subblock), dim=1)), dim=0)
        print("Dense shape: ", dense.size())
        expected = torch.matmul(dense, vector)
        print("Expected result: ", expected)
        self.assertTrue(torch.equal(actual, expected))

    def test_medium(self):
        subblock = torch.randn((256, 256), dtype=torch.float)
        vector = torch.randn((256*64, 1), dtype=torch.float)
        start_time = time.time()
        result = custom_kernels.hessianorpinv_matmul_vector(subblock, vector, 64)
        end_time = time.time()
        print("Hessian matmul kernel medium execution time: ", (end_time - start_time), "seconds.")

    def test_huge(self):
        subblock = torch.randn((784, 784), dtype=torch.float)
        vector = torch.randn((784*512, 1), dtype=torch.float)
        start_time = time.time()
        result = custom_kernels.hessianorpinv_matmul_vector(subblock, vector, 512)
        end_time = time.time()
        print("Hessian matmul kernel huge execution time: ", (end_time - start_time), "seconds.")

    def test_v1_simple(self):
        block1 = torch.tensor([[8,12],[12,18]], dtype=torch.float)
        block2 = torch.tensor([[2]], dtype=torch.float)
        l = [block1, block2]
        vector = torch.tensor([[0.4], [0.6], [0.8]], dtype=torch.float)
        actual = custom_kernels.hessianorpinv_matmul_vector_v1(l, vector)
        zblock1 = torch.zeros((2, 1), dtype=torch.float)
        zblock2 = torch.zeros((1, 2), dtype=torch.float)
        print("Actual result:", actual)
        dense = torch.cat((torch.cat((block1, zblock1), dim=1), torch.cat((zblock2, block2), dim=1)), dim=0)
        print("Dense shape: ", dense.size())
        expected = torch.matmul(dense, vector)
        print("Expected result: ", expected)
        self.assertTrue(torch.equal(actual, expected))

    def test_v1_medium(self):
        l = []
        n = 0
        for i in range(0, 64):
            lb = int(256 * 0.05)
            ub = int(256 * 0.15)
            stride = random.randint(lb, ub)
            n += stride
            subblock = torch.randn((stride, stride), dtype=torch.float)
            l.append(subblock)
        vector = torch.randn((n, 1), dtype=torch.float)
        start_time = time.time()
        result = custom_kernels.hessianorpinv_matmul_vector_v1(l, vector)
        end_time = time.time()
        print("Hessian matmul v1 kernel medium execution time: ", (end_time - start_time), "seconds.")

    def test_v1_huge(self):
        l = []
        n = 0
        for i in range(0, 512):
            lb = int(784 * 0.05)
            ub = int(784 * 0.15)
            stride = random.randint(lb, ub)
            n += stride
            subblock = torch.randn((stride, stride), dtype=torch.float)
            l.append(subblock)
        vector = torch.randn((n, 1), dtype=torch.float)
        start_time = time.time()
        result = custom_kernels.hessianorpinv_matmul_vector_v1(l, vector)
        end_time = time.time()
        print("Hessian matmul v1 kernel huge execution time: ", (end_time - start_time), "seconds.")

if __name__ == "__main__":
    unittest.main()
