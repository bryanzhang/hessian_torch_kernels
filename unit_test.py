#! /usr/bin/python3

import torch
import hessianorpinv_matmul_vector as custom_kernels
import unittest
import time

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

if __name__ == "__main__":
    unittest.main()
