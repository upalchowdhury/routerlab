"""Run on Linux/NVIDIA after installing torch/triton. Skips on a CPU machine."""
import importlib.util
import unittest

import numpy as np

from routerlab.numerics import projection_oracle

TORCH = importlib.util.find_spec("torch") is not None
TRITON = importlib.util.find_spec("triton") is not None
if TORCH:
    import torch
GPU = bool(TORCH and TRITON and torch.cuda.is_available() and not torch.version.hip)


@unittest.skipUnless(GPU, "Requires CUDA-enabled PyTorch, Triton, and an NVIDIA GPU")
class GPUCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from routerlab.gpu_kernels import rowdot, tiled
        cls.backends = [rowdot, tiled]

    def test_irregular_shapes_and_strides(self):
        for m, h, e in [(1, 65, 17), (3, 257, 33), (8, 1024, 64)]:
            torch.manual_seed(0)
            x = torch.randn(m, h * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
            w = (torch.randn(e, h * 2, device="cuda", dtype=torch.bfloat16) / h**.5)[:, ::2]
            ref = projection_oracle(x.float().cpu().numpy(), w.float().cpu().numpy())
            for backend in self.backends:
                with self.subTest(shape=(m, h, e), backend=backend.__name__):
                    out = torch.full((m, e), float("nan"), device="cuda", dtype=torch.float32)
                    returned = backend(x, w, out)
                    self.assertEqual(returned.data_ptr(), out.data_ptr())
                    self.assertEqual(out.dtype, torch.float32)
                    np.testing.assert_allclose(out.cpu().numpy(), ref, rtol=1e-3, atol=1e-3)

    def test_zero_full_write(self):
        x = torch.zeros((1, 65), device="cuda", dtype=torch.bfloat16)
        w = torch.ones((17, 65), device="cuda", dtype=torch.bfloat16)
        for backend in self.backends:
            out = backend(x, w)
            self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_same_token_at_two_batch_sizes(self):
        x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16) / 32
        for backend in self.backends:
            torch.testing.assert_close(backend(x, w)[:1], backend(x[:1], w), rtol=1e-3, atol=1e-3)

    def test_cuda_graph_replay(self):
        x = torch.randn(2, 257, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(33, 257, device="cuda", dtype=torch.bfloat16) / 16
        for backend in self.backends:
            ref = projection_oracle(x.float().cpu().numpy(), w.float().cpu().numpy())
            out = torch.empty((2, 33), device="cuda", dtype=torch.float32)
            for _ in range(3):
                backend(x, w, out)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                backend(x, w, out)
            graph.replay()
            torch.cuda.synchronize()
            np.testing.assert_allclose(out.cpu().numpy(), ref, rtol=1e-3, atol=1e-3)
            # Replaying after changing values must compute new outputs, not replay values.
            x.add_(1)
            graph.replay()
            torch.cuda.synchronize()
            new_ref = projection_oracle(x.float().cpu().numpy(), w.float().cpu().numpy())
            np.testing.assert_allclose(out.cpu().numpy(), new_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
