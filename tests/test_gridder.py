import unittest
import pytest
import numpy as np

from dlg.droputils import DROPWaiterCtx
from dlg.exceptions import DaliugeException
from dlg.droputils import save_numpy, load_numpy
from dlg_nifty_components import MS2DirtyApp, NiftyGridderApp
from dlg.drop import InMemoryDROP

try:
    import cuda_nifty_gridder
    from dlg_nifty_components.cuda_gridder import CudaMS2DirtyApp
    cuda_enabled = True
except ImportError:
    cuda_enabled = False


class CpuTests(unittest.TestCase):

    def _test_gridder(self, app):
        """
        Tests gridding from measurement set tables arrays to dirty image
        """
        # Test data dimensions.
        num_rows = 16
        num_chan = 1
        image_size = 64

        # Create the frequency axis.
        freq_start_hz = 299792458.0
        freq_inc_hz = 1.0
        freq = np.linspace(
            freq_start_hz, freq_start_hz + (num_chan - 1) * freq_inc_hz, num_chan
        )

        # Allocate input arrays.
        vis = np.zeros((num_rows, num_chan), dtype=np.complex128)
        weight_spectrum = np.ones((num_rows, num_chan), dtype=np.float64)
        uvw = np.zeros((num_rows, 3), dtype=np.float64)

        # Generate synthetic data.
        for i in range(num_rows):
            vis[i, 0] = 1 + 1j * (i + 1) / 10
            uvw[i, 0] = (float(i) * image_size) / num_rows - image_size // 2
            uvw[i, 1] = (float(i) * image_size) / num_rows - image_size // 2
            uvw[i, 2] = 1.0

        uvw_drop = InMemoryDROP("uvw", "uvw")
        save_numpy(uvw_drop, uvw)
        app.addInput(uvw_drop)

        freq_drop = InMemoryDROP("freq", "freq")
        save_numpy(freq_drop, freq)
        app.addInput(freq_drop)

        vis_drop = InMemoryDROP("vis", "vis")
        save_numpy(vis_drop, vis)
        app.addInput(vis_drop)

        weight_spectrum_drop = InMemoryDROP("weight_spectrum", "weight_spectrum")
        save_numpy(weight_spectrum_drop, weight_spectrum)
        app.addInput(weight_spectrum_drop)

        output = InMemoryDROP("image", "image")
        app.addOutput(output)

        # daliuge executes drops on a seperate threads
        #app.run()
        with DROPWaiterCtx(self, output, 5):
            uvw_drop.setCompleted()
            freq_drop.setCompleted()
            vis_drop.setCompleted()
            weight_spectrum_drop.setCompleted()

        dirty = load_numpy(output)
        assert dirty.shape == (64, 64)

    def test_MS2DirtyApp_exceptions(self):
        """doc"""
        app = MS2DirtyApp("a", "a")
        with pytest.raises(DaliugeException):
            app.run()

    def test_MS2DirtyApp(self):
        """doc"""
        app = MS2DirtyApp("a", "a")
        self._test_gridder(app)

    def test_NiftyGridderApp_cpu(self):
        """doc"""
        app = NiftyGridderApp("a", "a", cuda_enabled=False)
        self._test_gridder(app)

    @unittest.skipIf(cuda_enabled is False, "ska-gridder-nifty-cuda not installed")
    def test_CudaMS2DirtyApp_exceptions(self):
        """Tests that component raises exception on invalid configurations"""
        app = CudaMS2DirtyApp("a", "a")
        with pytest.raises(DaliugeException):
            app.run()

    @unittest.skipIf(cuda_enabled is False, "ska-gridder-nifty-cuda not installed")
    def test_CudaMS2DirtyApp(self):
        """doc"""
        app = CudaMS2DirtyApp("a", "a")
        self._test_gridder(app)

    @unittest.skipIf(cuda_enabled is False, "ska-gridder-nifty-cuda not installed")
    def test_NiftyGridderApp_cuda(self):
        """doc"""
        app = NiftyGridderApp("a", "a", cuda_enabled=True)
        self._test_gridder(app)
