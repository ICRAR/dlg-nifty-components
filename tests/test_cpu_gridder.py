import unittest
import pytest
import numpy as np

from dlg.droputils import DROPWaiterCtx
from dlg.exceptions import DaliugeException
from dlg.droputils import save_numpy, load_numpy
from dlg_nifty_components import MS2DirtyApp, Dirty2MSApp
from dlg.drop import InMemoryDROP

given = pytest.mark.parametrize

class CpuTests(unittest.TestCase):
    def test_MS2DirtyApp_exceptions(self):
        app = MS2DirtyApp("a", "a")

        with pytest.raises(DaliugeException) as e:
            app.run()

    def test_MS2DirtyApp(self):
        """
        Tests gridding from measurement set tables arrays to dirty image
        """
        app = MS2DirtyApp("a", "a")

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

        with DROPWaiterCtx(self, output, 5):
            uvw_drop.setCompleted()
            freq_drop.setCompleted()
            vis_drop.setCompleted()
            weight_spectrum_drop.setCompleted()

        dirty = load_numpy(output)
        assert dirty.shape == (64,64)

    def test_Dirty2MSApp_exceptions(self):
        """
        Test Dirt2MSApp raises exceptions when not configured
        """
        app = Dirty2MSApp("a", "a")
        with pytest.raises(DaliugeException):
            app.run()

    def test_Dirty2MSApp(self):
        """
        Test degridding from dirty image to measurement set table arrays
        """
        app = Dirty2MSApp("a", "a")

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
        image = np.zeros((image_size, image_size), dtype=np.float64)
        weight_spectrum = np.ones((num_rows, num_chan), dtype=np.float64)
        uvw = np.zeros((num_rows, 3), dtype=np.float64)

        # Generate synthetic data.
        for i in range(num_rows):
            image[i, 0] = 1
            uvw[i, 0] = (float(i) * image_size) / num_rows - image_size // 2
            uvw[i, 1] = (float(i) * image_size) / num_rows - image_size // 2
            uvw[i, 2] = 1.0

        uvw_drop = InMemoryDROP("uvw", "uvw")
        save_numpy(uvw_drop, uvw)
        app.addInput(uvw_drop)

        freq_drop = InMemoryDROP("freq", "freq")
        save_numpy(freq_drop, freq)
        app.addInput(freq_drop)

        image_drop = InMemoryDROP("image", "image")
        save_numpy(image_drop, image)
        app.addInput(image_drop)

        weight_spectrum_drop = InMemoryDROP("weight_spectrum", "weight_spectrum")
        save_numpy(weight_spectrum_drop, weight_spectrum)
        app.addInput(weight_spectrum_drop)

        vis_drop = InMemoryDROP("vis", "vis")
        app.addOutput(vis_drop)

        with DROPWaiterCtx(self, vis_drop, 5):
            uvw_drop.setCompleted()
            freq_drop.setCompleted()
            image_drop.setCompleted()
            weight_spectrum_drop.setCompleted()

        vis = load_numpy(vis_drop)
        assert vis.shape == (16,1)
