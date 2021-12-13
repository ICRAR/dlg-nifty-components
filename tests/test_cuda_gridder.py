import pytest
import numpy as np

from dlg.exceptions import DaliugeException
from dlg_nifty_components import CudaMS2DirtyApp, CudaDirty2MSApp
from dlg_nifty_components.droputils import save_numpy
from dlg.drop import InMemoryDROP

given = pytest.mark.parametrize


def test_CudaMS2DirtyApp_exceptions():
    app = CudaMS2DirtyApp("a", "a")

    with pytest.raises(DaliugeException) as e:
        app.run()


def test_CudaMS2DirtyApp():
    app = CudaMS2DirtyApp("a", "a")

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

    app.addOutput(InMemoryDROP("image", "image"))

    app.run()


def test_CudaDirty2MSApp_exceptions():
    app = CudaDirty2MSApp("a", "a")
    with pytest.raises(DaliugeException) as e:
        app.run()


def test_CudaDirty2MSApp():
    app = CudaDirty2MSApp("a", "a")

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
    image = np.zeros((num_rows, num_chan), dtype=np.float64)
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

    app.addOutput(InMemoryDROP("vis", "vis"))
