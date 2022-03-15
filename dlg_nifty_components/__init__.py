# extend the following as required
from dlg_nifty_components.cpu_gridder import MS2DirtyApp, Dirty2MSApp
__all__ = ["MS2DirtyApp", "Dirty2MSApp"]

try:
    from dlg_nifty_components.cuda_gridder import CudaMS2DirtyApp, CudaDirty2MSApp
    __all__.extend(["CudaMS2DirtyApp", "CudaDirty2MSApp"])
except ImportError:
    # cuda components unavailable
    pass
