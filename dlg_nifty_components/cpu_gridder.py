import ducc0
from dlg.exceptions import DaliugeException
from dlg.drop import BarrierAppDROP
from dlg.meta import (
    dlg_batch_input,
    dlg_batch_output,
    dlg_component,
    dlg_float_param,
    dlg_int_param,
    dlg_streaming_input,
    dlg_bool_param,
)

from dlg_nifty_components.droputils import save_numpy, load_numpy


##
# @brief MS2DirtyApp
# @details Converts measurement set data to a dirty image
# @par EAGLE_START
# @param category PythonApp
# @param requirements wagg/
# @param[in] param/appclass appclass/dlg_nifty_components.MS2DirtyApp/String/readonly/False/
#     \~English Application class
# @param[in] param/npix_x npix_x/64/Integer/readwrite/False/
#     \~English x dimensions of the dirty image
# @param[in] param/npix_y npix_y/64/Integer/readwrite/False/
#     \~English y dimensions of the dirty image
# @param[in] param/do_wstacking do_wstacking/True/Bool/readwrite/False/
#     \~English whether to perform wstacking
# @param[in] param/pixsize_x pixsize_x//Float/readwrite/False/
#     \~English pixel horizontal angular size in radians
# @param[in] param/pixsize_y pixsize_y//Float/readwrite/False/
#     \~English pixel vertical angular size in radians
# @param[in] port/uvw uvw/npy/
#     \~English uvw port
# @param[in] port/freq freq/npy/
#     \~English freq port
# @param[in] port/vis vis/npy/
#     \~English vis port
# @param[in] port/weight_spectrum weight_spectrum/npy/
#     \~English weight spectrum port
# @param[out] port/image image/npy/
#     \~English dirty image port
# @par EAGLE_END
class MS2DirtyApp(BarrierAppDROP):
    component_meta = dlg_component(
        "MS2DirtyApp",
        "Nifty MS2Dirty App.",
        [dlg_batch_input("binary/*", [])],
        [dlg_batch_output("binary/*", [])],
        [dlg_streaming_input("binary/*")],
    )
    npix_x = dlg_int_param("npix_x", 64)
    npix_y = dlg_int_param("npix_y", 64)
    do_wstacking = dlg_bool_param("do_wstacking", True)
    pixsize_x = dlg_float_param("pixsize_x", None)
    pixsize_y = dlg_float_param("pixsize_y", None)

    def run(self):
        if len(self.inputs) < 4:
            raise DaliugeException(
                f"MS2DirtyApp has {len(self.inputs)} input drops but requires at least 4"
            )
        uvw = load_numpy(self.inputs[0])
        freq = load_numpy(self.inputs[1])
        vis = load_numpy(self.inputs[2])
        weight_spectrum = load_numpy(self.inputs[3])
        epsilon = 1e-6  # unused

        if self.pixsize_x is None:
            self.pixsize_x = 1.0 / self.npix_x
        if self.pixsize_y is None:
            self.pixsize_y = 1.0 / self.npix_y

        image = ducc0.wgridder.ms2dirty(
            uvw,
            freq,
            vis,
            weight_spectrum,
            npix_x=self.npix_x,
            npix_y=self.npix_y,
            pixsize_x=self.pixsize_x,
            pixsize_y=self.pixsize_y,
            epsilon=epsilon,
            do_wstacking=self.do_wstacking,
        )

        save_numpy(self.outputs[0], image)


##
# @brief Dirty2MSApp
# @details Converts a dirty image to measurement set visibilities
# @par EAGLE_START
# @param category PythonApp
# @param[in] param/appclass appclass/dlg_nifty_components.Dirty2MSApp/String/readonly/False/
#     \~English Application class
# @param[in] param/do_wstacking do_wstacking/True/Bool/readwrite/False/
#     \~English whether to perform wstacking
# @param[in] param/pixsize_x pixsize_x//Float/readwrite/False/
#     \~English pixel horizontal angular size in radians
# @param[in] param/pixsize_y pixsize_y//Float/readwrite/False/
#     \~English pixel vertical angular size in radians
# @param[in] port/uvw uvw/npy/
#     \~English uvw port
# @param[in] port/freq freq/npy/
#     \~English freq port
# @param[in] port/image image/npy/
#     \~English dirty image port
# @param[in] port/weight_spectrum weight_spectrum/npy/
#     \~English weight spectrum port
# @param[out] port/vis vis/npy/
#     \~English vis port
# @par EAGLE_END
class Dirty2MSApp(BarrierAppDROP):
    component_meta = dlg_component(
        "Dirty2MSApp",
        "Nifty Dirty2MS App.",
        [dlg_batch_input("binary/*", [])],
        [dlg_batch_output("binary/*", [])],
        [dlg_streaming_input("binary/*")],
    )
    pixsize_x = dlg_float_param("pixsize_x", None)
    pixsize_y = dlg_float_param("pixsize_y", None)
    do_wstacking = dlg_bool_param("do_wstacking", None)

    def run(self):
        if len(self.inputs) < 4:
            raise DaliugeException(
                f"Dirty2MsApp has {len(self.inputs)} input drops but requires at least 4"
            )
        uvw = load_numpy(self.inputs[0])
        freq = load_numpy(self.inputs[1])
        dirty = load_numpy(self.inputs[2])
        weight_spectrum = load_numpy(self.inputs[3])
        epsilon = 1e-6  # unused

        if self.pixsize_x is None:
            self.pixsize_x = 1.0 / dirty.shape[0]
        if self.pixsize_y is None:
            self.pixsize_y = 1.0 / dirty.shape[1]

        vis = ducc0.wgridder.dirty2ms(
            uvw,
            freq,
            dirty,
            weight_spectrum,
            self.pixsize_x,
            self.pixsize_y,
            epsilon,
            self.do_wstacking,
        )

        save_numpy(self.outputs[0], vis)
