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
from dlg.droputils import save_numpy, load_numpy


##
# @brief MS2DirtyApp
# @details Converts measurement set data to a dirty image
# @par EAGLE_START
# @param category PythonApp
# @param[in] cparam/appclass appclass/dlg_nifty_components.MS2DirtyApp/String/readonly/False//False/
#     \~English Application class
# @param[in] cparam/execution_time Execution Time/5/Float/readonly/False//False/
#     \~English Estimated execution time
# @param[in] cparam/num_cpus No. of CPUs/1/Integer/readonly/False//False/
#     \~English Number of cores used
# @param[in] cparam/group_start Group start/False/Boolean/readwrite/False//False/
#     \~English Is this node the start of a group?
# @param[in] cparam/input_error_threshold "Input error rate (%)"/0/Integer/readwrite/False//False/
#     \~English the allowed failure rate of the inputs (in percent), before this component goes to ERROR state and is not executed
# @param[in] cparam/n_tries Number of tries/1/Integer/readwrite/False//False/
#     \~English Specifies the number of times the 'run' method will be executed before finally giving up
# @param[in] cparam/npix_x npix_x/64/Integer/readwrite/False//False/
#     \~English x dimensions of the dirty image
# @param[in] cparam/npix_y npix_y/64/Integer/readwrite/False//False/
#     \~English y dimensions of the dirty image
# @param[in] cparam/do_wstacking do_wstacking/True/Boolean/readwrite/False//False/
#     \~English whether to perform wstacking
# @param[in] cparam/pixsize_x pixsize_x//Float/readwrite/False//False/
#     \~English pixel horizontal angular size in radians
# @param[in] cparam/pixsize_y pixsize_y//Float/readwrite/False//False/
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
    npix_x: int = dlg_int_param("npix_x", 64)  # type: ignore
    npix_y: int = dlg_int_param("npix_y", 64)  # type: ignore
    do_wstacking: bool = dlg_bool_param("do_wstacking", True)  # type: ignore
    pixsize_x: float = dlg_float_param("pixsize_x", None)  # type: ignore
    pixsize_y: float = dlg_float_param("pixsize_y", None)  # type: ignore

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
# @param[in] cparam/appclass appclass/dlg_nifty_components.Dirty2MSApp/String/readonly/False//False/
#     \~English Application class
# @param[in] cparam/execution_time Execution Time/5/Float/readonly/False//False/
#     \~English Estimated execution time
# @param[in] cparam/num_cpus No. of CPUs/1/Integer/readonly/False//False/
#     \~English Number of cores used
# @param[in] cparam/group_start Group start/False/Boolean/readwrite/False//False/
#     \~English Is this node the start of a group?
# @param[in] cparam/input_error_threshold "Input error rate (%)"/0/Integer/readwrite/False//False/
#     \~English the allowed failure rate of the inputs (in percent), before this component goes to ERROR state and is not executed
# @param[in] cparam/n_tries Number of tries/1/Integer/readwrite/False//False/
#     \~English Specifies the number of times the 'run' method will be executed before finally giving up
# @param[in] cparam/do_wstacking do_wstacking/True/Boolean/readwrite/False//False/
#     \~English whether to perform wstacking
# @param[in] cparam/pixsize_x pixsize_x//Float/readwrite/False//False/
#     \~English pixel horizontal angular size in radians
# @param[in] cparam/pixsize_y pixsize_y//Float/readwrite/False//False/
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
    do_wstacking: bool = dlg_bool_param("do_wstacking", None)  # type: ignore
    pixsize_x: float = dlg_float_param("pixsize_x", None)  # type: ignore
    pixsize_y: float = dlg_float_param("pixsize_y", None)  # type: ignore

    def run(self):
        if len(self.inputs) < 4:
            raise DaliugeException(
                f"Dirty2MSApp has {len(self.inputs)} input drops but requires at least 4"
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
            uvw=uvw,
            freq=freq,
            dirty=dirty,
            wgt=weight_spectrum,
            pixsize_x=self.pixsize_x,
            pixsize_y=self.pixsize_y,
            epsilon=epsilon,
            do_wstacking=self.do_wstacking,
        )

        save_numpy(self.outputs[0], vis)
