#
#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This program is free software; you can redistribute it and/or
#    modify it under the terms of the GNU General Public License
#    as published by the Free Software Foundation; either version 2
#    of the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#    MA  02110-1301, USA.

import io
import numpy as np

import wagg
from dlg.exceptions import DaliugeException
from dlg.drop import BarrierAppDROP
from dlg.droputils import save_numpy, load_numpy
from dlg.meta import (dlg_batch_input, dlg_batch_output, dlg_component,
                      dlg_float_param, dlg_int_param, dlg_streaming_input,
                      dlg_string_param, dlg_bool_param)


##
# @brief CudaMS2DirtyApp
# @details CudaMS2DirtyApp
# @par EAGLE_START
# @param category PythonApp
# @param requirements wagg/
# @param[in] param/appclass appclass/dlg_nifty_components.CudaMS2DirtyApp/String/readonly/False/
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
# @param[in] port/uvw uvw/ndarray/
#     \~English uvw port
# @param[in] port/freq freq/ndarray/
#     \~English freq port
# @param[in] port/vis vis/ndarray/
#     \~English vis port
# @param[in] port/weight_spectrum weight_spectrum/ndarray/
#     \~English weight spectrum port
# @param[out] port/image image/ndarray/
#     \~English dirty image port
# @par EAGLE_END
class CudaMS2DirtyApp(BarrierAppDROP):
    component_meta = dlg_component('CudaMS2DirtyApp', 'Nifty Ms2Dirty App.',
                                    [dlg_batch_input('binary/*', [])],
                                    [dlg_batch_output('binary/*', [])],
                                    [dlg_streaming_input('binary/*')])
    npix_x = dlg_int_param('npix_x', 64)
    npix_y = dlg_int_param('npix_y', 64)
    do_wstacking = dlg_bool_param('do_wstacking', True)
    pixsize_x = dlg_float_param('pixsize_x', None)
    pixsize_y = dlg_float_param('pixsize_y', None)

    def run(self):
        if len(self.inputs) < 4:
            raise DaliugeException(f"CudaDirt2MsApp has {len(self.inputs)} input drops but requires at least 4")
        uvw = load_numpy(self.inputs[0])
        freq = load_numpy(self.inputs[1])
        vis = load_numpy(self.inputs[2])
        weight_spectrum = load_numpy(self.inputs[3])
        epsilon = 1e-6  # unused

        if self.pixsize_x == None:
            self.pixsize_x = 1.0 / self.npix_x
        if self.pixsize_y == None:
            self.pixsize_y = 1.0 / self.npix_y

        image = wagg.ms2dirty(uvw, freq, vis, weight_spectrum,
            self.npix_x, self.npix_y, self.pixsize_x, self.pixsize_y,
            epsilon, self.do_wstacking)

        save_numpy(self.outputs[0], image)


##
# @brief CudaDirty2MSApp
# @details CudaDirty2MSApp
# @par EAGLE_START
# @param category PythonApp
# @param[in] param/appclass appclass/dlg_nifty_components.CudaDirty2MSApp/String/readonly/False/
#     \~English Application class
# @param[in] param/do_wstacking do_wstacking/True/Bool/readwrite/False/
#     \~English whether to perform wstacking
# @param[in] param/pixsize_x pixsize_x//Float/readwrite/False/
#     \~English pixel horizontal angular size in radians
# @param[in] param/pixsize_y pixsize_y//Float/readwrite/False/
#     \~English pixel vertical angular size in radians
# @param[in] port/uvw uvw/ndarray/
#     \~English uvw port
# @param[in] port/freq freq/ndarray/
#     \~English freq port
# @param[in] port/image image/ndarray/
#     \~English dirty image port
# @param[in] port/weight_spectrum weight_spectrum/ndarray/
#     \~English weight spectrum port
# @param[out] port/vis vis/ndarray/
#     \~English vis port
# @par EAGLE_END
class CudaDirty2MSApp(BarrierAppDROP):
    component_meta = dlg_component('CudaDirty2MSApp', 'Nifty Ms2Dirty App.',
                                    [dlg_batch_input('binary/*', [])],
                                    [dlg_batch_output('binary/*', [])],
                                    [dlg_streaming_input('binary/*')])
    pixsize_x = dlg_float_param('pixsize_x', None)
    pixsize_y = dlg_float_param('pixsize_y', None)
    do_wstacking = dlg_bool_param('do_wstacking', None)

    def run(self):
        if len(self.inputs) < 4:
            raise DaliugeException(f"CudaDirt2MsApp has {len(self.inputs)} input drops but requires at least 4")
        uvw = load_numpy(self.inputs[0])
        freq = load_numpy(self.inputs[1])
        dirty = load_numpy(self.inputs[2])
        weight_spectrum = load_numpy(self.inputs[3])
        epsilon = 1e-6  # unused

        if self.pixsize_x == None:
            self.pixsize_x = 1.0 / dirty.shape[0]
        if self.pixsize_y == None:
            self.pixsize_y = 1.0 / dirty.shape[1]

        vis = wagg.dirty2ms(uvw, freq, dirty, weight_spectrum,
            self.pixsize_x, self.pixsize_y, epsilon, self.do_wstacking)

        save_numpy(self.outputs[0], vis)


##
# @brief CudaNiftyApp
# @details A gridder and degridder APP of ska-gridder-nifty-cuda that produces
# a gridded image from visibilities and a set of degridded visibilities of the image.
# @par EAGLE_START
# @param category PythonApp
# @param requirements wagg/
# @param[in] param/appclass appclass/dlg_nifty_components.CudaNiftyApp/String/readonly/False/
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
# @param[in] param/polarization polarization/0/Integer/readwrite/False/
#     \~English polarization to perform gridding to
# @param[in] port/uvw uvw/ndarray/
#     \~English Port containing UVWs of shape (baselines, 3)
# @param[in] port/freq freq/ndarray/
#     \~English Port containing channel frequencies in Hz 
# @param[in] port/vis vis/ndarray/
#     \~English Port containing visibilities of shape (baselines, channels, pols)
# @param[in] port/weight_spectrum weight_spectrum/ndarray/
#     \~English Port containing weight spectrum of shape (baselines, channels)
# @param[out] port/image image/ndarray/
#     \~English Port carrying the output dirty image of shape (npix_x, npix_y)
# @param[out] port/vis vis/ndarray/
#     \~English Port carrying output degridded visibilities of shape (baselines, channels, pols)
# @par EAGLE_END
class CudaNiftyApp(BarrierAppDROP):
    component_meta = dlg_component('CudaNiftyApp', 'Cuda Nifty App.',
                                    [dlg_batch_input('binary/*', [])],
                                    [dlg_batch_output('binary/*', [])],
                                    [dlg_streaming_input('binary/*')])
    npix_x = dlg_int_param('npix_x', 64)
    npix_y = dlg_int_param('npix_y', 64)
    do_wstacking = dlg_bool_param('do_wstacking', None)
    pixsize_x = dlg_float_param('pixsize_x', None)
    pixsize_y = dlg_float_param('pixsize_y', None)
    polarization = dlg_int_param('polarization', 0)

    def run(self):
        if len(self.inputs) < 4:
            raise DaliugeException(f"CudaDirt2MsApp has {len(self.inputs)} input drops but requires at least 4")

        if self.pixsize_x == None:
            self.pixsize_x = 1.0 / self.npix_x
        if self.pixsize_y == None:
            self.pixsize_y = 1.0 / self.npix_y
        epsilon = 1e-10

        uvw = load_numpy(self.inputs[0])
        freq = load_numpy(self.inputs[1])
        vis = load_numpy(self.inputs[2])
        weight_spectrum = load_numpy(self.inputs[3])

        image_dirty = wagg.ms2dirty(uvw, freq, vis, weight_spectrum,
            self.npix_x, self.npix_y, self.pixsize_x, self.pixsize_y,
            epsilon, self.do_wstacking)
        save_numpy(self.outputs[0], image_dirty)

        vis_degridded = wagg.dirty2ms(uvw, freq, image_dirty, weight_spectrum,
            self.pixsize_x, self.pixsize_y, epsilon, self.do_wstacking)
        vis[:,:,self.polarization] = vis_degridded
        save_numpy(self.outputs[1], vis)