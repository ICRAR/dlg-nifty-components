import io
from dlg.io import OpenMode
import numpy as np


def save_numpy(drop, ndarray: np.ndarray, allow_pickle=False):
    """
    Saves a numpy ndarray to a drop
    """
    bio = io.BytesIO()
    np.save(bio, ndarray, allow_pickle=allow_pickle)
    drop.write(bio.getbuffer())


def load_numpy(drop, allow_pickle=False) -> np.ndarray:
    """
    Loads a numpy ndarray from a drop
    """
    dropio = drop.getIO()
    dropio.open(OpenMode.OPEN_READ)
    res = np.load(io.BytesIO(dropio.buffer()), allow_pickle=allow_pickle)
    dropio.close()
    return res
