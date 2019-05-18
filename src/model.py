import os
import tarfile
import numpy as np
import tensorflow as tf
from PIL import Image


class DeepLabModel(object):
    """Class for loading and inference"""

    INPUT_TENSOR = "ImageTensor:0"
    OUTPUT_TENSOR = "SemanticPredictions:0"
    DEFAULT_IMAGE_SIZE = 513

    def __init__(self, tar_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        gdef = None
        # Extract frozen graph from tar archive.
        tar = tarfile.open(tar_path)
        for tar_info in tar.getmembers():
            if "frozen_inference_graph" in os.path.basename(tar_info.name):
                handle = tar.extractfile(tar_info)
                gdef = tf.GraphDef.FromString(handle.read())
                break
        tar.close()

        if not gdef:
            raise RuntimeError("Invalid saved model!")
        with self.graph.as_default():
            tf.import_graph_def(gdef, name="")

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Inference modle with one image

    @param: image in numpy arrays
    @return: raw image resized
    @return: segmantation of resized image
    """
        w, h = image.size
        resized = image.convert("RGB").resize(
            (
                int(float(self.DEFAULT_IMAGE_SIZE / max(w, h)) * w),
                int(float(self.DEFAULT_IMAGE_SIZE / max(w, h)) * h),
            ),
            Image.ANTIALIAS,
        )
        result = self.sess.run(
            self.OUTPUT_TENSOR, feed_dict={self.INPUT_TENSOR: [np.asarray(resized)]}
        )
        seg = result[0]  # will always be 0
        return resized, seg
