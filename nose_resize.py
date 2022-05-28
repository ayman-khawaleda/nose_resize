from Tool import FaceTool
import cv2
import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use("GTK4Agg")

class NoseResizeTool(FaceTool):
    def __init__(self, img_path, faceDetector):
        self.img_path = img_path
        self.faceDetector = faceDetector
        self.image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        self.orig = self.image.copy()

    def apply(self, size=0.9, *args, **kwargs):
        """size: Scaling Factor of Nose
        \nkwargs:
            \nFile: Path For The Image To Be Modifed.
            \nRadius: The Region Around The Eye Where All Processing Are Done.
        """
        if "File" in kwargs:
            self.image = cv2.cvtColor(cv2.imread(kwargs["File"]), cv2.COLOR_BGR2RGB)
        if "Radius" in kwargs:
            self.radius = kwargs["Radius"]
        self.power = size
        results = self.faceDetector.process(self.image)
        if not results:
            raise Exception("No Face was found")
        