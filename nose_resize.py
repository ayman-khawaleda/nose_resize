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
        else:
            self.radius = 170 if h > 800 or w > 800 else 45
        self.power = size
        results = self.faceDetector.process(self.image)
        if not results:
            raise Exception("No Face was found")
        rows, cols, _ = self.image.shape
        rbb = results.detections[0].location_data.relative_bounding_box
        nose_p = results.detections[0].location_data.relative_keypoints[2]

        nose_p = self.normaliz_pixel(nose_p.x, nose_p.y, cols, rows)
        face_upper = self.normaliz_pixel(rbb.xmin, rbb.ymin, cols, rows)
        face_lower = self.normaliz_pixel(
            rbb.xmin + rbb.width, rbb.ymin + rbb.height, cols, rows
        )
        face = self.image[
            face_upper[1] : face_lower[1], face_upper[0] : face_lower[0], :
        ].copy()

        self.image[nose_p[1], nose_p[0], :] = (255, 0, 0)
        face2 = face.copy()
        h,w = face.shape[0] ,face.shape[1]
        nose_p = self.detect_nose_tip(face2)
        self.creat_index_map(h,w)
        
    def detect_nose_tip(self,subface):
        indx = np.where(subface[:,:,0] == 255)
        p = 0
        for x,y in zip(indx[0],indx[1]):
            r,g,b = subface[x,y,:]
            if r == 255 and g == 0 and b == 0:
                p = (y,x)
        if p == 0:
            raise Exception("No Nose Point found for face")
        return p

    def create_index_map(self, h, w):
        xs = np.arange(0,h,1,dtype=np.float32)
        ys = np.arange(0,w,1,dtype=np.float32)
        self.nose_map_x, self.nose_map_y = np.meshgrid(xs,ys)