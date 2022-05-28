from Tool import FaceTool
import cv2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("GTK4Agg")


class NoseResizeTool(FaceTool):
    def __init__(self, img_path, faceDetector):
        """Image After Call Apply Will be in self.image

        Args:
            img_path (String): The Path of Image
            faceDetector (FaceDetaction): MediaPipe Moudle
        """
        self.img_path = img_path
        self.faceDetector = faceDetector
        self.image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        self.orig = self.image.copy()

    def apply(self, size=0.9,k=5, *args, **kwargs):
        """size: Scaling Factor of Nose
        \nk: Is For The Kernal of Gaussian Blur
        \nkwargs:
            \nFile: Path For The Image To Be Modifed.
            \nRadius: The Region Around The Nose Where All Processing Are Done.
        """
        if "File" in kwargs:
            self.image = cv2.cvtColor(cv2.imread(kwargs["File"]), cv2.COLOR_BGR2RGB)
        if "Radius" in kwargs:
            self.radius = kwargs["Radius"]
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

        h, w = face.shape[0], face.shape[1]
        self.radius = 160 if h > 800 or w > 800 else 50
        self.nose_p = self.__detect_nose_tip(face_upper, face_lower)
        self.__create_index_map(h, w)
        self.__edit_nose_area()
        self.__smothe_border(k=k)
        self.__remaping(face, face_upper, face_lower)

    def __detect_nose_tip(self, face_upper, face_lower):
        subface = self.image[
            face_upper[1] : face_lower[1], face_upper[0] : face_lower[0], :
        ].copy()
        indx = np.where(subface[:, :, 0] == 255)
        p = 0
        for x, y in zip(indx[0], indx[1]):
            r, g, b = subface[x, y, :]
            if r == 255 and g == 0 and b == 0:
                p = np.array([y, x])
        if type(p) is int:
            if p == 0:
                raise Exception("No Nose Point found for face")
        return p

    def __create_index_map(self, h, w):
        xs = np.arange(0, h, 1, dtype=np.float32)
        ys = np.arange(0, w, 1, dtype=np.float32)
        self.nose_map_x, self.nose_map_y = np.meshgrid(xs, ys)

    def __edit_nose_area(self):
        for i in range(-self.radius, self.radius):
            for j in range(-self.radius, self.radius):
                if i**2 + j**2 > self.radius**2:
                    continue
                if i > 0:
                    self.nose_map_y[self.nose_p[1] + i][self.nose_p[0] + j] = (
                        self.nose_p[1] + (i / self.radius) ** self.power * self.radius
                    )
                if i < 0:
                    self.nose_map_y[self.nose_p[1] + i][self.nose_p[0] + j] = (
                        self.nose_p[1] - (-i / self.radius) ** self.power * self.radius
                    )
                if j > 0:
                    self.nose_map_x[self.nose_p[1] + i][self.nose_p[0] + j] = (
                        self.nose_p[0] + (j / self.radius) ** self.power * self.radius
                    )
                if j < 0:
                    self.nose_map_x[self.nose_p[1] + i][self.nose_p[0] + j] = (
                        self.nose_p[0] - (-j / self.radius) ** self.power * self.radius
                    )

    def __smothe_border(self, k=5, xspace=10, yspace=10, sigmax=0):
        y, x = self.nose_p
        r = self.radius
        lU = [y - r - yspace, x - r - xspace]  # Left Upper
        rL = [y + r + yspace, x + r + xspace]  # Right Lower
        self.nose_map_x[lU[1] : rL[1], lU[0] : rL[0]] = cv2.GaussianBlur(
            self.nose_map_x[lU[1] : rL[1], lU[0] : rL[0]].copy(), (k,k), sigmax
        )
        self.nose_map_y[lU[1] : rL[1], lU[0] : rL[0]] = cv2.GaussianBlur(
            self.nose_map_y[lU[1] : rL[1], lU[0] : rL[0]].copy(), (k,k), sigmax
        )

    def __remaping(self, face, face_upper, face_lower):
        warped = cv2.remap(face, self.nose_map_x, self.nose_map_y, cv2.INTER_CUBIC)
        self.image[
            face_upper[1] : face_lower[1], face_upper[0] : face_lower[0], :
        ] = warped

    def show_result(self,axis=1):
        plt.imshow(np.concatenate((self.image, self.orig), axis=axis))
        plt.show()
