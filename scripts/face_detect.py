import cv2.cv2 as cv2
import os

SHAPE = (512, 512)
TEST_PATH = "DSC_0019.JPG"

# Reads haarcascade xml file from cv2
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, "data/haarcascade_frontalface_default.xml")


class FaceDetector:
    """
    Wrapper class which manages the following:
    1. Takes input as a image path and shape
    2. Stores image as grayscale and in RGB format.
    3. Implements face detection from image using opencv haar-cascade
    4. Display image
    """

    def show_image(self, img=None):
        """
        Takes an image as an arguement and shows the image. If no image is passed
        then shows image passed to it during construction of object
        """

        print("Displaying image...")
        if img is None:
            cv2.imshow("image", self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def read_image(self, path=None, shape=SHAPE):
        """
        Specify the path arguement to read the image using cv2.imread.
        This resizes the image to 512,512 pixels
        """
        if path is None:
            print("Enter a path.")
            return None

        # Try except clause for to catch invalid paths
        try:
            img = cv2.imread(path)
            img = cv2.resize(img, SHAPE)
            return img
        except:
            print("Enter valid path")
            return None

    def __init__(self, path, shape):
        super().__init__()
        # reads image using imread
        print("Reading Image..")
        self.img = self.read_image(path, shape)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def detect_faces(self):
        # instantiates face_cascade model for prediction
        face_cascade = cv2.CascadeClassifier(haar_model)

        # Find faces
        # First parameter after image is Scale factor
        # Second parameter is minNeighbours
        # https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
        print("Detecting faces...")
        faces = face_cascade.detectMultiScale(self.gray, 1.1, 4)

        self.faces = faces

    def draw_boxes(self, faces=None):
        """
        Draws boxes on faces in the image, stores it in a variable faces and shows the marked faces.
        If the object has no faces then it can be used on any image.
        """
        temp = self.img.copy()

        self.detect_faces()
        # Draw rectangle around the faces

        print("Marking Boxes...")
        if faces is None:
            for (x, y, w, h) in self.faces:
                cv2.rectangle(temp, (x, y), (x + w, y + h), (255, 0, 0), 2)

        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(temp, (x, y), (x + w, y + h), (255, 0, 0), 2)

        self.show_image(temp)

    def get_cropped(self):
        """
        Gets cropped faces from image.
        """
        self.cropped = []

        self.detect_faces()

        for (x, y, w, h) in self.faces:
            # Here we have added and subtracted 20 to get full face
            # Normalise picture
            self.cropped.append(
                cv2.resize(self.img[y - 20 : y + h + 20, x : x + w], (80, 100)) / 255.0
            )

        # self.names = []
        # for face in self.cropped:

        #     self.show_image(face)
        #     self.names.append(
        #         input("Close the window and then enter name of this person : ")
        #     )
        # turned off for faster debugging
        # for image in self.cropped:
        #     self.show_image(image)


# test = FaceDetector(TEST_PATH, SHAPE)
# test.get_cropped()