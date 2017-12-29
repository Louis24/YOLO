import sys
sys.path.insert(0, '/opt/ros/kinetic/lib/python2.7/dist-packages/darkflow')

from darkflow.net.build import TFNet
import cv2
import threading
import time
import rospy
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class EuclidObjectRecognizer():
    def __init__(self):
        self.detectionImage = self.image = None
        self.lastTime = time.time()
        self.elapsedTime = 1
        self.boxes = []

        script_path = os.path.dirname(os.path.realpath(__file__))

        self.options = {"model": os.path.join(script_path, "tiny-coco.cfg"), "load": os.path.join(script_path, "tiny-yolo.weights"), "threshold": 0.20, "config": script_path}
        self.tfnet = TFNet(self.options)
        self.colors = self.tfnet.meta['colors']
        self.classesColorMap = {}

        # Start ROS
        rospy.init_node("object_recognizer", anonymous=True)
        self.bridge = CvBridge()
        self.imagePub = rospy.Publisher("/euclid/object/live", Image)
        self.imageSub = rospy.Subscriber("/camera/color/image_raw", Image, self.newColorImage)
        self.rate = rospy.Rate(10)

        while self.image == None:
            self.rate.sleep()
        self.liveThread = threading.Thread(target=self.liveRecognitionThread)
        self.liveThread.start()
        self.mainThread()

    def newColorImage(self, imageMsg):
        self.image = cv2.cvtColor(self.bridge.imgmsg_to_cv2(imageMsg), cv2.COLOR_RGB2BGR)

    def getClassColor(self, className):
        if className in self.classesColorMap:
            return self.classesColorMap[className]
        self.classesColorMap[className] = self.colors[len(self.classesColorMap) + 10]

    def mainThread(self):
        h, w, _ = self.image.shape
        while not rospy.is_shutdown():
            self.detectionImage = self.image.copy()
            for bbox in self.boxes:
                left, top, right, bot, label = bbox['topleft']['x'], bbox['topleft']['y'], bbox['bottomright']['x'], bbox['bottomright']['y'], bbox['label']
                color = self.getClassColor(label)
                cv2.rectangle(self.detectionImage, (left, top), (right, bot), color, 3)
                cv2.putText(self.detectionImage, label, (left, top - 12), 0, 2e-3 * h, color, 1)
            self.imagePub.publish(self.bridge.cv2_to_imgmsg(self.detectionImage, "bgr8"))
            self.rate.sleep()

    def liveRecognitionThread(self):
        print("Starting live recognition")
        while not rospy.is_shutdown():
            self.lastTime = time.time()
            self.boxes = self.tfnet.return_predict(self.image)
            self.elapsedTime = time.time() - self.lastTime
        print("Live recognition Stopped")
if __name__ == "__main__":
    try:
        recgonizer = EuclidObjectRecognizer()
    except Exception as e:
        print(e)
        rospy.signal_shutdown()
