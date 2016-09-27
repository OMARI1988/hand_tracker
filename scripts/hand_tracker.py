#! /usr/bin/env python2.7

import roslib
roslib.load_manifest('hand_tracker')
import rospy
import sensor_msgs.msg
from cv_bridge import CvBridge
import cv2
import numpy as np
import pickle

# Core stuff, like containers.
import PyMBVCore as Core
# Image acquisition.
import PyMBVAcquisition as Acquisition
# 3D Multi-hypothesis rendering.
import PyMBVRendering as Rendering
# Conversion of hypotheses to 3D renderables.
import PyMBVDecoding as dec
# A library which puts together the aforementioned
# and some extras to make up 3D hand tracking.
import PyHandTracker as HT
# Timing.
from time import clock
import time


#--------------------------------------------------------------------------------------#
class hand_tracking():
    def __init__(self):
        self._create_renderer()                      # initialize renderer
        self.cv_bridge = CvBridge()	                # initilize opencv
        xtion_rgb_topic = rospy.resolve_name("/camera/rgb/image_raw")
        rospy.Subscriber(xtion_rgb_topic, sensor_msgs.msg.Image, self._xtion_rgb)
        xtion_depth_topic = rospy.resolve_name("/camera/depth/image_raw")
        rospy.Subscriber(xtion_depth_topic, sensor_msgs.msg.Image, self.depth_callback)
        xtion_info_topic = rospy.resolve_name("/camera/rgb/camera_info")
        rospy.Subscriber(xtion_info_topic, sensor_msgs.msg.CameraInfo, self._xtion_info)

    def _xtion_rgb(self,imgmsg):
        img = self.cv_bridge.imgmsg_to_cv2(imgmsg, desired_encoding="passthrough")
        self.xtion_img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        # cv2.imshow('xtion rgb',self.xtion_img)
        # k = cv2.waitKey(1) & 0xff

    def depth_callback(self, msg):
        # self.depth_msg = msg
        depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_array = np.array(depth_image, dtype=np.ushort)
        # cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
        # print np.max(depth_array)
        self.xtion_img_d = depth_array
        # self.xtion_img_d = depth_array
        # if self._flag_depth is 0:
            # print ' >depth image received'
            # self._flag_depth = 1

    # def _xtion_depth(self,imgmsg):
    #     self.xtion_img_d = self.cv_bridge.imgmsg_to_cv2(imgmsg, desired_encoding="passthrough")
    #     # self.xtion_img_d.setflags(write=True)                   # allow to change the values
    #     print self.xtion_img_d
        # self.xtion_img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        # cv2.imshow('xtion depth',self.xtion_img_d)
        # k = cv2.waitKey(1) & 0xff
        # print '##################################################'

    def _xtion_info(self,msg):
        self.clbs = msg
        # print '##################################################'

    def _create_renderer(self):
        print "Creating Renderer..."
        # Turn off logging
        Core.InitLog(['handTracker', 'log.severity', 'error'])

        # The 3D renderer is a singleton. The single instance is accessed.
        renderer = Rendering.RendererOGLCudaExposed.get()
        # OpenCV coordinate system is right handed but the renderer's
        # coordinate system is left handed. Conversion is handled, but
        # in the process front facing triangles become back facing triangles.
        # Thus, we inverse the culling order.
        # Try to set it to CullBack or CullNone to see the differences.
        renderer.culling = Rendering.RendererOGLBase.Culling.CullFront

        # An exposed renderer is one whose data are exposed through
        # some API. The hand tracker lib requires such a renderer.
        erenderer = Rendering.ExposedRenderer(renderer, renderer)

        # Create the hand tracker lib
        # params:
        #   - width (2048): max width preallocated for rendering
        #   - height (2048): max height preallocated for rendering
        #   - tileWidth (64): width of hypothesis rendering tile
        #   - tileHeight (64): height of hypothesis rendering tile
        # With the given parameter the handtracker lib will be able to
        # render at most (2048/64)x(2048x64)=1024 hypotheses in parallel.
        # The greatest this number the more the hypothesis evaluation
        # throughput. Default optimization only requires to render 64
        # hypotheses at a time.

        self.ht = HT.HandTrackerLib(2048, 2048, 128, 128, erenderer) #note: cd ~/catkin_ws/src/hand_tracker and then run

        # Create a decoder, i.e. an object which can transform
        # 27-D parameter vectors to 3D renderable hands.
        handDec = dec.GenericDecoder()
        # A description for a hand can be found at a file.
        handDec.loadFromFile("media/hand_right_low_RH.xml")
        # Set the decoder to the hand tracker lib.
        self.ht.decoder = handDec

        # Setup randomization variances to use during heuristic search.
        posvar = [10, 10, 10]               # 3D global translation variance
        rotvar = [0.1, 0.1, 0.1, 0.1]       # Quaternion global rotation variance
        fingervar = [ 0.1, 0.1, 0.1, 0.1]   # Per finger relative angles variance

        # 27-D = 3D position + 4D rotation + 5 x 4D per finger angles.
        self.ht.variances = Core.DoubleVector( posvar + rotvar + 5 * fingervar)

        print "Variances: ",list(self.ht.variances)
        print "Low Bounds: ",list(self.ht.lowBounds)
        print "High Bounds: ",list(self.ht.highBounds)
        print "Randomization Indices: ",list(self.ht.randomizationIndices)

        # Set the PSO budget, i.e. particles and generations.
        self.ht.particles = 64
        self.ht.generations = 25

    def _main_loop(self):
        # # Initialize RGBD acquisition. We will be acquiring images
        # # from a saved sequence, in oni format.
        #
        # # User should define a path to a saved sequence in oni format.
        # # Set path to empty string to perform live capture from an existing sensor.
        # oniPath = 'loop.oni'
        # acq = Acquisition.OpenNIGrabber(True, True, 'media/openni.xml', oniPath, True)
        # acq.initialize()

        # Initialization for the hand pose of the first frame is specified.
        # If track is lost, resetting will revert track to this pose.
        defaultInitPos = Core.ParamVector([ 0, 80, 900, 0, 0, 1, 0, 1.20946707135219810e-001, 1.57187812868051640e+000, 9.58033504364020840e-003, -1.78593063562731860e-001, 7.89636216585289100e-002, 2.67967456875403400e+000, 1.88385552327860720e-001, 2.20049375319072360e-002, -4.09740579183203310e-002, 1.52145111735213370e+000, 1.48366400350912500e-001, 2.85607073734409630e-002, -4.53781680931323280e-003, 1.52743247624671910e+000, 1.01751907812505270e-001, 1.08706683246161150e-001, 8.10845240231484330e-003, 1.49009228214971090e+000, 4.64716068193632560e-002, -1.44370358851376110e-001])

        # The 3D hand pose, as is tracked in the tracking loop.
        currentHandPose = defaultInitPos

        # State.
        paused = False
        delay = {True:0,False:1}
        frame = 0
        count=0
        tracking = 1
        actualFPS = 0.0
        # data = pickle.load(open("matrices.p","rb"))
        # viewMatrix = data[0]
        # projectionMatrix = data[1]
        # print projectionMatrix
        # print ttt
        print "Entering main Loop."
        while True:
            loopStart = time.time()*1000;
            try:
                # Acquire images and image calibrations and break if unsuccessful.
                # imgs is a list of numpy.andrray and clbs a list of Core.CameraMeta.
                # The two lists are of equal size and elements correspond to one another.
                # In OpenNIGrabber, the first image is the depth and the second is the RGB.
                # In the media/openni.xml file it is specified that the depth will be aligned
                # to the RGB image and that mirroring will be off. The resolution is VGA.
                # It is not obligatory to use the OpenNIGrabber. As long as you can somehow provide
                # aligned depth and RGB images and corresponding Core.CameraMeta, you can use 3D
                # hand tracking.
                # imgs, clbs = acq.grab()
                # Get the depth calibration to extract some basic info.
                # c = self.clbs
                # img1 = self.xtion_img_d
                # print np.max(img1)
                # img2 = self.xtion_img
                # It is assumed that both RGB and depth streams have the same dimensions as images

                # The calibration used corresponds to the depth stream
                # print 't1'
                imgs = [ self.xtion_img_d, # Make sure 16bit loading is perserved
                         self.xtion_img ]
                # print 't2'
                cfr = Core.CameraFrustum()
                fx = 525.0
                fy = 525.0
                cx = 319.5
                cy = 239.5
                width = 640
                height = 480
                zNear = 200
                zFar = 1500
                cfr.setIntrinsics(fx, fy, cx, cy, width, height, zNear, zFar) # assuming some focal lengths fx, fy and principal point cx, cy, image size width X height and a reasonable clipping plane for near (zNear) and far (zFar) in the target units (e.g. millimeters).
                # cfr.setExtrinsics(translation, rotation) # assuming a 3D translation and rodrigues rotation. Optional.
                c = Core.CameraMeta(cfr, width, height) # assuming the width and height of the images in pixels
            except:
                continue

            width,height = int(c.width),int(c.height)

            # Step 1: configure 3D rendering to match depth calibration.
            # step1_setupVirtualCamera returns a view matrix and a projection matrix (graphics).
            viewMatrix,projectionMatrix = self.ht.step1_setupVirtualCamera(c)

            # Step 2: compute the bounding box of the previously tracked hand pose.
            # For the sake of efficiency, search is performed in the vicinity of
            # the previous hand tracking solution. Rendering will be constrained
            # in the bounding box (plus some padding) of the previous tracking solution,
            # in image space.
            # The user might chose to bypass this call and compute a bounding box differently,
            # so as to incorporate other information as well.
            bb = self.ht.step2_computeBoundingBox(currentHandPose, width, height, 0.1)

            # Step 3: Zoom rendering to given bounding box.
            # The renderer is configures so as to map its projection space
            # to the given bounding box, i.e. zoom in.
            self.ht.step3_zoomVirtualCamera(projectionMatrix, bb,width,height)

            # Step 4: Preprocess input.
            # RGBD frames are processed to as to isolate the hand.
            # This is usually done through skin color detection in the RGB frame.
            # The user might chose to bypass this call and do foreground detection
            # in some other way. What is required is a labels image which is non-zero
            # for foreground and a depth image which contains depth values in mm.
            labels, depths = self.ht.step4_preprocessInput(imgs[1], imgs[0], bb)

            # Step5: Upload observations for GPU evaluation.
            # Hypothesis testing is performed on the GPU. Therefore, observations
            # are also uploaded to the GPU.
            self.ht.step5_setObservations(labels, depths)

            fps = 0
            if tracking:
                t = clock()
                # Step 6: Track.
                # Tracking is initialized with the solution for the previous frame
                # and computes the solution for the current frame. The user might
                # chose to initialize tracking from a pose other than the solution
                # from the previous frame. This solution needs to be 27-D for 3D
                # hand tracking with the specified decoder.
                score, currentHandPose = self.ht.step6_track(currentHandPose)
                t = clock() - t
                fps = 1.0 / t


            # Step 7 : Visualize.
            # This call superimposes a hand tracking solution on a RGB image
            viz = self.ht.step7_visualize(imgs[1], viewMatrix,projectionMatrix, currentHandPose)
            cv2.putText(viz, 'UI FPS = %f, Track FPS = %f' % (actualFPS , fps), (20, 20), 0, 0.5, (0, 0, 255))

            cv2.imshow("Hand Tracker",viz)

            key = cv2.waitKey(delay[paused])

            # Press 's' to start/stop tracking.
            if key & 255 == ord('s'):
                tracking = not tracking
                currentHandPose = defaultInitPos

            # Press 'q' to quit.
            if key & 255 == ord('q'):
                break

            # Press 'p' to pause.
            if key &255 == ord('p'):
                paused = not paused

            frame += 1
            loopEnd = time.time()*1000;
            actualFPS = (1000.0/(loopEnd-loopStart))



#--------------------------------------------------------------------------------------#
def main():
    rospy.init_node('hand_tracker')
    rospy.loginfo('hand tracker running..')
    H=hand_tracking()
    H._main_loop()

    # rospy.spin()

#--------------------------------------------------------------------------------------#
if __name__ == '__main__':
    main()
