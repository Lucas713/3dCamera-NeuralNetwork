#!/usr/bin/python3

# Copyright (C) 2019 Infineon Technologies & pmdtechnologies ag
#
# THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
# PARTICULAR PURPOSE.

"""This sample shows how to shows how to capture image data.

It uses Python's numpy and matplotlib to process and display the data.
"""

import argparse
import roypy
import time
import queue
import numpy as np

from sample_camera_info import print_camera_info
from roypy_sample_utils import CameraOpener, add_camera_opener_options
from roypy_platform_utils import PlatformHelper

import numpy as np
import matplotlib.pyplot as plt



class MyListener(roypy.IDepthImageListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.queue = q

    def onNewData(self, data):
        zvalues = []
        for i in range(data.getNumPoints()):
            zvalues.append(data.getCDData(i))
        zarray = np.asarray(zvalues)
        p = zarray.reshape (-1, data.width)        
        self.queue.put(p)

    def paint (self, data,i):
        """Called in the main thread, with data containing one of the items that was added to the
        queue in onNewData.
        """
        # create a figure and show the raw data
        plt.figure(1)
        
        data2=np.int_(data)
        

        
        data3= np.delete(data2, range(25), axis=0)
        sizeY = np.shape(data3)[0]
        data3= np.delete(data3, range(110,sizeY), axis=0)
        data3= np.delete(data3, range(18), axis=1)
        sizeX = np.shape(data3)[1]
        data3= np.delete(data3, range(200,sizeX), axis=1)
                
        sizeX = np.shape(data3)[1]
        sizeY = np.shape(data3)[0]
        for i2 in range(sizeX):
            for j in range(sizeY):
                if data3[j,i2]>600:
                    somme=0
                    compteur=0
                    indiceminh= min(0,j-10)
                    indiceminl= min(0,i2-10)
                    indicemaxh= min(0,j+10)
                    indicemaxl= min(0,i2+10)
                    for k in range(indiceminl,indicemaxl):
                        for l in range(indiceminh,indicemaxh):
                            if data3[l,k]<600:
                                somme=somme+data3[k,l]
                                compteur=compteur+1
                    if somme > 1 and somme//compteur < 600:
                        data3[j,i2]=somme//compteur
                        
                    else:
                        data3[j,i2]=400
                else:
                    somme=0
                    compteur=0
                    distance = 10
                    indiceminh= min(0,j-distance)
                    indiceminl= min(0,i2-distance)
                    indicemaxh= min(0,j+distance)
                    indicemaxl= min(0,i2+distance)
                    for k in range(indiceminl,indicemaxl):
                        for l in range(indiceminh,indicemaxh):
                            somme=somme+data3[k,l]
                            compteur=compteur+1
                    if compteur>0:
                        
                        data3[j,i2]=somme/compteur

        data3[0,0]=380

        
        plt.imshow(data3)
        np.savetxt('/home/nvidia/Desktop/repository/matrix'+str(i)+'.txt', data3,fmt='%i', delimiter = ',')  
        
        
        plt.show(block = False)
        plt.draw()
        
        # this pause is needed to ensure the drawing for
        # some backends
        plt.pause(0.001)

def main ():
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser (usage = __doc__)
    add_camera_opener_options (parser)
    parser.add_argument ("--seconds", type=int, default=10, help="duration to capture data (default is 15 seconds)")
    options = parser.parse_args()
    opener = CameraOpener (options)
    cam = opener.open_camera ()
    cam.setUseCase('MODE_5_45FPS_500')
    print_camera_info (cam)
    print("isConnected", cam.isConnected())
    print("getFrameRate", cam.getFrameRate())

    # we will use this queue to synchronize the callback with the main
    # thread, as drawing should happen in the main thread
    q = queue.Queue()
    l = MyListener(q)
    cam.registerDepthImageListener(l)
    cam.startCapture()
    # create a loop that will run for a time (default 15 seconds)
    process_event_queue (q, l, options.seconds)
    cam.stopCapture()

def process_event_queue (q, painter, seconds):
    # create a loop that will run for the given amount of time
    i=0
    while i<20:
        time.sleep(1)
        
        try:
            # try to retrieve an item from the queue.
            # this will block until an item can be retrieved
            # or the timeout of 1 second is hit
            item = q.get(True, 1)
        except queue.Empty:
            # this will be thrown when the timeout is hit
            break
        else:
            painter.paint (item,i)
            i= 1+i
            print("iteration numero "+str(i))

if (__name__ == "__main__"):
    main()
