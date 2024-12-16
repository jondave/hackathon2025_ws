import cv2
import numpy as np
from skimage.draw import line
import time
import math

global_anchor = 277
global_eor = 0
eor_count = 0

class SaturatedInteger(object):
    """Emulates an integer, but with a built-in minimum and maximum."""

    def __init__(self, min_, max_, value=None):
        self.min = min_
        self.max = max_
        self.value = min_ if value is None else value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_val):
        self._value = min(self.max, max(self.min, new_val))

    @staticmethod
    def _other_val(other):
        """Get the value from the other object."""
        if hasattr(other, 'value'):
            return other.value
        return other

    def __add__(self, other):
        new_val = self.value + self._other_val(other)
        return SaturatedInteger(self.min, self.max, new_val)

    __radd__ = __add__

    def __eq__(self, other):
        return self.value == self._other_val(other)

class LineScan(object):
    def __init__(self, image, rgb):
        self.image = image       #                            ←――― Width of Image ――→
        self.ascans = [] #For anchor scans                    |⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺Ⓐ⎺⎺⎺⎺⎺⎺⎺⎺⎺|↑
        self.rgb = rgb           #                            |          / \          ||
        self.A = [0, 277] #Anchor Point (y,x) from top left   |         /   \         |Height of Image 
        self.B = [511, 200] #Begin Point (y,x) from top left  |        /     \        ||
        self.C = [511, 400] #Cease Point (y,x) from top left  |_______Ⓑ______Ⓒ______|↓
        self.ScanPeriod = 1 #Periodicity of Pixel Scan through Ⓑ to Ⓒ 
        self.scale = 0.2# Percentage of the image used for anchor scans (from top)
        self.a_range = [100, 350]#Range for anchor scans
        self.rgb = rgb
        self.rgb2 = self.rgb.copy()
        self.EOR = 0
        self.escans = []#For eor scans
        self.entry_mode = 'l' #Turning direction at the EOR

    def anchor_scan(self, I, step):
        self.ascans = [0]*self.a_range[0]#For anchor scans
        for i in range(self.a_range[0],self.a_range[1]):
            rows, columns = line(0+(step*int(((self.image.shape[0]-1)*self.scale))), i, ((step+1)*int(((self.image.shape[0]-1)*self.scale))), i)
            single_count = np.sum(I[rows, columns])
            self.ascans.append(single_count)
        #self.A[1] = int((0.95*float(global_anchor)) + (0.05*float(np.argmax(self.ascans)))) # anchor update with complementary filter
        self.A[1] = int(np.argmax(self.ascans))#Uncomment for non-filtered image specific anchor
        self.B[1] = SaturatedInteger(200,450,self.A[1]-100).value
        self.C[1] = SaturatedInteger(200,450,self.A[1]+100).value
        return 0

    def eor_scan(self, I, step):
        self.escans = []#For eor scans
        global global_eor
        global eor_count
        for i in range(0+(step*int(((self.image.shape[0]-1)*self.scale))),((step+1)*int(((self.image.shape[0]-1)*self.scale)))):
            rows, columns = line(i, 0, i, (self.image.shape[0]-1))
            single_count = np.sum(I[rows, columns])
            self.escans.append(single_count)
        if eor_count == 0:
            eor_count = 1
            global_eor = (step*int((self.image.shape[0]-1)*self.scale))+int(np.argmax(self.escans))
            self.EOR = global_eor
        else:
            eor_count += 1
            self.EOR = int( (0.8*float(global_eor)) + int(0.2*float((step*int((self.image.shape[0]-1)*self.scale))+int(np.argmax(self.escans)))) )
            global_eor = self.EOR 
        cv2.line(self.image, (0,self.EOR), ((self.image.shape[0]-1),self.EOR), (0, 255, 0), thickness=2)#EOR
        return 0

    def entry_scan(self, I, mode):
        if mode == 'l':
            B = [int(self.image.shape[0]/2), 0]  
            C = self.B
            self.scans = [0] * B[0] #Create a zero list to fill image origin to Ⓑ

            for y in range(B[0],(self.image.shape[0]-1),self.ScanPeriod):
                rows, columns = line(self.A[0], self.A[1], y, (self.image.shape[0]-1))
                single_count = np.sum(I[rows, columns])
                self.scans.append(single_count)

            for x in range(0,C[1],self.ScanPeriod):
                rows, columns = line(self.A[0], self.A[1], (self.image.shape[0]-1), x)
                single_count = np.sum(I[rows, columns])
                self.scans.append(single_count)

            selector = int(np.argmax(self.scans))
            if selector <= (self.image.shape[0]-1):
                S = [selector, 0]

            elif selector > (self.image.shape[0]-1):
                S = [(self.image.shape[0]-1), (selector - (self.image.shape[0]-1))]

        elif mode == 'r':
            B = self.C
            C = [int(self.image.shape[0]/2), int(self.image.shape[0]-1)]   
            self.scans = [0] * B[1] #Create a zero list to fill image origin to Ⓑ

            for x in range(B[1],(self.image.shape[0]-1),self.ScanPeriod):
                rows, columns = line(self.A[0], self.A[1], (self.image.shape[0]-1), x)
                single_count = np.sum(I[rows, columns])
                self.scans.append(single_count)

            for y in range((self.image.shape[0]-1), C[0],-self.ScanPeriod):
                rows, columns = line(self.A[0], self.A[1], y, (self.image.shape[0]-1))
                single_count = np.sum(I[rows, columns])
                self.scans.append(single_count)

            selector = int(np.argmax(self.scans))

            if selector <= (self.image.shape[0]-1):
                S = [(self.image.shape[0]-1), selector]

            elif selector > (self.image.shape[0]-1):
                S = [(2*(self.image.shape[0]-1)-selector), (self.image.shape[0]-1)]

        else:
            print("Invalid Re-Entry Mode")
            return 0
        #cv2.line(self.rgb, (self.A[1],self.A[0]), (S[1],S[0]), (0, 255, 255), thickness=2)
        if S[0] != 0:#Failsafe for divide by zero error
            r = int((self.EOR*(S[1]-self.A[1])/S[0]) + self.A[1])-0
        else:
            r = 0

        return r

    def scan(self):
        #Preprocess Image
        I = self.image
        #I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
        ret,I = cv2.threshold(I, 200, 1, 0)
        exit_flag = False
        re_entry = 0

        #Anchor Scans
        global global_anchor
        self.anchor_scan(I,0)
        self.A[1] = int((0.9*float(global_anchor)) + (0.1*float(np.argmax(self.ascans)))) # anchor update with complementary filter

        #For row following with EOR
        if np.max(self.ascans) < 40:#If anchor scan strength is below thresold, scan next stage
            self.anchor_scan(I,1)
            self.eor_scan(I,0)
            re_entry = self.entry_scan(I, self.entry_mode)
            cv2.line(self.rgb, (0,self.EOR), ((self.image.shape[0]-1),self.EOR), (0, 255, 0), thickness=2)#EOR
            cv2.circle(self.rgb, (re_entry,self.EOR), 20, (255, 0, 0), 2)
            cv2.line(self.rgb, (self.A[1],self.A[0]), (re_entry,self.EOR), (0, 255, 255), thickness=2)
            self.A[1] = int((0.9*float(global_anchor)) + (0.1*float(np.argmax(self.ascans)))) # anchor update with complementary filter
            #print("Step 1")
        if np.max(self.ascans) < 40:#If anchor scan strength is below thresold, scan next stage
            self.anchor_scan(I,2)
            self.eor_scan(I,1)
            re_entry = self.entry_scan(I, self.entry_mode)
            self.rgb = self.rgb2
            cv2.circle(self.rgb, (re_entry,self.EOR), 20, (255, 0, 0), 2)
            cv2.line(self.rgb, (0,self.EOR), ((self.image.shape[0]-1),self.EOR), (0, 255, 0), thickness=2)#EOR
            cv2.line(self.rgb, (self.A[1],self.A[0]), (re_entry,self.EOR), (0, 255, 255), thickness=2)
            self.A[1] = int((0.9*float(global_anchor)) + (0.1*float(np.argmax(self.ascans)))) # anchor update with complementary filter
            #print("Step 2")
        if np.max(self.ascans) < 40:#If anchor scan strength is below thresold, reset anchor point
            self.A[1] = 277
            self.eor_scan(I,2)
            re_entry = self.entry_scan(I, self.entry_mode)
            self.rgb = self.rgb2
            cv2.circle(self.rgb, (re_entry,self.EOR), 20, (255, 0, 0), 2)
            cv2.line(self.rgb, (0,self.EOR), ((self.image.shape[0]-1),self.EOR), (0, 255, 0), thickness=2)#EOR
            cv2.line(self.rgb, (self.A[1],self.A[0]), (re_entry,self.EOR), (0, 255, 255), thickness=2)
            exit_flag = True
            print("Exit Row")


        global_anchor = self.A[1] #Update global anchor
        #self.A[1] = int(np.argmax(self.ascans))#Uncomment for non-filtered image specific anchor


        #Primary Scans
        self.scans = [0] * self.B[1] #Create a zero list to fill image origin to Ⓑ
        for i in range(self.B[1],self.C[1],self.ScanPeriod):
            rows, columns = line(self.A[0], self.A[1], (self.image.shape[0]-1), i)
            single_count = np.sum(I[rows, columns])
            self.scans.append(single_count)
        self.image = cv2.addWeighted(self.rgb,1.0,cv2.cvtColor(self.image,cv2.COLOR_GRAY2BGR),0.45,0)


        selector = int(np.argmax(self.scans))
        #print(selector)
        cv2.line(self.image, (self.A[1],self.A[0]), (selector, (self.image.shape[0]-1)), (0, 0, 255), thickness=2)
        cv2.imshow("Image window", self.image)
        cv2.waitKey(1)


        #Line Angle Calculation
        angl = math.degrees(math.atan2((self.A[1]-selector),self.C[0]))
        #print(angl)
        #print(selector)

        return angl, selector, exit_flag, re_entry, self.EOR




