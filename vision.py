import cv2
import numpy as np
import os

from config import Config


class Vision:
    config = Config()

    def __init__(self, debug=False):
        if debug:
            cv2.namedWindow("frame")
            cv2.createTrackbar("lh", "frame", 0, 179, self.pass_function)
            cv2.createTrackbar("uh", "frame", 0, 179, self.pass_function)
            cv2.createTrackbar("ls", "frame", 0, 255, self.pass_function)
            cv2.createTrackbar("us", "frame", 0, 255, self.pass_function)
            cv2.createTrackbar("lv", "frame", 0, 255, self.pass_function)
            cv2.createTrackbar("uv", "frame", 0, 255, self.pass_function)
        t_path = os.getcwd()
        self.cap = cv2.VideoCapture(0)
        self.rect_cascade = cv2.CascadeClassifier(t_path + "/data/resistors.xml")
        self.debug = debug

    def get_camera(self):
        return self.cap.read()

    def print_result(self, live_img):
        res_close = self.find_resistors(live_img=live_img)
        for i in range(len(res_close)):
            sorted_bands = self.__find_bands(res_close[i])
            self.__draw_result(sorted_bands, live_img, res_close[i][1])

    def __is_valid_contour(self, cnt):
        # looking for a large enough area and correct aspect ratio
        if cv2.contourArea(cnt) < self.config.MIN_AREA:
            return False
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if aspect_ratio > 0.4:
                return False
        return True

    def __draw_result(self, sorted_bands, live_img, res_pos):
        x, y, w, h = res_pos
        str_val = ""
        if len(sorted_bands) in [3, 4, 5]:
            for band in sorted_bands[:-1]:
                str_val += str(band[3])
            int_val = int(str_val)
            int_val *= 10 ** sorted_bands[-1][3]
            cv2.rectangle(live_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(live_img, str(int_val) + " OHMS", (x + w + 10, y), self.config.FONT, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            return
        cv2.rectangle(live_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def __find_bands(self, resistor_info):
        if self.debug:
            uh = cv2.getTrackbarPos("uh", "frame")
            us = cv2.getTrackbarPos("us", "frame")
            uv = cv2.getTrackbarPos("uv", "frame")
            lh = cv2.getTrackbarPos("lh", "frame")
            ls = cv2.getTrackbarPos("ls", "frame")
            lv = cv2.getTrackbarPos("lv", "frame")
        # enlarge image
        res_img = cv2.resize(resistor_info[0], (400, 200))
        pre_bil = cv2.bilateralFilter(res_img, 5, 80, 80)
        hsv = cv2.cvtColor(pre_bil, cv2.COLOR_BGR2HSV)
        thresh = cv2.adaptiveThreshold(cv2.cvtColor(pre_bil, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 59, 5)
        thresh = cv2.bitwise_not(thresh)
        bands_pos = []

        if self.debug:
            check_colours = self.config.COLOUR_BOUNDS[0:1]
        else:
            check_colours = self.config.COLOUR_BOUNDS

        for clr in check_colours:
            if self.debug:
                mask = cv2.inRange(hsv, (lh, ls, lv), (uh, us, uv))
            else:
                mask = cv2.inRange(hsv, clr[0], clr[1])
                if clr[2] == "RED":  # combining the 2 RED ranges in hsv
                    red_mask2 = cv2.inRange(hsv, self.config.RED_TOP_LOWER, self.config.RED_TOP_UPPER)
                    mask = cv2.bitwise_or(red_mask2, mask, mask)

            mask = cv2.bitwise_and(mask, thresh, mask=mask)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # filter invalid contours, store valid ones
            for k in range(len(contours) - 1, -1, -1):
                if self.__is_valid_contour(contours[k]):
                    leftmost_point = tuple(contours[k][contours[k][:, :, 0].argmin()][0])
                    bands_pos += [leftmost_point + tuple(clr[2:])]
                    cv2.circle(pre_bil, leftmost_point, 5, (255, 0, 255), -1)
                else:
                    contours.pop(k)

            cv2.drawContours(pre_bil, contours, -1, clr[-1], 3)
            if self.debug:
                cv2.imshow("mask", mask)
                cv2.imshow('thresh', thresh)

        cv2.imshow('Contour Display', pre_bil)

        return sorted(bands_pos, key=lambda tup: tup[0])

    def find_resistors(self, live_img):
        gliveimg = cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY)
        res_close = []

        # detect resistors in main frame
        ress_find = self.rect_cascade.detectMultiScale(gliveimg, 1.1, 25)
        for (x, y, w, h) in ress_find:  # SWITCH TO H,W FOR <CV3

            roi_gray = gliveimg[y:y + h, x:x + w]
            roi_color = live_img[y:y + h, x:x + w]

            # apply another detection to filter false positives
            second_pass = self.rect_cascade.detectMultiScale(roi_gray, 1.01, 5)

            if len(second_pass) != 0:
                res_close.append((np.copy(roi_color), (x, y, w, h)))
        return res_close

    @staticmethod
    def pass_function():
        return None
