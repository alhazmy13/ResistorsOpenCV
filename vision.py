import cv2
import numpy as np
import os

from config import Config


class Vision:
    config = Config()

    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.rect_cascade = cv2.CascadeClassifier(os.getcwd() + "/data/resistors.xml")

    def get_camera(self):
        return self.video_capture.read()

    def release(self):
        self.video_capture.release()

    def print_result(self, live_img):
        resistor_close = self.find_resistors(live_img=live_img)
        for i in range(len(resistor_close)):
            sorted_bands = self.__find_bands(resistor_close[i])
            self.__draw_result(sorted_bands, live_img, resistor_close[i][1])

    def __is_valid_contour(self, cnt):
        if cv2.contourArea(cnt) < self.config.MIN_AREA:
            return False
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if aspect_ratio > 0.4:
                return False
        return True

    def __draw_result(self, sorted_bands, live_img, resistor_position):
        x, y, w, h = resistor_position
        start_value = ""
        if len(sorted_bands) in [3, 4, 5]:
            for band in sorted_bands[:-1]:
                start_value += str(band[3])
            int_val = int(start_value)
            int_val *= 10 ** sorted_bands[-1][3]
            cv2.rectangle(live_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(live_img, str(int_val) + " OHMS", (x + w + 10, y), self.config.FONT, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            return
        cv2.rectangle(live_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def __find_bands(self, resistor_info):
        resistor_img = cv2.resize(resistor_info[0], (400, 200))
        pre_bil = cv2.bilateralFilter(resistor_img, 5, 80, 80)
        hsv = cv2.cvtColor(pre_bil, cv2.COLOR_BGR2HSV)
        thresh = cv2.adaptiveThreshold(cv2.cvtColor(pre_bil, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 59, 5)
        thresh = cv2.bitwise_not(thresh)
        bands_position = []

        check_colours = self.config.COLOUR_BOUNDS

        for color in check_colours:
            mask = cv2.inRange(hsv, color[0], color[1])
            if color[2] == "RED":  # combining the 2 RED ranges in hsv
                red_mask2 = cv2.inRange(hsv, self.config.RED_TOP_LOWER, self.config.RED_TOP_UPPER)
                mask = cv2.bitwise_or(red_mask2, mask, mask)

            mask = cv2.bitwise_and(mask, thresh, mask=mask)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # filter invalid contours, store valid ones
            for k in range(len(contours) - 1, -1, -1):
                if self.__is_valid_contour(contours[k]):
                    leftmost_point = tuple(contours[k][contours[k][:, :, 0].argmin()][0])
                    bands_position += [leftmost_point + tuple(color[2:])]
                    cv2.circle(pre_bil, leftmost_point, 5, (255, 0, 255), -1)
                else:
                    contours.pop(k)

            cv2.drawContours(pre_bil, contours, -1, color[-1], 3)

        cv2.imshow('Contour Display', pre_bil)

        return sorted(bands_position, key=lambda tup: tup[0])

    def find_resistors(self, live_img):
        _live_img = cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY)
        resistors_close = []

        # detect resistors in main frame
        resistors_find = self.rect_cascade.detectMultiScale(_live_img, 1.1, 25)
        for (x, y, w, h) in resistors_find:  # SWITCH TO H,W FOR <CV3

            roi_gray = _live_img[y:y + h, x:x + w]
            roi_color = live_img[y:y + h, x:x + w]

            # apply another detection to filter false positives
            second_pass = self.rect_cascade.detectMultiScale(roi_gray, 1.01, 5)

            if len(second_pass) != 0:
                resistors_close.append((np.copy(roi_color), (x, y, w, h)))
        return resistors_close

    @staticmethod
    def pass_function():
        return None
