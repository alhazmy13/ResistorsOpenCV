import cv2

from vision import Vision

vision = Vision(debug=False)

if __name__ == "__main__":
    while not (cv2.waitKey(1) == ord('q')):
        _, live_img = vision.get_camera()
        vision.print_result(live_img=live_img)
        cv2.imshow("Frame", live_img)
    vision.cap.release()
    cv2.destroyAllWindows()
