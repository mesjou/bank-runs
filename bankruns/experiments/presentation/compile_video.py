import os

import cv2

if __name__ == "__main__":
    image_folder = "presentation/02"
    video_name = "presentation/02.mp4"

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    _fourcc = cv2.VideoWriter_fourcc(*"mp4V")
    video = cv2.VideoWriter(video_name, _fourcc, 15, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    print("The output video is {}".format(video_name))
