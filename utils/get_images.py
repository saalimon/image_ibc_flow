import cv2
import urllib.parse
from dotenv import dotenv_values
import os
import time


class GetImages:
    def __init__(self, CONFIG_PATH=".env"):
        CONFIG = dotenv_values(CONFIG_PATH)

        username = CONFIG["CAMERA_USER"]
        password = CONFIG["CAMERA_PASSWORD"]
        ip_address = CONFIG["CAMERA_IP"]
        port = CONFIG["CAMERA_PORT"]
        stream_path = "/Streaming/channels/2/"
        # Encode username and password
        encoded_password = urllib.parse.quote(password)
        self.url = (
            f"rtsp://{username}:{encoded_password}@{ip_address}:{port}{stream_path}"
        )

        print(f"Trying to connect to: {self.url}")  # Debugging output

    def get_images(self):
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print("Failed to open stream")
            exit()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("No frame received")
                break

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Ensure frames exist
            cv2.imshow("Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("s"):
                # Save the frame when 's' is pressed
                # Ensure the directory exists
                os.makedirs("../output", exist_ok=True)
                # Save the frame with a timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                # Save the frame with a timestamp
                name = f"captured_frame_{timestamp}.png"
                cv2.imwrite(f"../output/{name}", frame)
                print(f"Frame saved as {name}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    get_images = GetImages(".env")
    try:
        get_images.get_images()
    except KeyboardInterrupt:
        print("Stream interrupted by user.")


if __name__ == "__main__":
    main()
