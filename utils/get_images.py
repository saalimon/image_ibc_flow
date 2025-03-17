import cv2
import urllib.parse
from dotenv import dotenv_values
from datetime import datetime

CONFIG = dotenv_values(".env")

username = CONFIG["CAMERA_USERNAME"]
password = CONFIG["CAMERA_PASSWORD"]
ip_address = CONFIG["CAMERA_IP_ADDRESS"]
port = CONFIG["CAMERA_PORT"]
stream_path = "/Streaming/channels/2/"

# Encode username and password
encoded_password = urllib.parse.quote(password)

url = f"rtsp://{username}:{encoded_password}@{ip_address}:{port}{stream_path}"

print(f"Trying to connect to: {url}")  # Debugging output

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Failed to open stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Ensure frames exist
    cv2.imshow("Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if cv2.waitKey(1) & 0xFF == ord("s"):
        # get date and time for the image
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y%H%M%S")
        # write the frame to input/images folder
        cv2.imwrite(f"../input/images/frame_{dt_string}.jpg", frame)

cap.release()
cv2.destroyAllWindows()
