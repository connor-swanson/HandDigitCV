import cv2
from utils import make_prediction, load_models, draw_label


cam = cv2.VideoCapture(0)  # zero is webcam

cv2.namedWindow("My Window")

# Check if camera was opened correctly
if not (cam.isOpened()):
    print("Could not open video device")


# load model objects
model, encoder = load_models()

# run camera
while True:
    # Capture frame and format
    ret, frame = cam.read()

    # make predictions
    prediction_frame = cv2.resize(frame, (128, 128), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    pred = make_prediction(model, encoder, prediction_frame)

    # Overlay predictions on output frame
    draw_label(frame, 'Fersher: {}'.format(pred))

    # Display Resulting Frame
    cv2.imshow("preview", frame)

    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()

cv2.destroyAllWindows()
