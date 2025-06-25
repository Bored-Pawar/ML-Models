import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model

# Load trained model
model = load_model(r"C:\Users\Aditya Pramod Pawar\Machine Learning\Digit_recognizer\digit_model.h5", compile=False)

# Create a blank black canvas
canvas_size = 300
img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

drawing = False
ix, iy = -1, -1

# Mouse drawing function
# Mouse drawing function
def draw(event, x, y, flags, param):
    global drawing, ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), 4, (255), -1)  # thinner brush
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 2, (255), -1)

# Preprocessing to match MNIST style
def preprocess(image):
    # Threshold to binary
    _, img_bin = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

    # Find bounding box
    coords = cv2.findNonZero(img_bin)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop to bounding box
    cropped = img_bin[y:y+h, x:x+w]

    # Add padding to make square
    side = max(w, h)
    padded = np.zeros((side, side), dtype=np.uint8)
    x_offset = (side - w) // 2
    y_offset = (side - h) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    # Resize to 28x28 and normalize
    # Resize to 28x28 and normalize
    blurred = cv2.GaussianBlur(padded, (3, 3), 0)
    resized = cv2.resize(blurred, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0

    # Reshape for model input
    return normalized.reshape(1, 28, 28, 1), resized

# Set up drawing window
cv2.namedWindow("Draw Digit")
cv2.setMouseCallback("Draw Digit", draw)

# Main loop
while True:
    cv2.imshow("Draw Digit", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Clear canvas
        img[:] = 0

    elif key == ord('q'):  # Quit
        break

    elif key == ord('p'):  # Predict
        try:
            processed, model_input = preprocess(img)

            # Show what the model sees
            plt.imshow(model_input, cmap='gray')
            plt.title("Input to Model")
            plt.axis('off')
            plt.show()

            prediction = model.predict(processed)
            predicted_digit = np.argmax(prediction)
            cv2.putText(img, f'Prediction: {predicted_digit}', (10, 290),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2, cv2.LINE_AA)
            print(f"Predicted Digit: {predicted_digit}")
        except:
            print("Draw a digit before predicting!")

cv2.destroyAllWindows()
