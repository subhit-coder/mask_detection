from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
model = load_model("model/mask_detector.keras")

categories = ["with_mask", "without_mask"]
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# image preprocessing function
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    return image[y:y+h, x:x+w], (int(x), int(y), int(w), int(h))


@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    face, bbox = detect_face(img)
    if face is None:
        raise HTTPException(status_code=400, detail="No face detected")

    processed = preprocess_image(face)
    prediction = model.predict(processed, verbose=0)[0]

    label = categories[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    x, y, w, h = bbox

    return {
        "label": label,
        "confidence": confidence,
        "bbox": {"x": x, "y": y, "w": w, "h": h}
    }

# run locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)