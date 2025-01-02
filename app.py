from flask import Flask, render_template, Response, jsonify
import cv2
from deepface import DeepFace

app = Flask(__name__)

# Initialize video capture
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    """Render the HTML page."""
    return render_template('index.html')

def generate_frames():
    """Generate video frames for the frontend."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]

            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][emotion]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"{emotion} ({confidence:.2f}%)", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            except Exception as e:
                print(f"Error detecting emotion: {e}")

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route to serve video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['GET'])
def capture():
    """Capture the current emotion result."""
    ret, frame = cap.read()
    if ret:
        # Convert frame to an image and process it
        result = "Emotion captured successfully!"  # Placeholder
        return jsonify({"message": result})
    return jsonify({"message": "Failed to capture emotion."})

@app.route('/close', methods=['POST'])
def close():
    """Close the application."""
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Application closed."})

if __name__ == '__main__':
    app.run(debug=True)
