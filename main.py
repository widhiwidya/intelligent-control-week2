import cv2
import joblib
import numpy as np

# muat model knn dan scaler
knn = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# inisialisasi kamera
cap = cv2.VideoCapture(0)

while True :
    ret, frame = cap.read()
    if not ret :
        break

    # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame[height//2, width//2]

    # Normalisasi pixel sebelum prediksi
    pixel_center_scaled = scaler.transform([pixel_center])

    # Prediksi warna dan probabilitasnya
    color_pred = knn.predict(pixel_center_scaled)[0]
    prob_pred = knn.predict_proba(pixel_center_scaled).max() * 100

    # Tampilkan warna dan akurasi pada frame
    cv2.putText(frame, f'Color: {color_pred} ({prob_pred:.2f}%)', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
