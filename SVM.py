import numpy as np
import joblib
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Pilih model yang ingin digunakan ('decision_tree' atau 'svm')
MODEL_TYPE = 'svm'  # Ubah menjadi 'svm' jika ingin menggunakan SVM

# Load dataset
color_data = pd.read_csv('colors.csv')
X = color_data[['R', 'G', 'B']].values
y = color_data['color_name'].values

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Pilih model
if MODEL_TYPE == 'decision_tree':
    model = DecisionTreeClassifier(random_state=42)
elif MODEL_TYPE == 'svm':
    model = SVC(kernel='linear', random_state=42)
else:
    raise ValueError("MODEL_TYPE harus 'decision_tree' atau 'svm'")

# Latih model
model.fit(X_train, y_train)

# Evaluasi model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"Akurasi pada data latih: {train_acc * 100:.2f}%")
print(f"Akurasi pada data uji: {test_acc * 100:.2f}%")

# Simpan model dan scaler
joblib.dump(model, 'color_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model dan scaler berhasil disimpan!")

# Muat kembali model dan scaler untuk deteksi warna
model = joblib.load('color_model.pkl')
scaler = joblib.load('scaler.pkl')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Menggunakan deteksi kontur untuk mengikuti objek berwarna
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi ke format HSV untuk deteksi warna yang lebih baik
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Ambil batas bawah dan atas untuk segmentasi warna
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Temukan kontur objek
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter kontur kecil
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]
            
            avg_color = np.mean(roi, axis=(0, 1)).astype(int)
            avg_color_scaled = scaler.transform([avg_color])
            color_pred = model.predict(avg_color_scaled)[0]
            bbox_color = (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))
            
            # Gambar bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), bbox_color, 2)
            cv2.putText(frame, f'{color_pred}', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
