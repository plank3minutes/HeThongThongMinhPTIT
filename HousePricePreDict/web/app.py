from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
import numpy as np

# Tải mô hình mà không biên dịch
model = tf.keras.models.load_model('house_price_model.h5', compile=False)

# Biên dịch lại mô hình với hàm mất mát thích hợp
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Tải scaler và các cột đường phố đã lưu
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('y_scaler.pkl', 'rb') as f:
    y_scaler = pickle.load(f)
with open('my_list.pkl', 'rb') as f:
    street_columns = pickle.load(f)

# Lấy danh sách các tên đường từ các cột đường phố
street_names = [col.replace('street_', '') for col in street_columns]

# Tạo ứng dụng Flask
app = Flask(__name__)

# Route để hiển thị trang web
@app.route('/')
def home():
    return render_template('index.html', streets=street_names)

# Route để dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    area = float(data['area'])
    bedrooms = int(data['bedrooms'])
    bathrooms = int(data['bathrooms'])
    floors = float(data['floors'])
    street = data['street']

    # Tạo mảng đầu vào với các đặc điểm
    input_features = [area, bedrooms, bathrooms, floors]
    street_features = [1 if col == f'street_{street}' else 0 for col in street_columns]
    input_data = np.array(input_features + street_features).reshape(1, -1)

    # Chuẩn hóa dữ liệu đầu vào
    input_data_scaled = scaler.transform(input_data)

    # Dự đoán giá nhà
    prediction_scaled = model.predict(input_data_scaled)
    prediction = y_scaler.inverse_transform(prediction_scaled)

    return jsonify({'prediction': float(prediction[0][0])})

# Chạy ứng dụng Flask
if __name__ == '__main__':
    from flask_ngrok import run_with_ngrok
    run_with_ngrok(app)  # Khởi động ngrok khi chạy ứng dụng
    app.run()
