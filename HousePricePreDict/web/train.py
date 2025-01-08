import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

# 1. Tải dữ liệu
data = pd.read_csv("mogi_properties_ha_dong.csv")

# 2. Tiền xử lý dữ liệu
data.dropna(inplace=True)  # Loại bỏ các hàng có giá trị None
data.drop(columns=['title'], inplace=True)  # Loại bỏ cột title

# Chuyển đổi diện tích về m²
data['area'] = data['area'].apply(lambda x: float(re.sub(r'[^\d.]', '', x.lower())))  # Chuyển về chữ thường và loại bỏ ký tự không phải số

# Chuyển đổi giá về dạng số
data['price'] = data['price'].apply(lambda x: float(re.sub(r'[^\d.]', '', x.lower())))  # Chuyển về chữ thường và loại bỏ ký tự không phải số

# Loại bỏ các mô tả không cần thiết trong các cột
data['area'] = data['area'].apply(lambda x: float(re.sub(r' m²| m2| PN| WC| tầng| T| t', '', str(x))))  # Loại bỏ PN, WC, tầng...
data['price'] = data['price'].apply(lambda x: float(re.sub(r' tỷ| triệu| triệu đồng', '', str(x))))  # Loại bỏ tỷ, triệu...

# Tiền xử lý cho cột bedrooms và bathrooms
data['bedrooms'] = data['bedrooms'].apply(lambda x: int(re.sub(r'[^0-9]', '', str(x))))  # Loại bỏ WC, PN và chuyển về số nguyên
data['bathrooms'] = data['bathrooms'].apply(lambda x: int(re.sub(r'[^0-9]', '', str(x))))  # Loại bỏ WC, PN và chuyển về số nguyên

# One-hot encoding cho cột street
streets_one_hot = pd.get_dummies(data['street'], prefix='street')
data = pd.concat([data, streets_one_hot], axis=1)

# Lưu danh sách các tên đường
streets = streets_one_hot.columns.tolist()
with open('streets_list.pkl', 'wb') as f:
    pickle.dump(streets, f)

# Lấy các cột cần thiết
X = data[['area', 'bedrooms', 'bathrooms', 'floors'] + streets]  # Các đặc điểm bao gồm cả đường phố
y = data['price']  # Giá nhà

# Tiêu chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lưu scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

mlp_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # Adjust based on regression or classification
])

mlp_model.compile(optimizer='adam', loss='mse')  # Use 'binary_crossentropy' or 'categorical_crossentropy' for classification
mlp_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Lưu mô hình
mlp_model.save('house_price_model.h5')
