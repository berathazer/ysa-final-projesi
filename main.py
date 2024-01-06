# 20010011071 Berat Hazer
# 20010011066 Burcu Gül


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay , classification_report, accuracy_score


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def fillMissingValuesWithMean(data, featureLength):

    for i in range(featureLength):
        if is_binary_column_without_nan(data, i):
            data = fill_binary_column(data, i)
        else:
            data = calculate_mean(data, i)
    return data


def is_binary_column_without_nan(data, index):

    column_data = data.iloc[:, index]

    column_type = column_data.dtype

    if column_type in ['int64', 'float64']:
        unique_values = column_data.dropna().unique()
        # Benzersiz değerler sadece 0 ve 1 içeriyorsa binary bir sütundur.
        binary = all(value in {0, 1} for value in unique_values)
        return binary

    return False


def calculate_mean(data, index):

    column_data = data.iloc[:, index]
    mean_value = column_data.mean()
    data.iloc[:, index].fillna(mean_value, inplace=True)
    return data


def fill_binary_column(data, index):

    column_data = data.iloc[:, index]
    majority_value = column_data.mode().iloc[0]
    data.iloc[:, index].fillna(majority_value, inplace=True)
    return data


def calculate_min_max_normalization(data, index, new_min=0, new_max=1):
    column_data = data.iloc[:, index]
    current_min = column_data.min()
    current_max = column_data.max()

    normalized_data = (column_data - current_min) / (current_max - current_min)
    normalized_data = normalized_data * (new_max - new_min) + new_min

    data.iloc[:, index] = normalized_data

    return data

def normalization(data):

    for i in range(len(data.columns)-1):
        if is_binary_column_without_nan(data, i):
            continue
        data = calculate_min_max_normalization(data, i)
    return data


def calculate_column_statistics(data):
    column_statistics = {}

    for column in data.columns:
        column_data = data[column]
        stats = {
            "min": column_data.min(),
            "max": column_data.max(),
            "std": column_data.std()
        }
        column_statistics[column] = stats

        # Sütun istatistiklerini yazdır
        print(f"Sütun: {column}, Min: {stats['min']}, Max: {stats['max']}, Std: {stats['std']}")


###############################################################

# Verisetini yükleme ve eksik değerleri doldurma
path = './water_potability.csv'
dataset = pd.read_csv(path)
featureLength = len(dataset.columns) - 1
dataWithoutNormalization = fillMissingValuesWithMean(dataset.copy(), featureLength)
dataWithNormalization = normalization(dataWithoutNormalization.copy())

# Veriyi özellikler ve etiketler olarak ayırma
X_without_norm = dataWithoutNormalization.iloc[:, :-1].values
y_without_norm = dataWithoutNormalization.iloc[:, -1].values
X_with_norm = dataWithNormalization.iloc[:, :-1].values
y_with_norm = dataWithNormalization.iloc[:, -1].values

# Veriyi eğitim ve test setlerine ayırma
X_train_without_norm, X_test_without_norm, y_train_without_norm, y_test_without_norm = train_test_split(
    X_without_norm, y_without_norm, test_size=0.2, random_state=42)

X_train_with_norm, X_test_with_norm, y_train_with_norm, y_test_with_norm = train_test_split(
    X_with_norm, y_with_norm, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_without_norm = scaler.fit_transform(X_train_without_norm)
X_test_without_norm = scaler.transform(X_test_without_norm)

scaler = StandardScaler()
X_train_with_norm = scaler.fit_transform(X_train_with_norm)
X_test_with_norm = scaler.transform(X_test_with_norm)

# Model oluşturma ve eğitme (Normalize Edilmemiş Veri)
model_without_norm = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_without_norm.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_without_norm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_without_norm.fit(X_train_without_norm, y_train_without_norm, epochs=10, batch_size=32, validation_data=(X_test_without_norm, y_test_without_norm))


# Model oluşturma ve eğitme (Normalize Edilmiş Veri)
model_with_norm = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_with_norm.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_with_norm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_with_norm.fit(X_train_with_norm, y_train_with_norm, epochs=10, batch_size=32, validation_data=(X_test_with_norm, y_test_with_norm))

# Tahmin yapma
y_pred_without_norm = (model_without_norm.predict(X_test_without_norm) > 0.5).astype(int)
y_pred_with_norm = (model_with_norm.predict(X_test_with_norm) > 0.5).astype(int)

# Doğruluk hesaplama
accuracy_without_norm = accuracy_score(y_test_without_norm, y_pred_without_norm)
accuracy_with_norm = accuracy_score(y_test_with_norm, y_pred_with_norm)

print("Normalize Edilmemiş Veri Doğruluğu:", accuracy_without_norm)
print("Normalize Edilmiş Veri Doğruluğu:", accuracy_with_norm)



# classification report hesaplama
report_without_norm = classification_report(y_test_without_norm, y_pred_without_norm, target_names=['0', '1'])
report_with_norm = classification_report(y_test_with_norm, y_pred_with_norm, target_names=['0', '1'])

# Normalize Edilmemiş Veri için
print("Normalize Edilmemiş Veri Classification Report:")
print(report_without_norm)

# Normalize Edilmiş Veri için
print("Normalize Edilmiş Veri Classification Report:")
print(report_with_norm)



# Confusion Matrix hesaplama
cm_without_norm = confusion_matrix(y_test_without_norm, y_pred_without_norm)
cm_with_norm = confusion_matrix(y_test_with_norm, y_pred_with_norm)

# Confusion Matrix'leri görselleştirme
disp_without_norm = ConfusionMatrixDisplay(confusion_matrix=cm_without_norm, display_labels=[0, 1])
disp_with_norm = ConfusionMatrixDisplay(confusion_matrix=cm_with_norm, display_labels=[0, 1])

# İki plot'u yan yana birleştirme
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Normalize Edilmemiş Veri Confusion Matrix
disp_without_norm.plot(cmap='Blues', values_format='d', ax=axs[0])
axs[0].set_title('Normalize Edilmemiş Veri Confusion Matrix')

# Normalize Edilmiş Veri Confusion Matrix
disp_with_norm.plot(cmap='Blues', values_format='d', ax=axs[1])
axs[1].set_title('Normalize Edilmiş Veri Confusion Matrix')

plt.show()