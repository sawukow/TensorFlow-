import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import os


def create_model():
    # Загружаем данные
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]


    # Нормализуем данные (делим на 255)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Преобразуем метки в one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Создаем модель
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax'),
    ])

    # Компилируем модель
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Уменьшаем параметры для быстрого обучения
    print("Начало обучения модели...")
    history = model.fit(X_train, y_train,
                        batch_size=64,
                        epochs=3,
                        validation_split=0.1,
                        verbose=1)

    # Оцениваем на тестовых данных
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Точность на тестовых данных: {test_acc:.4f}")

    # Сохраняем модель
    model.save('cifar10_model.h5')
    print("Модель сохранена как 'cifar10_model.h5'")

    return model, history


def main():
    st.title('Классификатор на Cifar10')
    st.write("Загрузи картинку")
    file = st.file_uploader("Загрузи jpg или png", type=["jpg", "png"])

    if file is None:
        st.stop()
    else:
        img = Image.open(file)
        st.image(img, use_column_width=True, caption="Ваше изображение")

        resized = img.resize((32, 32))
        img_array = np.asarray(resized) / 255.0


        st.image(img_array, use_column_width=True)

        img_array = img_array.reshape(1, 32, 32, 3)

        model = tf.keras.models.load_model('cifar10_model.h5')
        prediction = model.predict(img_array, verbose=0)

        classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        fig, ax = plt.subplots(figsize=(10, 6))  # ⭐ Добавил размер графика

        y_pos = np.arange(len(classes))
        ax.barh(y_pos, prediction[0], align='center', alpha=0.7)  # ⭐ Изменил на barh
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes)
        ax.set_xlabel('Вероятность')
        ax.set_title('Распределение вероятностей')

        max_idx = np.argmax(prediction[0])
        st.success(f"**Результат:** {classes[max_idx]} ({prediction[0][max_idx] * 100:.1f}%)")

        st.pyplot(fig)


if __name__ == '__main__':
    if not os.path.exists('cifar10_model.h5'):
        st.write("Обучаю модель... Это займет несколько минут.")
        create_model()
    else:
        st.write("Модель уже обучена")
    main()