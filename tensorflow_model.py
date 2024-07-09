import tensorflow as tf
import pandas as pd
import numpy

CSV_COLUMN_NAMES = ['num']
tempString = ""
for i in range(25):
    for j in range(25):
        tempString = f"{i}.{j}"
        CSV_COLUMN_NAMES.append(tempString)
NUMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


train = pd.read_csv(r"C:\Users\ranr9\Desktop\Python\EE\FINAL\numbers_training.csv", names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(r"C:\Users\ranr9\Desktop\Python\EE\FINAL\numbers_test.csv", names=CSV_COLUMN_NAMES, header=0)


train = train.sample(frac=1)


train_y = train.pop("num")
test_y = test.pop("num")


print(train)
print(train_y)



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train, train_y, epochs=50, batch_size=32)




#eval
loss, acc = model.evaluate(test, test_y)
print(acc)


predict = model.predict([test])
for i in range(len(predict)):
    print(numpy.argmax(predict[i]))