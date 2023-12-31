file_path = 'C:/Users/Nikita/Desktop/Iris/iris.csv'  
df = pd.read_csv(file_path)
df.head()
![image](https://github.com/Niks1612/Iris-Classification/assets/133484285/94d54b1c-472e-4f61-b17c-ae45b24b578a)

X = df.drop('Species', axis=1) 
y = df['Species']
print(df.columns)
![image](https://github.com/Niks1612/Iris-Classification/assets/133484285/212d7d4a-0137-4646-96c9-d1fa5bc24078)

class_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
y = y.map(class_mapping)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(8, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
![image](https://github.com/Niks1612/Iris-Classification/assets/133484285/440c7a88-51fe-4426-8f36-58e7a93033cd)


y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
![image](https://github.com/Niks1612/Iris-Classification/assets/133484285/7c9ecf52-3bd4-4760-955b-f73b9e32cb1f)
