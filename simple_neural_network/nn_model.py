from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class NNModel:
    def __init__(self, input_size):
        self.model = self.make_model(input_size)

    def make_model(self, input_size):
        model = Sequential([
            Dense(128, activation="relu", input_shape=(input_size,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def compile_and_train(self, X_train, y_train, X_val, y_val, num_epoch):
        # Build early stopping feature to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=0.001, patience=10, verbose=1, 
            mode='min', restore_best_weights=True)
        # Training
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=num_epoch, 
            batch_size=32,
            callbacks=[early_stopping]
            )
    
    def predict(self, X_test, threshold):
        y_pred_raw = self.model.predict(X_test)
        y_pred = (y_pred_raw >= threshold).astype(int)
        return y_pred
    
    # Obtain evaluation metrics for the model
    def evaluate_model(self, y_pred, y_true):
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy}")
        precision = precision_score(y_true, y_pred)
        print(f"Precision: {precision}")
        recall = recall_score(y_true, y_pred)
        print(f"Recall: {recall}")
        f1 = f1_score(y_true, y_pred)
        print(f"F1 Score: {f1}")
    
    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = load_model(filename)
