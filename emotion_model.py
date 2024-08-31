from keras.models import load_model

def load_emotion_model(model_path='model_file.h5'):
    return load_model(model_path)

if __name__ == "__main__":
    model = load_emotion_model()
