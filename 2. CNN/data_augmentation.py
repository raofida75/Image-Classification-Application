import tensorflow as tf


def augmentation(X,y, batch_size):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        shear_range = 45.0,
        zoom_range=[0.5, 1.5],
    )
    
    train_generator = train_datagen.flow(X, y, batch_size=batch_size)
    return train_generator