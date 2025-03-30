from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.utils import save_img
import tensorflow as tf
import numpy as np

#Adjust TARGET SIZE
target_size=(600,600)

tf.config.optimizer.set_jit(True)

def load_and_preprocess_image(image,target_size=target_size):
    img = load_img(image,target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0) #adding batch dimension for VGG19 compatibility #(224,244,3) -> (1,224,224,3)
    img = tf.keras.applications.vgg19.preprocess_input(img) #preprocessing for VGG19
    return img

def deprocess_image(image):
    img = image[0] #removes batch dimension
    img += [103.939, 116.779, 123.68]  #Reverses VGG19-specific preprocessing (mean subtraction) to restore original pixel values
    img = np.flip(img, axis=-1) #Converts BGR to RGB
    img = np.clip(img,0,255).astype("uint8") #After various operations, pixel values might have gone outside the valid range, hence adjusting
    return img



content_layer = "block5_conv2"  #deeper layer because we need the main object and its location
style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"] #mutiple early layers because we need the art style and texture, not any specific object
num_style_layers = len(style_layers)



def build_model():
    vgg = VGG19(weights="imagenet", include_top=False)
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in style_layers + [content_layer]]
    return Model([vgg.input],outputs)

def compute_content_loss(base,generated):
    return tf.reduce_mean(tf.square(base - generated))

def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    tensor = tf.reshape(tensor,(-1,channels))
    gram = tf.matmul(tensor,tensor,transpose_a=True)
    return gram

def compute_style_loss(base,generated):
    base_gram = gram_matrix(base)
    generated_gram = gram_matrix(generated)
    return tf.reduce_mean(tf.square(base_gram - generated_gram))

def total_variation_loss(image):
    x_deltas = image[:, :-1, 1:, :] - image[:, :-1, :-1, :]
    y_deltas = image[:, 1:, :-1, :] - image[:, :-1, :-1, :]
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

def compute_loss(model, generated_image, base_content, base_style, content_weight, style_weight, tv_weight):
    outputs = model(generated_image)

    generated_content = outputs[num_style_layers:]
    content_loss = compute_content_loss(base_content[0], generated_content[0])
    content_loss = content_weight * content_loss

    generated_style = outputs[:num_style_layers]
    style_loss = tf.add_n([compute_style_loss(base_style[i], generated_style[i]) for i in range(num_style_layers)])
    style_loss *= style_weight / num_style_layers

    tv_loss = tv_weight * total_variation_loss(generated_image)

    total_loss = content_loss + style_loss + tv_loss
    return total_loss



content_image = load_and_preprocess_image("content.jpg")
style_image = load_and_preprocess_image("style.jpg")

model = build_model()

outputs = model(content_image)
base_content = outputs[num_style_layers:]

outputs = model(style_image)
base_style = outputs[:num_style_layers]

generated_image = tf.Variable(content_image,dtype=tf.float32)

optimizer = tf.optimizers.Adam(learning_rate=0.02)



#Adjust HYPERPARAMETERS
content_weight = 1
style_weight = 1e4
tv_weight = 10
iterations = 17000 #optimal at 17000, highly artistic at 30000


for i in range(iterations):
    with tf.GradientTape() as tape:
        loss = compute_loss(model,generated_image,base_content,base_style, content_weight, style_weight, tv_weight)

        gradients = tape.gradient(loss,generated_image)
        optimizer.apply_gradients([(gradients,generated_image)])

        #cliping generated image to maintain pixel values within range
        generated_image.assign(tf.clip_by_value(generated_image, -103.939, 255 - 103.939))

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.numpy()}")


final_image = deprocess_image(generated_image)
save_img("output.jpg",final_image)