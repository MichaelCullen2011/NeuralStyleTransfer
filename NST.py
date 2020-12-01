
'''
Neural Style Transfer   -   Drawing images in the style of another image
'''
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image
import time
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


'''
Logic and Important Variables
'''
epochs = 1
steps_per_epoch = 100

Train = False
Save = False
Show = False

image_to_use = ['Dog']
style_to_use = ['Kandinsky']


def run_nst():
    import tensorflow as tf
    import tensorflow_hub as hub

    global Train, Display, Save, epochs, steps_per_epoch, image_to_use, style_to_use

    '''
    Variables
    '''
    originals_dir = "./images/Originals/"
    image_extension = ".jpg"
    Dog = False
    Jasper = False
    Teasdale1 = False
    Teasdale2 = False
    image_logic = [Dog, Jasper, Teasdale1, Teasdale2]
    image_name = ["Dog", "Jasper", "Teasdale1", "Teasdale2"]

    Kandinsky = False
    Picasso = False
    VanGogh = False
    Portrait = False
    Pollock = False
    Space = False
    style_logic = [Kandinsky, Picasso, VanGogh, Portrait, Space]
    style_name = ["Kandinsky", "Picasso", "VanGogh", "Portrait", "Space"]

    # Setting the image and style to uses name based on above logic
    logic_count = 0
    for image in image_logic:
        if image:
            image_to_use = str(image_name[logic_count])
        logic_count += 1

    logic_count = 0
    for style in style_logic:
        if style:
            style_to_use = str(style_name[logic_count])
        logic_count += 1

    # print("From NST, image to use: ", image_to_use)
    # print("From NST, style to use: ", style_to_use)
    # print(Train, Display, Save)

    def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    '''
    Functions to visualise the input
    '''

    # Limit max dimensions to 512px

    def load_img(path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]

        return img

    # Display the Image

    def show_image(image, title=None):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)

        plt.imshow(image)
        if title:
            plt.title(title)
    print("From NST", '{}-{}.jpg'.format(image_to_use[0], style_to_use[0]))
    '''
    TRAINING SECTION
    '''

    if Train:
        print("Training")
        '''
        Configuring Modules
        '''



        # Loads and displays the Images
        # content_image_set = []
        # for image in image_to_use:
        #     content_image_set.append(load_img(image))
        #
        # style_image_set = []
        # for style in style_to_use:
        #     style_image_set.append(load_img(style))

        if len(image_to_use) and len(style_to_use) == 1:
            content_image = load_img(originals_dir + '/Images/' + image_to_use[0] + image_extension)
            # print(originals_dir + '/Images/' + image_to_use[0] + image_extension)
            style_image = load_img(originals_dir + '/Styles/' + style_to_use[0] + image_extension)
            # plt.subplot(1, 2, 1)
            # show_image(content_image, 'Content Image')
            # plt.subplot(1, 2, 2)
            # show_image(style_image, 'Style Image')
            # plt.show()
        else:
            print("Select only one image and one style")

        '''
        Fast Style Transfer (using tensorflow-hub)
        '''

        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        stylised_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        # tensor_to_image(stylised_image).save('images/dog_fast_style.png')

        '''
        Define Content and Style Representations
        '''
        x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
        x = tf.image.resize(x, (224, 224))
        vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
        prediction_probabilities = vgg(x)
        # print(prediction_probabilities.shape)

        predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
        # print([(class_name, prob) for (number, class_name, prob) in predicted_top_5])

        # Loading a VGG19 without its head to find layer names
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        # print()
        # for layer in vgg.layers:
        #     print(layer.name)

        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']
        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)

        '''
        Build the Model
        '''

        def vgg_layers(layer_names):
            vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
            vgg.trainable = False

            outputs = [vgg.get_layer(name).output for name in layer_names]

            model = tf.keras.Model([vgg.input], outputs)
            return model

        style_extractor = vgg_layers(style_layers)
        style_outputs = style_extractor(style_image*255)

        '''
        Calculate Style and Extracting Style and Content
        '''

        def gram_matrix(input_tensor):
            result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
            input_shape = tf.shape(input_tensor)
            num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
            return result/num_locations

        class StyleContentModel(tf.keras.models.Model):
            def __init__(self, style_layers, content_layers):
                super(StyleContentModel, self).__init__()
                self.vgg = vgg_layers(style_layers + content_layers)
                self.style_layers = style_layers
                self.content_layers = content_layers
                self.num_style_layers = len(style_layers)
                self.vgg.trainable = False

            def call(self, inputs):
                inputs = inputs*255.0
                preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
                outputs = self.vgg(preprocessed_input)
                style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                                  outputs[self.num_style_layers:])

                style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
                content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
                style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

                return {'content': content_dict, 'style': style_dict}

        extractor = StyleContentModel(style_layers, content_layers)
        results = extractor(tf.constant(content_image))

        '''
        Run Gradiant Descent
        '''
        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']
        image = tf.Variable(content_image)

        def clip_0_1(image):
            return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        style_weight = 1e-2
        content_weight = 1e4

        def style_content_loss(outputs):
            style_outputs = outputs['style']
            content_outputs = outputs['content']
            style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                                   for name in style_outputs.keys()])
            style_loss *= style_weight / num_style_layers

            content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                     for name in content_outputs.keys()])
            content_loss *= content_weight / num_content_layers
            loss = style_loss + content_loss
            return loss

        total_variation_weight = 30

        @tf.function()
        def train_step(image):
            with tf.GradientTape() as tape:
                outputs = extractor(image)
                loss = style_content_loss(outputs)

            grad = tape.gradient(loss, image)
            opt.apply_gradients([(grad, image)])
            image.assign(clip_0_1(image))

        '''
        Perform a Long Optimisation
        '''
        start = time.time()
        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                train_step(image)
                print(".", end='')
            display.clear_output(wait=True)
            display.display(tensor_to_image(image))
            print("Train step: {}".format(step))
        end = time.time()
        print("Total time: {:.1f}".format(end-start))

        '''
        Total variation loss
        '''

        def high_pass_x_y(image):
            x_var = image[:,:,1:,:] - image[:,:,:-1,:]
            y_var = image[:,1:,:,:] - image[:,:-1,:,:]

            return x_var, y_var

        x_deltas, y_deltas = high_pass_x_y(content_image)

        # plt.figure(figsize=(14,10))
        # plt.subplot(2,2,1)
        # imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")
        #
        # plt.subplot(2,2,2)
        # imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")
        #
        # x_deltas, y_deltas = high_pass_x_y(image)
        #
        # plt.subplot(2,2,3)
        # imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")
        #
        # plt.subplot(2,2,4)
        # imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")

        def total_variation_loss(image):
            x_deltas, y_deltas = high_pass_x_y(image)
            return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

        tf.image.total_variation(image).numpy()

        '''
        Rerun Optimisation
        '''
        total_variation_weight = 30

        @tf.function()
        def train_step(image):
            with tf.GradientTape() as tape:
                outputs = extractor(image)
                loss = style_content_loss(outputs)
                loss += total_variation_weight*tf.image.total_variation(image)

            grad = tape.gradient(loss, image)
            opt.apply_gradients([(grad, image)])
            image.assign(clip_0_1(image))

        image = tf.Variable(content_image)
        start = time.time()
        epochs = 10
        steps_per_epoch = 100
        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                train_step(image)
                print(".", end='')
            display.clear_output(wait=True)
            display.display(tensor_to_image(image))
            print("Train step: {}".format(step))
        end = time.time()
        print("Total time: {:.1f}".format(end-start))

    '''
    Save the Results
    '''
    generated_dir = './images/Generated/'
    if Save:
        global Show
        # print("SAVING")
        file_name = '{}-{}.jpg'.format(image_to_use[0], style_to_use[0])
        print(file_name)
        tensor_to_image(image).save(generated_dir + file_name)
        Show = True



