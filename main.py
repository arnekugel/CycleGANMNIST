import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import save_img
import os
import shutil
from PIL import Image
import time
import random
import sys


from layers import *
from model import *

img_height = 28
img_width = 28
img_layer = 1
img_size = img_height * img_width

to_train = True
to_test = False
to_restore = False
output_path = "./output"
check_dir = "./output/checkpoints/"


temp_check = 0



max_epoch = 1
max_images = 600

max_images_test= 100

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 1
pool_size = 50
sample_size = 10
save_training_images = False
ngf = 32
ndf = 64




# Load data from keras.datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def add_salt_and_pepper_noise(images, prob=0.0001):
    """
    Add salt and pepper noise to a batch of images.
    images: shape (N, H, W)
    prob: probability of noise
    Returns: noisy images with same shape
    """
    noisy_images = images.copy()

    # Create salt & pepper masks
    salt = np.random.rand(*images.shape) < (prob / 2)
    pepper = np.random.rand(*images.shape) < (prob / 2)

    noisy_images[salt] = 255
    noisy_images[pepper] = 0

    return noisy_images.astype(np.uint8)


def scale_images_to_minus1_1(images):
    """
    Scale images with pixel values in [0,255] to [-1,1].
    images: shape (N, H, W), dtype uint8 or float
    Returns: scaled images, dtype float32
    """
    images = images.astype(np.float32) / 255.0  
    images = images * 2.0 - 1.0                
    return images


def rescale_single_image_to_0_255(image):
        """
        Rescale a single image from [-1,1] back to [0,255].
        image: numpy array of shape (H,W) or (H,W,1), dtype float32/float64, values in [-1,1]
        Returns: uint8 image in [0,255]
        """
        image_rescaled = (image + 1.0) / 2.0 * 255.0  
        return np.clip(image_rescaled, 0, 255).astype(np.uint8)



x_train_pepper_salt = add_salt_and_pepper_noise(x_train, prob=0.02)
x_test_pepper_salt = add_salt_and_pepper_noise(x_test, prob=0.02)




class CycleGAN():

    def input_setup(self):

        ''' 
        This function basically setup variables for taking image input.

        filenames_A/filenames_B -> takes the list of all training images
        self.image_A/self.image_B -> Input image with each values ranging from [-1,1]
        '''

        def preprocess(image):
            # Add channel dimension and normalize to [-1, 1]
            image = tf.expand_dims(image, axis=-1)          
            image = tf.cast(image, tf.float32)
            image = image * 2.0 - 1.0             
            return image

        if(to_train):
            train_A = x_train
            train_B = x_train_pepper_salt
        else:
            train_A = x_test
            train_B = x_test_pepper_salt
        

         

        # Create tf.data.Dataset from NumPy arrays
        dataset_A = tf.compat.v1.data.Dataset.from_tensor_slices(train_A)
        dataset_B = tf.compat.v1.data.Dataset.from_tensor_slices(train_B)

        # Apply preprocessing
        dataset_A = dataset_A.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset_B = dataset_B.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)


        # shuffle 
        #dataset_A = dataset_A.shuffle(buffer_size=20000).batch(1)
        #dataset_B = dataset_B.shuffle(buffer_size=20000).batch(1)


        # Create iterators for TF 1.x compatibility
        self.image_A = dataset_A.make_one_shot_iterator().get_next()
        self.image_B = dataset_B.make_one_shot_iterator().get_next()

        # Save dataset lengths
        self.queue_length_A = len(x_train)
        self.queue_length_B = len(x_train_pepper_salt)

    



    def input_read(self, sess):
        '''
        It reads the input into from the image folder.
        '''

        num_files_A = self.queue_length_A
        num_files_B = self.queue_length_B

        if to_train:
            size = max_images
        else:
            size = max_images_test


        self.fake_images_A = np.zeros((pool_size,1,img_height, img_width, img_layer))
        self.fake_images_B = np.zeros((pool_size,1,img_height, img_width, img_layer))

        self.A_input = np.zeros((size, batch_size, img_height, img_width, img_layer))
        self.B_input = np.zeros((size, batch_size, img_height, img_width, img_layer))

        for i in range(size): 
            try:
                image_tensor = sess.run(self.image_A)
                if image_tensor.size == img_size*batch_size*img_layer:
                    self.A_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))
            except tf.errors.OutOfRangeError:
                break

        for i in range(size):
            try:
                image_tensor = sess.run(self.image_B)
                if image_tensor.size == img_size*batch_size*img_layer:
                    self.B_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))
            except tf.errors.OutOfRangeError:
                break




    def model_setup(self):

        ''' This function sets up the model to train

        self.input_A/self.input_B -> Set of training images.
        self.fake_A/self.fake_B -> Generated images by corresponding generator of input_A and input_B
        self.lr -> Learning rate variable
        self.cyc_A/ self.cyc_B -> Images generated after feeding self.fake_A/self.fake_B to corresponding generator. This is use to calcualte cyclic loss
        '''

        self.input_A = tf.compat.v1.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
        self.input_B = tf.compat.v1.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")
        
        self.fake_pool_A = tf.compat.v1.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_A")
        self.fake_pool_B = tf.compat.v1.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_B")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.num_fake_inputs = 0

        self.lr = tf.compat.v1.placeholder(tf.float32, shape=[], name="lr")

        with tf.compat.v1.variable_scope("Model") as scope:
            self.fake_B = build_generator_MNIST(self.input_A, name="g_A")
            self.fake_A = build_generator_MNIST(self.input_B, name="g_B")
            self.rec_A = build_discriminator_MNIST(self.input_A, "d_A")
            self.rec_B = build_discriminator_MNIST(self.input_B, "d_B")

            scope.reuse_variables()

            self.fake_rec_A = build_discriminator_MNIST(self.fake_A, "d_A")
            self.fake_rec_B = build_discriminator_MNIST(self.fake_B, "d_B")
            self.cyc_A = build_generator_MNIST(self.fake_B, "g_B")
            self.cyc_B = build_generator_MNIST(self.fake_A, "g_A")

            scope.reuse_variables()

            self.fake_pool_rec_A = build_discriminator_MNIST(self.fake_pool_A, "d_A")
            self.fake_pool_rec_B = build_discriminator_MNIST(self.fake_pool_B, "d_B")

    def loss_calc(self):

        ''' In this function we are defining the variables for loss calcultions and traning model

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Various trainer for above loss functions
        *_summ -> Summary variables for above loss functions'''

        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        
        disc_loss_A = tf.reduce_mean(tf.compat.v1.squared_difference(self.fake_rec_A,1))
        disc_loss_B = tf.reduce_mean(tf.compat.v1.squared_difference(self.fake_rec_B,1))
        
        g_loss_A = cyc_loss*5 + disc_loss_B
        g_loss_B = cyc_loss*5 + disc_loss_A



        d_loss_A = (tf.reduce_mean(tf.square(self.fake_pool_rec_A)) + tf.reduce_mean(tf.compat.v1.squared_difference(self.rec_A,1)))/2.0
        d_loss_B = (tf.reduce_mean(tf.square(self.fake_pool_rec_B)) + tf.reduce_mean(tf.compat.v1.squared_difference(self.rec_B,1)))/2.0
    

        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, beta1=0.5)

        self.model_vars = tf.compat.v1.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]
        
        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        for var in self.model_vars: print(var.name)

        #Summary variables for tensorboard

        self.g_A_loss_summ = tf.compat.v1.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.compat.v1.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.compat.v1.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.compat.v1.summary.scalar("d_B_loss", d_loss_B)

    def save_training_images(self, sess, epoch):
        """
        This function saves the output images
        """
        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")

        for i in range(0,10):
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})
            save_img("./output/imgs/fakeB_"+ str(epoch) + "_" + str(i)+".jpg",(rescale_single_image_to_0_255(fake_A_temp[0])))
            save_img("./output/imgs/fakeA_"+ str(epoch) + "_" + str(i)+".jpg",(rescale_single_image_to_0_255(fake_B_temp[0])))
            save_img("./output/imgs/cycA_"+ str(epoch) + "_" + str(i)+".jpg",(rescale_single_image_to_0_255(cyc_A_temp[0])))
            save_img("./output/imgs/cycB_"+ str(epoch) + "_" + str(i)+".jpg",(rescale_single_image_to_0_255(cyc_B_temp[0])))
            save_img("./output/imgs/inputA_"+ str(epoch) + "_" + str(i)+".jpg",(rescale_single_image_to_0_255(self.A_input[i][0])))
            save_img("./output/imgs/inputB_"+ str(epoch) + "_" + str(i)+".jpg",(rescale_single_image_to_0_255(self.B_input[i][0])))

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        ''' This function saves the generated image to corresponding pool of images.
        In starting. It keeps on feeling the pool till it is full and then randomly selects an
        already stored image and replace it with new one.'''

        if(num_fakes < pool_size):
            fake_pool[num_fakes] = fake
            return fake
        else :
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0,pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else :
                return fake


    def train(self):


        ''' Training Function '''


        # Load Dataset from the dataset folder
        self.input_setup()  

        #Build the network
        self.model_setup()

        #Loss function calculations
        self.loss_calc()



        # Initializing the global variables
        init = tf.compat.v1.global_variables_initializer()
        saver = tf.compat.v1.train.Saver()     

        with tf.compat.v1.Session() as sess:
            sess.run(init)

            #Read input to nd array
            
            self.input_read(sess)

            #Restore the model to run the model from last checkpoint
            if to_restore:
                chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.compat.v1.summary.FileWriter("./output/2")

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            # Training Loop
            for epoch in range(sess.run(self.global_step),100):                
                print ("In the epoch ", epoch)
                saver.save(sess,os.path.join(check_dir,"cyclegan"),global_step=epoch)

                # Dealing with the learning rate as per the epoch number is not used in this setup
                if(epoch < 5) :
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002 - 0.0002*(epoch-5)/5

                

                if(save_training_images):
                    self.save_training_images(sess, epoch)

                # sys.exit()

                for ptr in range(0,max_images):
                    #print("In the iteration ",ptr)
                    #print("Starting",time.time()*1000.0)

                    # Optimizing the G_A network

                    _, fake_B_temp, summary_str = sess.run([self.g_A_trainer, self.fake_B, self.g_A_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})
                    
                    writer.add_summary(summary_str, epoch*max_images + ptr)                    
                    fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)
                    
                    # Optimizing the D_B network
                    _, summary_str = sess.run([self.d_B_trainer, self.d_B_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_B:fake_B_temp1})
                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    
                    
                    # Optimizing the G_B network
                    _, fake_A_temp, summary_str = sess.run([self.g_B_trainer, self.fake_A, self.g_B_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})

                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    
                    
                    fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, summary_str = sess.run([self.d_A_trainer, self.d_A_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_A:fake_A_temp1})

                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    
                    self.num_fake_inputs+=1
            
                        

                sess.run(tf.compat.v1.assign(self.global_step, epoch + 1))

            writer.add_graph(sess.graph)
            print("Training done")

    def test(self):


        ''' Testing Function'''

        print("Testing the results")

        self.input_setup()

        self.model_setup()
        saver = tf.compat.v1.train.Saver()
        init = tf.compat.v1.global_variables_initializer()

        with tf.compat.v1.Session() as sess:

            sess.run(init)

            self.input_read(sess)

            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            saver.restore(sess, chkpt_fname)

            if not os.path.exists("./output/imgs/test/"):
                os.makedirs("./output/imgs/test/")            

            for i in range(0,100):
                fake_A_temp, fake_B_temp = sess.run([self.fake_A, self.fake_B],feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})
                save_img("./output/imgs/test/fakeB_"+str(i)+".jpg",(rescale_single_image_to_0_255(fake_A_temp[0])))
                save_img("./output/imgs/test/fakeA_"+str(i)+".jpg",(rescale_single_image_to_0_255(fake_B_temp[0])))
                save_img("./output/imgs/test/inputA_"+str(i)+".jpg",(rescale_single_image_to_0_255(self.A_input[i][0])))
                save_img("./output/imgs/test/inputB_"+str(i)+".jpg",(rescale_single_image_to_0_255(self.B_input[i][0])))


def main():
    
    model = CycleGAN()
    if to_train:
        model.train()
    elif to_test:
        model.test()

if __name__ == '__main__':
    main()