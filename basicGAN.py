#basic keras GAN using celebA

from os import makedirs
from numpy import load
from numpy import zeros
from numpy import ones
# from PIL import Image
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot


#DEFINING FUNCTIONS BELOW

#define and compile standalone discriminator model per above
def define_discriminator(in_shape=(80,80,3)):
    #all of these 'functions' are from keras, see the first cell above
    model = Sequential()
    #normal input start
    model.add(Conv2D(128, (5,5), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    #downsample to 40x40
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #downsample to 20x20
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #downsample to 10x10
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #downsample to 5x5
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #flatten and classify
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
    #COMPILE MODEL
    
    #optimizer
    opt = Adam(lr=0.0002, beta_1=0.5)
    #compile
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


#define and DON'T compile the standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    #foundation for 5x5 feature maps  per above
    n_nodes = 128 * 5 * 5
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((5, 5, 128)))
    #upsample to 10x10
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #upsample to 20x20
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #upsample to 40x40
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #upsample to 80x80
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #output conv2D layer to get RGB?! -> 80x80x3
    model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
    
    return model


#define and compile a new combined generator--discriminator model,
#the purpose of which is to update the generator kernal (weights?)
def define_gan(g_model, d_model):
    #make weights in the discriminator not trainable
    d_model.trainable = False
    #connect the two component models into a new one
    model = Sequential()
    #add generator
    model.add(g_model)
    model.add(d_model)
    #compile umbrella model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


#import and normalize the training dataset
def load_real_samples():
    #load the dataset
    data = load('align-80px_celebA_10000set.npz')
    X = data['arr_0']
    #convert from unsigned ints to floats
    X = X.astype('float32')
    #map (normalize) data from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X


#function to return random batch of real image samples
def generate_real_samples(dataset, n_samples):
    #select random instances
    #NOTE: constructor randint(low, high, size)
    ix = randint(0, dataset.shape[0], n_samples)
    #retrieve selected images
    X = dataset[ix]
    #generate 'real' class labels (literally '1')
    #NOTE: constructor ones(shape, dtype)
    #which means that the (()) here means that we are specifying an array
    #as the shape of the ones-array....slightly odd to see (x,1) but hey
    y = ones((n_samples, 1))
    #return the random real samples in the form of the images, plus
    #a single ones-array indicating their veracity
    return X, y


#make random points in (generator) latent space
#as seed inputs for generator
def generate_latent_points(latent_dim, n_samples):
    #generate points in the latent space
    #NOTE: constructor randn(d0, d1, ... dn) so the shape itself, no other params
    x_input = randn(latent_dim * n_samples)
    #reshape into a batch of inputs for the network
    #NOTE: constructor numpy.reshape(newshape[int or tuple])
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

#NOTA BENE: strangely enough, this function conjures a horribly
#undifferentiated, 1D string of numbers (latent_dim * n_samples) long,
#THEN 'reshapes' it into an array with n_samples as the first dimension..


#use the generator to generate n fake images/samples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    #generate points in latents space via above function
    x_input = generate_latent_points(latent_dim, n_samples)
    #predict outputs
    #???? i have no idea what this means but it seems to simply run
    #the model without compiling it..? LOOK INTO THIS!!
    X = g_model.predict(x_input)
    #generate 'fake' class labels (literally '0')
    #NOTE: as above, the double parens indicate an array passed as
    #the first functional argument, i.e. the shape of the zeros array
    y = zeros((n_samples, 1))
    return X, y


#function to TRAIN THE GAN MODEL! per above
def train(g_model, d_model, gan_model, dataset, latent_dim, render_latent, n_epochs=200, n_batch=150):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    
    #manually loop epochs
    for i in range(n_epochs):
        #loop batches over the training set
        for j in range(bat_per_epo):
            #get half batch of random real samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            #update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            #generate half batch of fake samples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            #update discriminator model weights again
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            #prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            #create inverted labels for the fake samples (i dont get this)
            y_gan = ones((n_batch, 1))
            #update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            
            #summarize loss on this batch
            print('>>>>%d, %d/%d, d1=%.3f, d2=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            
            #render single latent
            render_plot(g_model, render_latent, i, j)
            
        #evaluate model performance, sometimes
        if (i+1) % 1 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


#create and save a plot of generated images
def save_plot(examples, epoch, n=10):
    #scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    #plot images
    for i in range(n * n):
        #define subplot
        pyplot.subplot(n, n, 1 + i)
        #turn off axes
        pyplot.axis('off')
        #plot raw pixel data
        pyplot.imshow(examples[i])
    
    #save plot to file
    filename = plot_path + 'generated_plot_e%03d.png' % epoch
    pyplot.savefig(filename)
    pyplot.close()


#create and save a plot of a single latent point, saved with ###e### filename
def render_plot(g_model, render_latent, epoch, batch):
    #as in generate_fake_samples()
    render = g_model.predict(render_latent)
    #scale from [-1,1] to [0,1]
    render = (render + 1) / 2
    #plot render
    fig = pyplot.figure(figsize=(1,1))
    ax = pyplot.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    pyplot.imshow(render[0], interpolation='none')
    
    #save plot to file
    filename = render_path + 'render_%03de%03d.png' % (epoch, batch)
    pyplot.savefig(filename, dpi=render.shape[1])
    pyplot.close()


#function to evaluate the discriminator, plot generated images, and save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    #prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    #evaluate discriminator on real samples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    #prepare fake samples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    #evaluate discriminator on fake samples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    #summarize and print
    print('discriminator accuracy: real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    #save plot
    save_plot(x_fake, epoch)
    #save generator model tile file
    filename = 'generator_model_%03d.h5' % epoch
    g_model.save(filename)



#THE ACTUAL RUNNING OF THE THING

#size of latent space
latent_dim = 100
#create discriminator
d_model = define_discriminator()
#create generator
g_model = define_generator(latent_dim)
#create gan umbrella model
gan_model = define_gan(g_model, d_model)
#load dataset
dataset = load_real_samples()

#plot path
plot_path = 'plot04\\'
makedirs(plot_path, exist_ok=True)

#render path
render_path = 'render04\\'
makedirs(render_path, exist_ok=True)
#render latent definition
render_latent = generate_latent_points(latent_dim, 1)

#TRAIN!!!!
train(g_model, d_model, gan_model, dataset, latent_dim, render_latent)