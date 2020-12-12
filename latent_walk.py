#to start we will load the generator model and begin generating images
from os import makedirs
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from numpy import arccos
from numpy import clip
from numpy import dot
from numpy import sin
from numpy import linspace
from numpy.linalg import norm
from keras.models import load_model
from matplotlib import pyplot
from tqdm import tqdm


model_load = 'generator_model_189.h5'

num_points = 10

interp_res = 60

render_path = 'walk04\\'


#function to generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    #generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    #reshape into the shape of the latent space for input
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input


#latent space is hypersphere?! so we need spherical linear interp (SLERP)
def slerp(val, low, high):
    #no way in hell i can annotate this maths lol
    omega = arccos(clip(dot(low/norm(low), high/norm(high)), -1, 1))
    so = sin(omega)
    if so == 0:
        #l'hopital's rule/LERP
        #(this means if the sine of all that is 0, it's a straight line,
        #i.e. you do a linear interpolation anyway)
        return(1.0-val) * low + val * high
    return sin((1.0-val)*omega) / so * low + sin(val*omega) / so * high


#slerp interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=10):
    #interpolate ratios between the points
    ratios = linspace(0,1, num = n_steps)
    #linear interpolation vectors
    vectors = list()
    for ratio in ratios:
        v = slerp(ratio, p1, p2)
        vectors.append(v)
    return asarray(vectors)


#generate and save a plot of a single latent point, saved with a numerical filename
def render_continuous(g_model, latent_set):
	render = g_model.predict(latent_set)
    #scale from [-1,1] to [0,1]
	render = (render + 1) / 2
	for i in tqdm(range(len(latent_set)), unit=' frames rendered'):
	    
	    #plot render
	    fig = pyplot.figure(figsize=(1,1))
	    ax = pyplot.Axes(fig, [0., 0., 1., 1.])
	    ax.set_axis_off()
	    fig.add_axes(ax)
	    pyplot.imshow(render[i], interpolation='none')
	    
	    #save plot to file
	    filename = render_path + 'frame_%06d.png' % i
	    pyplot.savefig(filename, dpi=render.shape[1])
	    pyplot.close()
	    # print('frame ' + str(i) + ' rendered!')


#take a list of latent points and use slerp() to interpolate between them at the given resolution; return array of all latents in sequence
#NOTE: this will duplicate the individual point latents twice: think 1, 2, 3, 4, 5; 5, 6, 7, 8, 9, 10; 10, 11, etc..
def compile_latents(latent_points, interp_res):
	frame_latents = list()

	for n in range(len(latent_points) - 1):
	    V = interpolate_points(latent_points[n], latent_points[n+1], n_steps=interp_res)
	    for point in V:
	        frame_latents.append(point)
	V = interpolate_points(latent_points[-1], latent_points[0], n_steps=interp_res)
	for point in V:
	        frame_latents.append(point)

	return asarray(frame_latents)


makedirs(render_path, exist_ok=True)

model = load_model(model_load)

latent_points = generate_latent_points(100, num_points)

frame_latents = compile_latents(latent_points, interp_res)


render_continuous(model, frame_latents)

print('\nwalk complete!!')
