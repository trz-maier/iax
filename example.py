import numpy as np
import iax

model = None
image = iax.Image(input_array=np.array([1, 2, 3]), mask=None, label=1)
engine = iax.engines.PSO(number_of_particles=[10, 10, 10], bounds=[0, 255])
space = iax.Space(image, engine, model.predict)
space.search(iterations=50, w=0.8, c1=0.3, c2=0.6, c3=0.9)
