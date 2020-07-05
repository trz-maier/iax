# IAX

IAX, or Informed Adversarial Examples, is a library created to 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install iax
```

## Usage

```python
import iax

# define iax objects
image = iax.Image(input_array=img, mask=mask, label=3)
engine = iax.engines.PSO(number_of_particles=[15, 15, 15], bounds=[0, 255])
space = iax.Space(image, engine, model.predict)

# perform adversarial search in defined search space
space.search(iterations=30, w=0.8, c1=0.3, c2=0.6, c3=0.9)

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
