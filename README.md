# iax 
 
`iax` [ˈaːjɑks], or Informed Adversarial Examples, is a library created to simplify the search of adversarial examples within a masked area of an image in a black-box scenario. The library is model agnostic, i.e. it can be used with any predictive model which implements a `predict` method.

## Installation

Install `iax` directly from github.

```bash
pip install https://github.com/trz-maier/iax.git
```

<br>

If the above fails I recommend you clone the repository to your local directory 

```bash
git clone https://github.com/trz-maier/iax
```
and install it into your local environment
```bash
python iax\setup.py bdist_wheel
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

## Project
This repository is part of an MSc thesis titled *"Informed Adversarial Attacks with Explainable AI and Metaheuristics in Medical Imaging"* submitted for the award of MSc in Data Science at Heriot-Watt University.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
