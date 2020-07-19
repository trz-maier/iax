# iax 
 
`iax` [ˈaːjɑks], or Informed Adversarial Examples, is a library created to simplify the search of adversarial examples within a masked area of an image in a black-box scenario. The library is model agnostic, i.e. it can be used with any predictive model which implements a `predict` method.

## Installation

Install `iax` directly from github.

```bash
pip install https://github.com/trz-maier/iax.git
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

```
def get_mask(model, image, n):
    """
    Generates Lime Mask
    :param model: classifier object
    :param image: image array
    :param n: number of features to return
    :return: mask as a 2-d array of booleans
    """
    explainer = lime.lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image, model.predict, 
                                             top_labels=1, hide_color=0, num_samples=200)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, 
                                                num_features=n, hide_rest=False)
    return mask.astype(bool)
```

```
def box_out_mask(mask: np.array, pixel_expand: int = 0) -> np.array:
    """
    Function boxes out the mask overs its far-most edges
    :param mask: boolean mask in form of a 2-d array
    :param pixel_expand: value by which to expand the resulting box in each dimension
    :return: new mask as a 2-d array
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    # expand box by the number of pixels provided in each direction where possible
    r_min = max(0, r_min - pixel_expand)
    c_min = max(0, c_min - pixel_expand)
    r_max = min(mask.shape[0], r_max + pixel_expand)
    c_max = min(mask.shape[1], c_max + pixel_expand)

    out = mask.copy()
    out[r_min:r_max, c_min:c_max] = True

    return out
```

## Project
This repository is part of an MSc thesis titled *"Informed Adversarial Attacks with Explainable AI and Metaheuristics in Medical Imaging"* submitted for the award of MSc in Data Science at Heriot-Watt University.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
