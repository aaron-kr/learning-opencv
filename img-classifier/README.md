# Image Classifier Project

After learning some basic image classification techniques in [Udacity's Computer Vision Nanodegree](), I decided to try to build my own.

## Computer Vision Pipeline

|#| Step | Description | Methods |
|---|---|---|---|
| 1 | Input data | An array of arrays including the image and relevant metadata  | Images or image frames |
| 2 | Pre-processing | Make every image uniform | <ul><li>Scaling</li><li>Change color spaces</li><li>Noise reduction</li></ul> |
| 3 | Select AOI / ROI | <ul><li>AOI = Area of interest</li><li>ROI = Region of Interest</li></ul> | <ul><li>Object detection</li><li>Image segmentation</li></ul> |
| 4 | Feature extraction | Extract data about features | ... |
| 5 | Recognition / Prediction | ... | <ul><li>Object recognition</li><li>Feature matching</li></ul> | 

### Udacity Example

Build an image classifier to determine whether or not an image was shot in the "day" or the "night." For image processing, it is easier to use integers than tags to classify images, so:

* `day = 1`
* `night = 0`

A total of 200 RGB images were used:

* 100 day 
* 100 night
* 60% for training, in `image_dir_training`
* 40% for testing, in `image_dir_test`

### Seasons Classifier

* `winter = 0`
* `spring = 1`
* `summer = 2`
* `autumn = 3`

#### Photos

* Source: <https://pexels.com>
* 30 per season
* 20 in `train` (total 80) 
* 10 in `test` (total 40)

#### Features

* Winter
  * Lots of white
  * Grays, grayscale
  * More blue than any other color
* Spring
* Summer
* Fall
  * Lots of oranges and browns
  * Equal or more than green
  * Darker, not lighter green
  * Muted tones