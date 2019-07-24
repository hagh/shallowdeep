from PIL import Image
import os
import tf_utils as utils

# https://www.blog.pythonlibrary.org/2017/10/12/how-to-resize-a-photo-with-python/
def resize_image(input_image_path,
                 output_image_path,
                 size, show = False):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    print('The original image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))
 
    resized_image = original_image.resize(size)
    width, height = resized_image.size
    print('The resized image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))
    if show:
        resized_image.show()
    resized_image.save(output_image_path)

# Resize images
def resize_images(sourceFolder, destFolder, size):
    for imageName in os.listdir(sourceFolder):
        print('Processing ' + imageName)
        resize_image(sourceFolder + "\\" + imageName,
            destFolder + "\\" + imageName,
            size)

# Generate resized images
def generate_resized_images(size):
    sourceFolder = 'E:\\Projects\\Hach2019\\Data\\ImageDataSetOrg\\NonePlate'
    destFolder = 'E:\\Projects\\Hach2019\\Data\\ImageDataSetS' + str(size) + '\\NonePlate'
    resize_images(sourceFolder=sourceFolder,
        destFolder=destFolder,
        size=(size, size))

    sourceFolder = 'E:\\Projects\\Hach2019\\Data\\ImageDataSetOrg\\Plate'
    destFolder = 'E:\\Projects\\Hach2019\\Data\\ImageDataSetS' + str(size) + '\\Plate'
    resize_images(sourceFolder=sourceFolder,
        destFolder=destFolder,
        size=(size, size))
