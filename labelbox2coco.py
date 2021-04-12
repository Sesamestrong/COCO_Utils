# Labelbox json file to COCO json files
# Based on the LBExporters 0.1.1 file (https://pypi.org/project/LBExporters/), but heavily modified

# Import libraries
import json
import datetime as dt
import logging
import requests
from PIL import Image
import numpy as np

import rasterio
import shapely


def labelbox_to_json(labeled_data, coco_output, images_output_dir):
    
    """
    Converts from labelbox json export format to COCO json
    stores the png files locally
    tested only for instance segmentation datasets

    Args:
        labeled_data: json file exported from labelbox
        coco output: path of json file to store the coco labels
        images_output_dir: folder to put the 

    """


    # read labelbox JSON output
    with open(labeled_data, 'r') as f:
        label_data = json.loads(f.read())

    # setup COCO dataset container and info
    coco = {
        'info': None,
        'images': [],
        'annotations': [],
        'licenses': [],
        'categories': []
    }

    # Include base information about the export
    coco['info'] = {
        'year': dt.datetime.now(dt.timezone.utc).year,
        'version': None,
        'description': label_data[0]['Project Name'],
        'contributor': label_data[0]['Created By'],
        'url': 'labelbox.com',
        'date_created': dt.datetime.now(dt.timezone.utc).isoformat()
    }

    count = 1

    # Go though each labelled image

    for data in label_data:
        # Download and get image name
        try:
            response = requests.get(data['Labeled Data'], stream=True)
        except requests.exceptions.MissingSchema as e:
            logging.exception(('"Labeled Data" field must be a URL. '
                              'Support for local files coming soon'))
            continue
        except requests.exceptions.ConnectionError as e:
            logging.exception('Failed to fetch image from {}'
                              .format(data['Labeled Data']))
            continue

        response.raw.decode_content = True

        # Open image and get image size
        im = Image.open(response.raw)
        width, height = im.size

        # Create an id using consecutive numbers
        image_id = len(coco['images']) + 1

        # Print status
        print('###### Processing ' + data['ID'] + ' image, ' + 'Image ' + str(count) + ' of ' +  str(len(label_data)))
        count = count + 1

        # Write image in png format, is will have the ID as name (e.g. 23.png)
        image_path = images_output_dir + '/' + str(image_id) + '.png'
        im.save(image_path)
        image_name = str(image_id) + '.png'

        # Include only images with annotations
        if not ('objects' in data['Label']):
            print("Image without annotations")
        else:

            # build the file name name (path), ID, dimensions
            image = {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": image_name,
                "license": None,
                "flickr_url": data['ID'],
                "coco_url": image_name,
                "date_captured": None,
            }

            # Write it inbto the images list
            coco['images'].append(image)

            # remove classification labels (Skip, etc...)
            labels = data['Label']
            if not callable(getattr(labels, 'keys', None)):
                continue

            # convert the masks into polygon format
            for category_name, binary_masks in labels.items():
                            
                polygons = []

                count2 = 1

                for mask in binary_masks:

                    print('processing ' + mask['value'] + ' instance. Instance ' + str(count2) + ' of ' + str(len(binary_masks)))
                    count2 = count2+ 1

                    try:
                    # check if label category exists in 'categories' field
                        category_id = [c['id'] for c in coco['categories'] if c['supercategory'] == mask['value']][0]
                    # If it doesnt, create it
                    except IndexError:
                        category_id = len(coco['categories']) + 1
                        category = {
                            'supercategory': mask['value'],
                            'id': category_id,
                            'name': mask['value']
                        }
                        coco['categories'].append(category)

                    # Get the binary mask name
                    try:
                        response = requests.get(mask['instanceURI'], stream=True)
                    except requests.exceptions.MissingSchema as e:
                        logging.exception(('"Labeled Data" field must be a URL. '
                                'Support for local files coming soon'))
                        continue
                    except requests.exceptions.ConnectionError as e:
                        logging.exception('Failed to fetch image from {}'
                                .format(mask['instanceURI']))
                        continue
                    
                    response.raw.decode_content = True

                    # Open the binary mask (it is just a png image with 1 and 0)
                    im = Image.open(response.raw)

                    # Transform to numpy array, as numpy reads columns and rows differently we need to reshape the array
                    im_np = np.array(im)
                    im.np = im_np.reshape(im.size[0], im.size[1],4)

                    # Transform the masks to a listo of polygons
                    all_polygons = mask_to_polygons_layer(im_np)
                    print('Instance consisting in ' + str(len(all_polygons)) + ' polygon')

                    all_segmentation = []

                    # Transform the list of polygons in multipolygons
                    all_polygons_multi = shapely.geometry.MultiPolygon(all_polygons)

                    # Get the coordinates of all the polygons and put them on a list
                    for polygon in all_polygons:

                        segmentation = []

                        for x, y in polygon.exterior.coords:
                            segmentation.extend([x, y])

                        all_segmentation.append(segmentation)

                    # Create the anotation dic, with all the segmentation data
                    annotation = {
                        "id": len(coco['annotations']) + 1,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": all_segmentation,
                        "area": all_polygons_multi.area,  # float
                        "bbox": [all_polygons_multi.bounds[0], all_polygons_multi.bounds[1],
                                all_polygons_multi.bounds[2] - all_polygons_multi.bounds[0],
                                all_polygons_multi.bounds[3] - all_polygons_multi.bounds[1]],
                        "iscrowd": 0
                    }

                    coco['annotations'].append(annotation)

    # Write the coco json file
    with open(coco_output, 'w+') as f:
        f.write(json.dumps(coco))
        print("Image preprocessing ready")

# 
def mask_to_polygons_layer(mask):

    """
    Function to convert from binary mask to polygon

    Args:
        mask: numpy array containing the binary mask

    """

    # Use rasterio to generate shapes of pixels greater than 1 (I am using the first band)
    generator = rasterio.features.shapes(mask[:,:,0].astype(np.int16),connectivity=8, mask=(mask[:,:,0] >0),transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0))
    
    all_polygons = []

    # Put all the polygons in a list
    for poly,value in generator:
        all_polygons.append(shapely.geometry.shape(poly))

    return all_polygons
