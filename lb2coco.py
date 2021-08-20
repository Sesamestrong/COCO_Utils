import json
import datetime as dt
import os
from PIL import Image
import requests
import re

def lb_to_json(labeled_file,coco_file,image_dir,cat_order=[]):
    with open(labeled_file,'r') as f:
        labelbox=json.loads(f.read())
    coco={
        'info':None,
        'images':[],
        'annotations':[],
        'licenses':[],
        'categories':[]
    }

    coco['info'] = {
        'year': dt.datetime.now(dt.timezone.utc).year,
        'version': None,
        'description': label_data[0]['Project Name'],
        'contributor': label_data[0]['Created By'],
        'url': 'labelbox.com',
        'date_created': dt.datetime.now(dt.timezone.utc).isoformat()
    }

    existing_images=os.listdir(image_dir)
    used_images=set()

    cat_list=[*cat_order]
    def get_cat_id(name):
        if name not in cat_list:
            cat_list.append(name)
        return cat_list.index(name)+1

    for row in labelbox:
        img_filename=data['External ID']

        if img_filename in used_images: continue

        path=image_dir+'/'+img_filename

        if img_filename in existing_images: 
            buf=open(path,'rb')
        else:
            resp=requests.get(data['Labeled Data'],stream=True)
            resp.raw.decode_content=True
            buf=resp.raw
            with open(path,'wb') as f:
                f.write(buf)
        
        img=Image.open(buf)

        used_images.add(img_filename)

        width,height=img.size

        date=int(re.search("\\d+",img_filename).match)

        coco_image={
            'id':date,
            'width':width,
            'height':height,
            'file_name':img_filename,
            'license':None,
            'coco_url':data['Labeled Data'],
            'date_captured':int(re.search("\\d+",img_filename).match)
        }
        coco['images'].append(coco_image)

        if "objects" in data["Label"]:
            for obj in data["Label"]["objects"]:
                cat=obj['value']
                cat_id=get_cat_id(cat)
                if "polygon" in obj:
                    segm=[]
                    polygon=obj["polygon"]
                    min_x,min_y,max_x,max_y=min([coord["x"] for coord in polygon]),min([coord["y"] for coord in polygon]),max([coord["x"] for coord in polygon]),max([coord["y"] for coord in polygon])
                    for coord in polygon:
                        segm+=[coord["x"],coord["y"]]
                    box=[min_x,min_y,max_x-min_x,max_y-min_y]
                    annot={
                        "bbox":box,
                        "iscrowd":0,
                        "category_id":cat_id,
                        "image_id":date,
                        "id":len(coco["annotations"])+1,
                        "segmentation":segm
                    }
                    annotations.append(annot)
    coco["annotations"]=[{
        "id":get_cat_id(cat),
        "name":cat,
        "supercategory":cat
    } for cat in cat_list]

    with open(coco_file,"w") as f:
        f.write(json.dumps(coco))
