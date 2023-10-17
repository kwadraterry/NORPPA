# from segmentation.detectron_segment import detectron_segment
from segmentation.seem.seem_segment import seem_segment


def add_instance_info(label, instance, num_instances):
    if type(label) is dict:
        label["instance"] = instance
        label["num_instances"] = num_instances
        return label
    else:
        return (label, instance)

def segment(input, predictor, instance_segmentation=False):
    image, img_label = input
    model = predictor["model"]
    transform = predictor["transform"]
    species = predictor["species"]
    if image is None:
        return [input]

    result_images = seem_segment(image, model, transform, species)
    # result_images, num_instances = detectron_segment(predictor, image, instance_segmentation)
    # if instance_segmentation:
    #     return [(img, add_instance_info(img_label, i, num_instances)) for img,i in enumerate(result_images)]
    # else:
    return [(img, img_label) for img in result_images]
