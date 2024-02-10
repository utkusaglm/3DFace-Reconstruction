import face_alignment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import sys
import argparse
import json
from PIL import Image


# Optionally set detector and some additional detector parameters


def get_landmarks_from_image(image_path):
    input_img = image_path
    face_detector = 'sfd'
    face_detector_kwargs = {
        "filter_threshold" : 0.8
    }
    # Run the 3D face alignment on a test image, without CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cpu', flip_input=True,
                                      face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)
    try:
        preds = fa.get_landmarks(input_img)[-1]
    except Exception as e:
        print('error: ', e)
        sys.exit(1)

    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {"face": pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                  "eyebrow1": pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                  "eyebrow2": pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                  "nose": pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                  "nostril": pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                  "eye1": pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                  "eye2": pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                  "lips": pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                  "teeth": pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                  }
    axis_on_2d = {}
    axis_on_3d = {}
    for k,v in pred_types.items():
        axis_on_2d[k] = {"x":preds[v.slice, 0], "y":preds[v.slice, 1]}
    for k,v in pred_types.items():
        axis_on_3d[k] = {"x":preds[v.slice, 0] * 1.2, "y":preds[v.slice, 1], "z":preds[v.slice, 2]}
    #write to file

    return axis_on_2d, axis_on_3d
    #filename = input_img[input_img.find("/") + 1:input_img.find(".jpeg")]

    # with open(f"landmark_2d_{filename}.txt", 'w') as f:
    #     f.write(str(axis_on_2d))
    # with open(f"landmark_3d_{filename}.txt", 'w') as f:
    #     f.write(str(axis_on_3d))


def get_list_of_points(data):
    points = []
    for k, value in data.items():
        points += list(zip(value['x'], value['y']))
    return points


def writePoints(points,landmarks_file_path):
    with open(landmarks_file_path, 'w') as file:
        for point in points:
            file.write(f'{point[0]} {point[1]}\n')


if __name__ == '__main__':
    # Run parser
    parser = argparse.ArgumentParser(description='Parameter Processing')
    #no name raise error
    parser.add_argument('--name', type=str,  help='image_name',)
    if len(sys.argv) == 1:
        #help
        print('Give image path as argument --name')
        sys.exit(1)
    image_path = parser.parse_args().name
    landmarks_file_path = image_path.split('/')[-1].split('.')[0]+".txt"
    #to run it python3 landmark_script.py --name images/emma.jpeg
    data, _ = get_landmarks_from_image(image_path)
    landmark_points = get_list_of_points(data)
    writePoints(landmark_points,landmarks_file_path)


    # Load your image using PIL
    image = Image.open(image_path)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Mark points on the image with red color
    for point in landmark_points:
        ax.add_patch(patches.Circle((point[0], point[1]), radius=1, color='red'))

    # Show the plot
    plt.show()