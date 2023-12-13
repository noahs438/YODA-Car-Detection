import torch
from YodaModel import YodaModel
import os
import cv2
import argparse
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet18_Weights
import matplotlib.pyplot as plt
from KittiDataset import KittiDataset
from KittiAnchors import Anchors


def calc_IoU(boxA, boxB):
    # print('break 209: ', boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][1], boxB[0][1])
    yA = max(boxA[0][0], boxB[0][0])
    xB = min(boxA[1][1], boxB[1][1])
    yB = min(boxA[1][0], boxB[1][0])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1][1] - boxA[0][1] + 1) * (boxA[1][0] - boxA[0][0] + 1)
    boxBArea = (boxB[1][1] - boxB[0][1] + 1) * (boxB[1][0] - boxB[0][0] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def calc_max_IoU(ROI, ROI_list):
    max_IoU = 0
    for i in range(len(ROI_list)):
        max_IoU = max(max_IoU, calc_IoU(ROI, ROI_list[i]))
    return max_IoU


def mean_IoU(all_cars, roi_list, indexes):
    mean = 0
    for i in range(len(indexes)):
        max_IoU = 0
        for j in range(len(all_cars[i])):
            max_IoU = max(max_IoU, calc_max_IoU(all_cars[i][j], roi_list[i]))
        mean += max_IoU
    mean /= len(indexes)
    return mean


if __name__ == "__main__":
    print("Starting the YODA tests")

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='input_dir', type=str, required=True, help='input dir (./)')
    parser.add_argument('-o', metavar='output_dir', type=str, required=True, help='output dir (./)')
    parser.add_argument('-iter', metavar='iterations', type=int, default=10, help='number of iterations')
    args = parser.parse_args()

    input_dir = None
    if args.i is not None:
        input_dir = args.i

    output_dir = None
    if args.o is not None:
        output_dir = args.o

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device ', device)

    dataset = KittiDataset(input_dir, training=False)
    model = YodaModel(num_classes=2, weights=ResNet18_Weights.DEFAULT).to(device)
    model.load_state_dict(torch.load('yoda_classifier.pth'))

    car_list = []
    all_car_list = []
    iteration_count = 0

    for item in enumerate(dataset):
        if iteration_count >= args.iter:
            break
        idx = item[0]
        image = item[1][0]
        label = item[1][1]
        # print(i, idx, label)

        idx = dataset.class_label['Car']
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)
        car = image.copy()
        car_list.append(car_ROIs)

        for box in car_ROIs:
            pt1 = (box[0][1], box[0][0])
            pt2 = (box[1][1], box[1][0])
            cv2.rectangle(car, pt1, pt2, color=(0, 255, 255))

        # # Show the image
        # cv2.imshow('Car', car)
        # cv2.waitKey(0)

        anchors = Anchors()
        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)

        image_boxes = image.copy()
        for box in boxes:
            pt1 = (box[0][1], box[0][0])
            pt2 = (box[1][1], box[1][0])
            cv2.rectangle(image_boxes, pt1, pt2, color=(0, 255, 255))
        # File name from idx
        cv2.imwrite(os.path.join(output_dir, str(idx) + '_boxes.png'), image_boxes)

        # Show the image
        cv2.imshow('Boxes', image_boxes)
        cv2.waitKey(0)

        k_list = []
        for k in ROIs:
            k = cv2.resize(k, (150, 150))
            k = transforms.ToTensor()(k)
            k = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(k)
            k_list.append(k)

        roi_stack = torch.stack(k_list)
        roi_stack = roi_stack.to(device)

        model.eval()
        with torch.no_grad():
            output = model(roi_stack).to(device)

        detected_list = []
        for j in range(len(boxes)):
            if output[j, 1].item() > 0.6:
                detected_list.append(boxes[j])

        detected_indexes = []
        if len(detected_list) > 0:
            all_car_list.append(detected_list)
            detected_indexes.append(idx)

        image_detected = image.copy()
        for box in detected_list:
            pt1 = (box[0][1], box[0][0])
            pt2 = (box[1][1], box[1][0])
            cv2.rectangle(image_detected, pt1, pt2, color=(0, 255, 255))

        # Show the image
        cv2.imshow('Detected', image_detected)
        cv2.waitKey(0)

        car_copy = car.copy()
        for box in detected_list:
            cv2.rectangle(car_copy, (box[0][1], box[0][0]), (box[1][1], box[1][0]), color=(0, 255, 255))
        for box in car_ROIs:
            cv2.rectangle(car_copy, (box[0][1], box[0][0]), (box[1][1], box[1][0]), color=(0, 0, 255))

        # Show the image
        cv2.imshow('Detected and car ROIs', car_copy)
        cv2.waitKey(0)
        iteration_count += 1

    mean = mean_IoU(all_car_list, car_list, detected_indexes)
    print('Mean IoU: ', mean)
