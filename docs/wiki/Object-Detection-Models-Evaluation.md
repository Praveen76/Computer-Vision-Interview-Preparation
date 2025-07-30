# Q1: How do you evaluate test data for object detection models?
Ans: 

```python
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    # Calculate area of intersection
    intersection_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)

    # Calculate area of union
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def calculate_metrics(y_true, y_pred):
    true_boxes = [box for boxes in y_true for box in boxes]
    pred_boxes = [box for boxes in y_pred for box in boxes]

    true_labels = [1] * len(true_boxes)
    pred_labels = [1] * len(pred_boxes)

    # Calculate precision, recall, and F1 score
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    # Calculate mean IoU
    iou_values = [calculate_iou(true_box, pred_box) for true_box in true_boxes for pred_box in pred_boxes]
    mean_iou = sum(iou_values) / len(iou_values)

    return precision, recall, f1, mean_iou

# Example usage
ground_truth_boxes = [[(x1, y1, x2, y2), (x1, y1, x2, y2)], [(x1, y1, x2, y2)]]
predicted_boxes = [[(x1, y1, x2, y2), (x1, y1, x2, y2)], [(x1, y1, x2, y2), (x1, y1, x2, y2)]]

precision, recall, f1, mean_iou = calculate_metrics(ground_truth_boxes, predicted_boxes)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Mean IoU: {mean_iou:.2f}')

```

# Q2:  How would you evaluate test data for an Information Retrieval System, which has got 2 parts: 1) Object detection model to detect the desired information in the documents 2) Retrieve identified information from the documents. Let's consider that we're interested in 4 pieces of information, hence, 4 classes: Name, Bank Account Number, Bank Account Holder Name, and Total Deposited Amount.
Ans: To evaluate an Information Retrieval System with two parts (object detection and information retrieval) for the specified classes, you can use a combination of metrics such as precision, recall, F1 score, and accuracy. Below is a sample Python code to get you started. Note that you would need the ground truth and predicted data for both the object detection and information retrieval steps.

```python
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_object_detection_metrics(true_boxes, predicted_boxes):
    # Assuming each box is associated with a class label
    true_labels = [label for labels in true_boxes for label in labels]
    pred_labels = [label for labels in predicted_boxes for label in labels]

    # Calculate precision, recall, and F1 score for object detection
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    return precision, recall, f1

def calculate_information_retrieval_metrics(true_info, pred_info):
    # Convert information labels to binary (present/absent)
    true_labels = [1 if info is not None else 0 for info in true_info]
    pred_labels = [1 if info is not None else 0 for info in pred_info]

    # Calculate precision, recall, and F1 score for information retrieval
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)

    return precision, recall, f1, accuracy

# Example usage
# Replace the following with your actual ground truth and predicted data
true_object_detection_boxes = [[(x1, y1, x2, y2), (x1, y1, x2, y2)], [(x1, y1, x2, y2)]]
predicted_object_detection_boxes = [[(x1, y1, x2, y2), (x1, y1, x2, y2)], [(x1, y1, x2, y2), (x1, y1, x2, y2)]]

true_information = ['John Doe', '1234567890', 'John Doe', '$5000']
predicted_information = ['Jane Doe', '0987654321', 'Jane Doe', '$4000']

# Object Detection Metrics
object_detection_precision, object_detection_recall, object_detection_f1 = \
    calculate_object_detection_metrics(true_object_detection_boxes, predicted_object_detection_boxes)

# Information Retrieval Metrics
info_retrieval_precision, info_retrieval_recall, info_retrieval_f1, info_retrieval_accuracy = \
    calculate_information_retrieval_metrics(true_information, predicted_information)

print(f'Object Detection Precision: {object_detection_precision:.2f}')
print(f'Object Detection Recall: {object_detection_recall:.2f}')
print(f'Object Detection F1 Score: {object_detection_f1:.2f}')

print(f'Information Retrieval Precision: {info_retrieval_precision:.2f}')
print(f'Information Retrieval Recall: {info_retrieval_recall:.2f}')
print(f'Information Retrieval F1 Score: {info_retrieval_f1:.2f}')
print(f'Information Retrieval Accuracy: {info_retrieval_accuracy:.2f}')
```

In this example, the `calculate_object_detection_metrics` function computes precision, recall, and F1 score for the object detection part, considering the bounding boxes and their associated class labels. The `calculate_information_retrieval_metrics` function computes precision, recall, F1 score, and accuracy for the information retrieval part, considering the identified information labels.

Replace the placeholder data with your actual ground truth and predicted data for both object detection and information retrieval steps. Adjust the code based on the specific format of your data.