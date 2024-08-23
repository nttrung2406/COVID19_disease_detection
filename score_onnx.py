import os, sys
sys.path.append('C:/Users/flori/OneDrive/Máy tính/Tai-lieu/HCMUS/Image processing')
from lib import *

def inference_onnx(ort_session, images):
    ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]

test_dataset = Covid_XRay(split='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
onnx_model_path = "C:/Users/flori/OneDrive/Máy tính/Tai-lieu/HCMUS/Image processing/model.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

all_preds = []
all_labels = []

for images, labels in test_dataloader:
    images = images.cpu()
    preds = inference_onnx(ort_session, images)
    predicted_labels = np.argmax(preds, axis=1)
    all_preds.extend(predicted_labels)
    all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_preds)


print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
print(f'Confusion Matrix:\n{conf_matrix}')
