import os, sys
sys.path.append('C:/Users/flori/OneDrive/Máy tính/Tai-lieu/HCMUS/Image processing')
from lib import *

def evaluate(model, criterion, dataloader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (image, labels) in enumerate(dataloader):
            image, labels = image.cuda(), labels.cuda()

            with autocast():
                pred = model(image)

            loss = criterion(pred.float(), labels)
            total_loss += loss.item()

            _, predicted = torch.max(pred, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(pred, dim=1).cpu().numpy())  

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f'Score - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    return avg_loss, accuracy, precision, recall, f1, conf_matrix



def main():
    model = CNN().cuda()
    criterion = CrossEntropyLoss()

    model.load_state_dict(torch.load('C:/Users/flori/OneDrive/Máy tính/Tai-lieu/HCMUS/Image processing/weight/best_weight_final.pth'))

    # Test dataset and dataloader
    test_dataset = Covid_XRay(split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    evaluate(model, criterion, test_dataloader)

if __name__  == '__main__':
    main()
