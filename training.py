import os, sys
sys.path.append('C:/Users/flori/OneDrive/Máy tính/Tai-lieu/HCMUS/Image processing')
from lib import *

def validate(model, criterion, val_dataloader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (image, labels) in enumerate(val_dataloader):
            image, labels = image.cuda(), labels.cuda()

            with autocast():
                pred = model(image)

            loss = criterion(pred.float(), labels)
            val_loss += loss.item()

            _, predicted = torch.max(pred, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = val_loss / len(val_dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    model = CNN().cuda()
    criterion = CrossEntropyLoss()

    lr = 0.001
    optimizer = opt.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    num_epochs = 50
    batch_size = 64

    train_dataset = Covid_XRay(split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = Covid_XRay(split='test')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Directory to save model weights
    save_dir = 'weight'
    os.makedirs(save_dir, exist_ok=True)

    # Load pretrained weights if available
    pretrained_weight_path = 'C:/Users/flori/OneDrive/Máy tính/Tai-lieu/HCMUS/Image processing/weight/best_weight_1.pth'
    if os.path.exists(pretrained_weight_path):
        model.load_state_dict(torch.load(pretrained_weight_path))
        print(f"Loaded pretrained weights from {pretrained_weight_path}")

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for i, (image, labels) in enumerate(train_dataloader):
            image, labels = image.cuda(), labels.cuda()
            optimizer.zero_grad()

            with autocast():
                pred = model(image)
                loss = criterion(pred, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_dataloader)
        accuracy = correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}')

        avg_val_loss, val_accuracy = validate(model, criterion, val_dataloader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    save_path = os.path.join(save_dir, f'model_after_{num_epochs}_epochs.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model weights saved at {save_path} after {num_epochs} epochs')

if __name__ == '__main__':
    main()
