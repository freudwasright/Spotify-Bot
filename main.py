import torch
from datapipeline import datareader
from mlpipeline import models
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
import datetime
import matplotlib.pyplot as plt
from logs.loghandler import TrainingLogger
from pathlib import Path
import logging

# hyper parameters
batch_size = 128
num_epochs = 10

# init logger
root_dir = Path(__file__).parent.absolute().__str__()
log_file_path = root_dir + '/logs/traininglogs/training.log'
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
training_logger = TrainingLogger.construct_logger(name='GTZan_model', log_file_path=log_file_path, logger_level=20,
                                                  formatter=formatter)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.memory_cached() / 1024 ** 2)
    torch.cuda.empty_cache()
    preprocessing = transforms.Compose([
        transforms.ToTensor(),  # converts (H W C) uint8 to (C W H) float32 [0-1]
        transforms.Normalize((0.5124, 0.4420, 0.4994), (0.4354, 0.4721, 0.4593)),
        transforms.Resize((128, 128), interpolation=Image.NEAREST)
    ])
    dataset = datareader.MusicDataset(csv_path='d:/Data/features_30_sec.csv', root_dir='d:/Data/images_original',
                                      transform=preprocessing)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    base_model = models.SimpleGTZANet(3, 32)
    base_model = base_model.to(device=device)

    cost_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-2)

    min_cost = 10000.0
    counter = 0
    cost_values = []
    accuracy_values = []
    epoch = 0
    print(f"Starting training, logging training info to {log_file_path}...")
    while True:

        total_cost = 0.0
        accuracy = 0

        for data, target in train_loader:
            y_pred = base_model(data.to(device=device))
            target = target.to(device=device)
            cost = cost_fn(y_pred, target)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()

            predicted_categories = torch.argmax(y_pred, dim=1)
            accuracy_tensor = torch.sum(target == predicted_categories)
            accuracy += accuracy_tensor.item()
            total_cost += cost.item() / len(train_loader)

        cost_values.append(total_cost / len(dataset))
        accuracy_values.append(accuracy / len(dataset))

        if total_cost > min_cost:
            counter += 1
            training_logger.log_info(f"Cost did not decrease at epoch: {epoch}, total epoch stagnation {counter}")

        elif total_cost < min_cost:
            min_cost = total_cost
            counter = 0
            training_logger.log_info(f'The minimum cost currently is: {min_cost}.')

        training_logger.log_info('{} Epoch {}, Training loss {}, Training Accuracy {}'.format(datetime.datetime.now(),
                                                                                              epoch, total_cost,
                                                                                              accuracy / len(dataset)))

        print('{} Epoch {}, Training loss {}, Training Accuracy {}'.format(datetime.datetime.now(),
                                                                           epoch, total_cost,
                                                                           accuracy / len(dataset)))

        epoch += 1
        if counter == 10:
            print("Stopping training due to over fitting...")
            break
    g = plt.figure(1)
    plt.title("Cost over epochs")
    plt.plot(cost_values, 'r')
    g.show()
    t = plt.figure(2)
    plt.title("Accuracy")
    plt.plot(accuracy_values, 'b')
    t.show()
