import torch
from datapipeline import datareader
from mlpipeline import models
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# hyper parameters
batch_size = 128
num_epochs = 10


if __name__ == '__main__':
    device = torch.device('cpu')
    print(torch.cuda.memory_allocated()/1024**2, torch.cuda.memory_cached()/1024**2)
    torch.cuda.empty_cache()
    preprocessing = transforms.Compose([
        transforms.ToTensor(),  # converts (H W C) uint8 to (C W H) float32 [0-1]
        transforms.Normalize((0.5124, 0.4420, 0.4994), (0.4354, 0.4721, 0.4593))
    ])
    dataset = datareader.MusicDataset(csv_path='d:/Data/features_30_sec.csv', root_dir='d:/Data/images_original',
                                      transform=preprocessing)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    base_model = models.ResNet(input_channel=3, output_channel=32, out_dim=10)
    base_model = base_model.to(device=device)

    cost_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-2)

    for epoch in range(num_epochs):
        counter = 0
        for data, target in train_loader:
            y_pred = base_model(data)
            cost = cost_fn(y_pred, target)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()

            counter += 1
            if counter % 100 == 0:
                print(f'At iteration: {counter} the loss is: {cost}')

