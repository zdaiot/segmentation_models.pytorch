import segmentation_models_pytorch as smp
import torch.nn as nn
import torch

model = smp.Unet('resnet34', encoder_weights=None, activation=None, use_ConvTranspose2d=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
print('Network initialized. Running a test batch.')
for _ in range(1):
    with torch.set_grad_enabled(True):
        batch = torch.empty(1, 3, 768, 768).normal_()
        targets = torch.empty(1, 1, 768, 768).normal_()

        out = model(batch)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
print(model)
print('out.shape:', out.shape)