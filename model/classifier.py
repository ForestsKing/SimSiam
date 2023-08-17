from torch import nn


class Classifier(nn.Module):
    def __init__(self, args, encoder):
        super(Classifier, self).__init__()
        self.encoder = args.encoder
        self.freeze = args.freeze

        if self.encoder:
            self.encoder = encoder
            self.fc = nn.Linear(64, 10)

            if self.freeze:
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False

        else:
            self.fc = nn.Linear(784, 10)

    def forward(self, x):
        if self.encoder:
            x = self.encoder(x)
        else:
            x = x.view(x.shape[0], -1)

        x = self.fc(x)
        return x
