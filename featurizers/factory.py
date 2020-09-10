from featurizers.resnet import ResnetEncoder

def get_encoder(name):
    if name == "resnet18":
        return ResnetEncoder(num_layers=18, pretrained=False)
