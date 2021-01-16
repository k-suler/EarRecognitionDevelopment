import torch
from PIL import Image
from torchvision import transforms
import argparse
from pprint import pprint
import os
import matplotlib.pyplot as plt

# model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)
model = torch.load('./models/finnetuned_model_100e2.pt')
model.eval()


# sample execution (requires torchvision)


def main():
    test_files = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(os.path.expanduser("awe_data/val"))
        for f in fn
    ]

    rank = [0 for i in range(0, 100)]

    for filename in test_files:
        _, _, subject, _ = filename.split("/")
        subject = int(subject)

        input_image = Image.open('./' + filename).convert('RGB')
        import pdb

        input_size = 224
        preprocess = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(
            0
        )  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        # Calculate ranks for CMC curve

        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        # pprint(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        predictions = torch.nn.functional.softmax(output[0], dim=0)
        predictions = list(enumerate(output.tolist()[0], 1))
        predictions.sort(key=lambda x: x[1], reverse=True)
        # pdb.set_trace()
        is_found = False
        for t, predicted in enumerate(predictions):
            if predicted[0] == subject:
                is_found = True
                rank[t] = rank[t] + 1
            elif is_found:
                rank[t] = rank[t] + 1

    rank = list(map(lambda x: x / len(test_files), rank))

    plt.plot(list(range(1, 101)), rank)
    plt.ylabel('Rank - t Identification Rate (%)')
    plt.xlabel('Rank (t)')
    plt.title("CMC")
    plt.show()


if __name__ == "__main__":
    main()