from utils import config
from utils import utils
from models.resnet50 import create_resnet50
from models.resnet101 import create_resnet101
from models.tester import get_predictions, plot_confusion_matrix, plot_most_incorrect
from data.dataloader import create_dataloader
from data.data_transformer import train_transformer, val_transformer, test_transformer

import torch
import torch.nn.functional as F
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Setup CPU/GPU
device = utils.setup_gpu()

# Data Paths
image_directory = os.path.join(os. getcwd(), 'data', 'images', 'res300crop')

train_file = os.path.join(os. getcwd(), 'data', 'data_split_clot_excl', 'train_set.csv')
validation_file = os.path.join(os. getcwd(), 'data', 'data_split_clot_excl', 'val_set.csv')
test_file = os.path.join(os. getcwd(), 'data', 'data_split_clot_excl', 'test_set.csv')

# train_file = os.path.join(os. getcwd(), 'data', 'data_split_day', 'train_set.csv')
# validation_file = os.path.join(os. getcwd(), 'data', 'data_split_day', 'val_set.csv')
# test_file = os.path.join(os. getcwd(), 'data', 'data_split_day', 'test_set.csv')

# Data Loading
target = config.CSV_CLOT_COLUMN
# target = config.CSV_DAY_COLUMN
trainloader = create_dataloader(images_csv=train_file, image_dir=image_directory, target=target,
                                transform=train_transformer(config.ARCHITECTURE),
                                batch_size=config.BATCH_SIZE, shuffle=True)
valloader = create_dataloader(images_csv=validation_file, image_dir=image_directory, target=target,
                              transform=val_transformer(config.ARCHITECTURE),
                              batch_size=config.BATCH_SIZE, shuffle=True)
testloader = create_dataloader(images_csv=test_file, image_dir=image_directory, target=target,
                               transform=test_transformer(config.ARCHITECTURE))
dataloaders = {"train": trainloader, "val": valloader, "test": testloader}


# Load Model
model_name = 'resnet50_fscore_ce_crop-lrs-epoch-75-fscore.pt'
model_path = os.path.join(os. getcwd(), 'reports', 'saved_models', 'clot-models', model_name)

model = None
if config.ARCHITECTURE == "resnet50":
    print("loading resnet50")
    model = create_resnet50().to(device)
elif config.ARCHITECTURE == "resnet101":
    print("loading resnet101")
    model = create_resnet101().to(device)
else:
    raise ValueError(f"No model {config.ARCHITECTURE}.")

model.load_state_dict(torch.load(model_path))
model.eval()

# Testing
classes = ['False', 'True']
# classes = ['Day_1', 'Day_3', 'Day_5', 'Day_7']

images, labels, probs, pred_labels = get_predictions(model, trainloader, device)
plot_confusion_matrix(labels, pred_labels, classes=classes)
# plot_most_incorrect(images, labels, probs, pred_labels, classes=classes)

images, labels, probs, pred_labels = get_predictions(model, valloader, device)
plot_confusion_matrix(labels, pred_labels, classes=classes)
plot_most_incorrect(images, labels, probs, pred_labels, classes=classes)
#
# for seed in [123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627, 282930]:
#     images, labels, probs, pred_labels = get_predictions(model, valloader, device, seed=seed)
#     plot_confusion_matrix(labels, pred_labels, classes=classes)
#
images, labels, probs, pred_labels = get_predictions(model, testloader, device)
plot_confusion_matrix(labels, pred_labels, classes=classes)
# plot_most_incorrect(images, labels, probs, pred_labels, classes=classes)
#
# for seed in [123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627, 282930]:
#     images, labels, probs, pred_labels = get_predictions(model, testloader, device, seed=seed)
#     plot_confusion_matrix(labels, pred_labels, classes=classes)
#
# def batch_predict(images):
#     model.eval()
#     imgs_t = torch.from_numpy(images)
#     imgs_t = imgs_t.permute(0, 3, 1, 2)
#     y_pred, _ = model(imgs_t)
#     y_prob = F.softmax(y_pred, dim=-1)
#
#     return y_prob.detach().numpy()
#
#
# # if __name__ == '__main__':
#
# indx = 0
# utils.show_image(images[indx], classes[labels[indx]])
#
#
# # LIME
# explainer = lime_image.LimeImageExplainer()
# img = images[indx].permute(1, 2, 0).numpy()
# explanation = explainer.explain_instance(img, batch_predict,
#                                          top_labels=3, hide_color=0, num_samples=1000)
#
#
# temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
# temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
# ax1.imshow(mark_boundaries(temp_1, mask_1))
# ax2.imshow(mark_boundaries(temp_2, mask_2))
# ax1.axis('off')
# ax2.axis('off')
# plt.show()
#
#
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=3, hide_rest=False)
# img_boundry1 = mark_boundaries(temp, mask)
# plt.imshow(img_boundry1); plt.show()
#
#
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
# img_boundry2 = mark_boundaries(temp/255.0, mask)
# plt.imshow(img_boundry2); plt.show()


# RISE
from models.tester import RISE
input_size = (224, 224)
gpu_batch = 1
explainer = RISE(model, input_size, gpu_batch)

maskspath = os.path.join(os. getcwd(), 'masks.npy')
generate_new = True

if generate_new or not os.path.isfile(maskspath):
    explainer.generate_masks(N=100, s=8, p1=0.1, savepath=maskspath)
else:
    explainer.load_masks(maskspath)
    print('Masks are loaded.')


def example(img, top_k=3):
    saliency = explainer(img).cpu().numpy()
    n_img = img[None, :, :, :]
    p, c = torch.topk(model(n_img), k=top_k)
    p, c = p[0], c[0]

    plt.figure(figsize=(10, 5 * top_k))
    for k in range(top_k):
        plt.subplot(top_k, 2, 2 * k + 1)
        plt.axis('off')
        # plt.title('{:.2f}% {}'.format(100 * p[k], get_class_name(c[k])))
        plt.title('{:.2f}% {}'.format(100 * p[k], "Test Title"))
        utils.tensor_imshow(img)

        plt.subplot(top_k, 2, 2 * k + 2)
        plt.axis('off')
        # plt.title(get_class_name(c[k]))
        plt.title("Test Title")
        utils.tensor_imshow(img)
        sal = saliency[c[k]]
        plt.imshow(sal, cmap='jet', alpha=0.5)
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.show()

indx = 0
img = images[indx]
example(img, 2)


# # SHAP
# import shap
# e = shap.DeepExplainer(model, images)
# shap_values = e.shap_values(images)
#
# # define a masker that is used to mask out partitions of the input image.
# masker = shap.maskers.Image("inpaint_telea", images[0].shape)
# explainer = shap.Explainer(model, masker, output_names=classes)
# shap_values = explainer(images, max_evals=100, batch_size=6, outputs=shap.Explanation.argsort.flip[:4])
