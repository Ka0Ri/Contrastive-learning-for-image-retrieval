from clr import SimCLR
import yaml
from retrieval import ImageRetrieval
from linear_classifier import LinearClassifier
# from test_resolution import Resolution_retrieval
from utils.dataset_wrapper import DataSetWrapper, Multiresolution_DataSetWrapper


from torchvision.utils import save_image, make_grid
import os

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config)

    # train_loader, valid_loader = dataset.get_train_validation_data_loaders()
    # samples = next(iter(train_loader))
    # print(samples)
    

    retrieve = ImageRetrieval(dataset, config)
    retrieve.extract_feature()
    retrieve.TopK_score_on_validation()
    retrieve.random_retrieve()

    # simclr = SimCLR(dataset, config)
    # simclr.train()

    # classifier = LinearClassifier(dataset, config)
    # classifier.train()

if __name__ == "__main__":
    main()
