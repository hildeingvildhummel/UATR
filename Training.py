"""https://music-classification.github.io/tutorial/part3_supervised/tutorial.html"""

import torch
from Encoder import Encoder, Identity
from loss_functions import VICRegLossFunction, SupConLoss
from efficient_data_loader import get_dataloader
import numpy as np
import wandb
import argparse
from sklearn.metrics import accuracy_score
from lars import LARS

parser = argparse.ArgumentParser(description='Training SimCLR framework for underwater sound.')

parser.add_argument('--train_path', dest='train_path', help='The path to the training samples')
parser.add_argument('--val_path', help='The path to the validation samples', dest='val_path')
parser.add_argument('--baseline_model', help='Boolean to choose between supervised and unsupervised learning',
                    dest='baseline_model', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--samplerate', help='The samplerate of the recordings', dest='samplerate')
parser.add_argument('--learning_rate', help='The learning rate during training', dest='learning_rate')
parser.add_argument('--num_epochs', help='The number of epochs during training', dest='num_epochs')
parser.add_argument('--batch_size', help='The batch size', dest='batch_size')
parser.add_argument('--data_size', help='The size of the datachunks in the batch, in seconds.', dest='data_size')
parser.add_argument('--wandb', help='Boolean to define whether or not to save the logs and models', dest='wandb',
                    default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--name', help='The name how to save the model applicable', dest='name')
parser.add_argument('--augmentation',
                    help='Whether positive samples should be generated instance-based by augmentation', required=False,
                    dest='augmentation', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--save_dir', help='Directory to save the model to', dest='save_path')
parser.add_argument('--num_classes', help='Define the number of classes the classifier needs to discrimininate',
                    required=False,
                    dest='num_classes', default=4)
parser.add_argument('--saved_model', help='Load this saved model for transfer learning.', required=False,
                    dest='saved_model', default=None)

parser.add_argument('--sigma',
                    help='Define the width of the Gaussian distribution for sampling of positive samples during contrastive learning.',
                    required=False,
                    dest='sigma', default=5)

args = parser.parse_args()


class TrainingModule():
    def __init__(self, Baseline):
        self.Baseline = Baseline
        if self.Baseline:
            self.loss_function = SupConLoss()
        else:
            self.loss_function = self._VICRegLoss()
    def _VICRegLoss(self):
        loss_function = VICRegLossFunction(sim_coeff=5, std_coeff=5, cov_coeff=1, num_features=128,
                                           batch_size=int(args.batch_size))
        return loss_function
    def _Unlabeled_data_loader(self, train_path, val_path, sample_rate, batch_size, data_size_sec, augmentation, sigma):
        train_loader = get_dataloader(recording_path=train_path, sample_rate=sample_rate,
                                      recording_len_sec=300, batch_size=batch_size,
                                      sample_len_sec=data_size_sec, labeled_data=False, augmentation=augmentation,
                                      sigma=sigma)
        test_loader = get_dataloader(recording_path=val_path, sample_rate=sample_rate,
                                     recording_len_sec=300, batch_size=batch_size,
                                     sample_len_sec=data_size_sec, labeled_data=False, augmentation=False,
                                     sigma=sigma)
        return train_loader, test_loader

    def _Labeled_data_loader(self, train_path, val_path, sample_rate, batch_size, data_size_sec):
        train_loader = get_dataloader(recording_path=train_path, sample_rate=sample_rate,
                                      recording_len_sec=None, batch_size=batch_size,
                                      sample_len_sec=data_size_sec, augmentation=False, labeled_data=True, AST_preprocess=False)
        test_loader = get_dataloader(recording_path=val_path, sample_rate=sample_rate,
                                     recording_len_sec=None, batch_size=batch_size,
                                     sample_len_sec=data_size_sec, augmentation=False, labeled_data=True, AST_preprocess=False)
        return train_loader, test_loader

    def load_saved_model(self, model_name, learning_rate=None):

        model = Encoder(Baseline=self.Baseline, num_classes=4)
        optimizer = None
        if model_name is not None:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            if learning_rate is not None:
                optimizer = LARS(model.parameters(), lr=learning_rate, momentum=0.9)
                # optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
                optimizer.load_state_dict(checkpoint['optmizer_state_dict'])

        if model_name is not None:
            return model, optimizer, checkpoint['epoch']
        else:
            return model, optimizer, 0

    def training_data(self, model,wav, masked_spec, label, positive_pair, device):

        """Training"""
        model.cuda()
        model.train()

        label = label.to(device)
        positive_pair = positive_pair.to(device)
        masked_spec = masked_spec.to(device)

        # Do the forward pass
        wav_contrast, wav_repr = model(masked_spec)
        pos_contrast, pos_repr = model(positive_pair)

        if not self.Baseline:
            loss, rep_loss, var_loss, cov_loss = self.loss_function(wav_contrast, pos_contrast)
        else:
            loss = self.loss_function(wav_contrast, label)
            loss.retain_grad()

            rep_loss = None
            var_loss = None
            cov_loss = None

        return loss, rep_loss, var_loss, cov_loss

    def testing_data(self, model, test_loader, device):
        # Validation
        model.cuda()
        model.eval()
        repr_loss_list = []
        var_loss_list = []
        cov_loss_list = []

        with torch.no_grad():
            # Define three empty lists
            y_true = []
            y_pred = []
            losses_list = []
            for masked_spec, positive_pair, label, wav in test_loader:
                label = label.to(device)
                positive_pair = positive_pair.to(device)
                masked_spec = masked_spec.to(device)

                wav_contrast, wav_repr = model(masked_spec)
                pos_contrast, pos_repr = model(positive_pair)

                if not self.Baseline:
                    loss, rep_loss, var_loss, cov_loss = self.loss_function(wav_contrast, pos_contrast)
                    repr_loss_list.append(rep_loss.item())
                    var_loss_list.append(var_loss.item())
                    cov_loss_list.append(cov_loss.item())
                else:
                    loss = self.loss_function(wav_contrast, label)

                losses_list.append(loss.item())
                _, pred = torch.max(wav_contrast.data, 1)
                y_true.extend(label.tolist())
                y_pred.extend(pred.tolist())

            test_loss = np.mean(np.array(losses_list))
            if self.Baseline:
                # Calculate the accuracy and mean loss
                accuracy = accuracy_score(y_true, y_pred)
            else:
                accuracy = 0

        return test_loss, accuracy, np.mean(repr_loss_list), np.mean(var_loss_list), np.mean(cov_loss_list)

    def __call__(self, train_path, test_path, sample_rate, data_size_sec, batch_size, learning_rate, num_epochs, name,
                 save_dir, classification, num_classes, augmentation, sigma, wandb_save=False, model=None):
        if wandb_save:
        # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project="SimCLR_ShipsEar",
                name=name,

                # track hyperparameters and run metadata
                config={
                    "learning_rate": learning_rate,
                    "architecture": "SimCLR_ResNet18_ShipsEar",
                    "dataset": "ShipsEar",
                    "epochs": num_epochs,
                }
            )
        if self.Baseline:
            train_loader, test_loader = self._Labeled_data_loader(train_path, test_path, sample_rate, batch_size,
                                                                  data_size_sec)
            model, _, start_epoch = self.load_saved_model(model)
            if augmentation:
               # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
               optimizer = LARS(model.parameters(), lr=learning_rate, momentum=0.9)
            else:
               optimizer = LARS(model.parameters(), lr=learning_rate, momentum=0.9)
               # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = None
        else:
            train_loader, test_loader = self._Unlabeled_data_loader(train_path, test_path, sample_rate, batch_size,
                                                                    data_size_sec, augmentation, sigma=sigma)

            if model is not None:
                model, optimizer, start_epoch = self.load_saved_model(model, learning_rate=learning_rate)
            else:
                model = Encoder(Baseline=self.Baseline, transformed=augmentation)
                optimizer = LARS(model.parameters(), lr=learning_rate, momentum=0.9)
                # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
                start_epoch = 0
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=3
            )

        test_losses = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for epoch in range(start_epoch, num_epochs):
            # Initiate an emtpy list
            losses_list_train = []
            repr_loss_list = []
            var_loss_list = []
            cov_loss_list = []

            for masked_spec, positive_pair, label, wav in train_loader:
                # print(wav.shape)
                loss, repr_loss, var_loss, cov_loss = self.training_data(model, wav, masked_spec, label, positive_pair, device)
                loss.retain_grad()
                if repr_loss is not None:
                    repr_loss_list.append(repr_loss.item())
                    var_loss_list.append(var_loss.item())
                    cov_loss_list.append(cov_loss.item())
                # Do the Backward pass
                optimizer.zero_grad()
                loss.backward(gradient=loss.grad)
                # Update the model
                optimizer.step()
                losses_list_train.append(loss.item())
            print('Epoch: [%d/%d], Train loss: %.4f, composed of repr loss: %.4f, var loss: %.4f, cov loss: %.4f' % (
            epoch + 1, num_epochs, np.mean(losses_list_train), np.mean(repr_loss_list), np.mean(var_loss_list), np.mean(cov_loss_list)))
            test_loss, accuracy, repr_loss_test, var_loss_test, cov_loss_test = self.testing_data(model, test_loader,
                                                                                                  device)

            # Print the validation update
            print('Epoch: [%d/%d], Test loss: %.4f, Test accuracy: %.4f' % (epoch + 1, num_epochs, test_loss, accuracy))
            # print('Epoch: [%d/%d], Valid loss: %.4f' % (epoch + 1, self.num_epochs, test_loss))

            if wandb_save:
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optmizer_state_dict': optimizer.state_dict(),
                            'train_loss': np.mean(losses_list_train), 'test_loss': test_loss},
                           save_dir + '{}_LastEpoch.ckpt'.format(name))
            # Save model
            test_losses.append(test_loss.item())
            if scheduler is not None:
                scheduler.step(test_loss)
            if np.argmin(test_losses) == epoch:
                print('Saving the best model at %d epochs!' % epoch)
                if wandb_save:
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                                'optmizer_state_dict': optimizer.state_dict(),
                                'train_loss': np.mean(losses_list_train), 'test_loss': test_loss},
                               save_dir + '{}_BestEpoch.ckpt'.format(name))
                    # wandb.save("{}_{}epoch.h5".format(name, str(epoch + 1)))
            if repr_loss is None:
                repr_loss = np.zeros(1)
                var_loss = np.zeros(1)
                cov_loss = np.zeros(1)
                repr_loss_test = np.zeros(1)
                var_loss_test = np.zeros(1)
                cov_loss_test = np.zeros(1)
            if wandb_save:
                wandb.log({
                    "Epoch": epoch,
                    "Train Loss": np.mean(losses_list_train),
                    "MSE Loss": np.mean(repr_loss_list),
                    "Variance Loss": np.mean(var_loss_list),
                    "Covariance Loss": np.mean(cov_loss_list),
                    "Valid Loss": test_loss,
                    "Valid Acc": accuracy,
                    "MSE Loss Val": np.mean(repr_loss_test),
                    "Variance Loss Val": np.mean(var_loss_test),
                    "Covariance Loss Val": np.mean(cov_loss_test)
                })


training_module = TrainingModule(Baseline=args.baseline_model)

training_module(train_path=args.train_path, test_path=args.val_path, sample_rate=int(args.samplerate),
                data_size_sec=int(args.data_size), batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                num_epochs=int(args.num_epochs), augmentation=args.augmentation, name=args.name,
                save_dir=args.save_path, wandb_save=args.wandb, model=args.saved_model,
                num_classes=int(args.num_classes), sigma=float(args.sigma))