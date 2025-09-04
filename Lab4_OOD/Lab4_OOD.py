import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import torchvision
import os
from torchvision.datasets import FakeData
from torchvision import transforms
import numpy as np
import random
from model import ResCNN, Autoencoder
from utility import test_metric_confusion_matrix, plot_logit_and_softmax_true, plot_logit_and_softmax_fake, compute_scores, max_logit, plot_distribution, compute_scores_autoencoder, plot_metrics
from FGSM import fgsm_attack_save_plots, attack_success_rate, train_with_fgsm_adversarial, success_rate_each_class

if __name__ == "__main__":
    
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 128

    # Datasets

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers= True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers= True)
    fakeset = FakeData(size=1000, image_size=(3, 32, 32), transform=transform)
    fakeloader = torch.utils.data.DataLoader(fakeset, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers= True)

    testset_classes = testset.classes
    class_dict = {class_name:id_class for id_class, class_name in enumerate(trainset.classes)} 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Exercise 1.1: Build a simple OOD detection pipeline

    load = True 

    if load:
        model = ResCNN().to(device)
        model.load_state_dict(torch.load('Save/Cifar10_ResCNN.pth')) #fully trained
    
    # CNN Data logits and softmat
    
    test_metric_confusion_matrix(model, testloader, device, testset.classes)
    plot_logit_and_softmax_true(model, testloader, testset.classes)
    plot_logit_and_softmax_fake(model, fakeloader, testset.classes)
    
    # CNN Distribution
    
    scores_test = compute_scores(model, device, testloader, max_logit)
    scores_fake = compute_scores(model, device, fakeloader, max_logit)
    
    plot_distribution(scores_test, scores_fake, "CNN")
    
    # Autoencoder Distribution
    
    if load:
        ae_model = Autoencoder().to(device)
        ae_model.load_state_dict(torch.load('Save/Cifar10_Autoencoder.pth')) #fully trained
    
    ae_scores_test = compute_scores_autoencoder(ae_model, device, testloader)
    ae_scores_fake = compute_scores_autoencoder(ae_model, device, fakeloader)
    
    plot_distribution(ae_scores_test, ae_scores_fake, "Autoencoder")
    
    #Exercise 1.2: Measure your OOD detection performance
    
    plot_metrics(scores_test, scores_fake, "CNN")
    plot_metrics(ae_scores_test, ae_scores_fake, "Autoencoder")
    
    #Exercise 2.1: Implement FGSM and generate adversarial examples
    
    fgsm_attack_save_plots(model, testloader, device, testset_classes, targeted_attack = False, eps=2/255,
                            output_folder="output/untargeted_baseline", sample_id= 104)
    
    #Exercise 2.2: Augment training with adversarial examples
    
    # untargeted attack success rate Baseline
    
    print("Untargeted attack success rate Baseline Cifar10_ResCNN:")
    uasr = attack_success_rate(model, testloader, device, testset_classes, eps=2/255, targeted=False)
    
    # Training with 
    
    train_with_fgsm_adversarial(model, trainloader, device, num_epochs=10, eps=2/255,
                                targeted=False, target_label=None, lr=0.001,
                                project_name="Lab4_OOD_FGSM", run_name="fgsm_adv_train_untargeted")
    
    #Note now model is trained on adversarial sample
    
    fgsm_attack_save_plots(model, testloader, device, testset_classes, targeted_attack = False, eps=2/255,
                            output_folder="output/untargeted_trained_on_adv_sample", sample_id= 104)
    
    # untargeted adversarial success rate after training
    print("Untargeted attack success rate Cifar10_ResCNN trained on untargeted adv sample:")
    uasr = attack_success_rate(model, testloader, device, testset_classes, eps=2/255, targeted=False)
    
    #Exercise 3.3: Experiment with *targeted* adversarial attacks
    
    # Reset model to test different training
    
    if load:
        model = ResCNN().to(device)
        model.load_state_dict(torch.load('Save/Cifar10_ResCNN.pth')) #fully trained
    
    # Sigle image targeted attack for cat class (same class as future training)
    
    fgsm_attack_save_plots(model, testloader, device, testset_classes, targeted_attack = True, target_label=3, eps=2/255,
                                output_folder="output/targeted_baseline_same_class", sample_id= 104)
    
    # Sigle image targeted attack for horse class (different class as future training)
    
    fgsm_attack_save_plots(model, testloader, device, testset_classes, targeted_attack = True, target_label=7, eps=2/255,
                                output_folder="output/targeted_baseline_different_class", sample_id= 104)
    
    # targeted adversarial success rate Baseline
    
    print("Baseline targeted (Cat) attack success rate Cifar10_ResCNN:")
    tasr = attack_success_rate(model, testloader, device, testset_classes, eps=2/255, targeted=True, target_label=3)
    
    # Targeted Attack Success Rate before training for each class

    print("Baseline Targeted (Cat) Attack Success Rate for each class")
    tasr_class = success_rate_each_class(model, testloader, device, testset_classes, class_dict, eps=2/255)
    
    train_with_fgsm_adversarial(model, trainloader, device, num_epochs=10, eps=2/255,
                                    targeted=True, target_label=3, lr=0.001,
                                    project_name="Lab4_OOD_FGSM", run_name="fgsm_adv_train_targeted_cat")
    
    # Note now model is trained on targeted adversarial sample
    
    fgsm_attack_save_plots(model, testloader, device, testset_classes, targeted_attack = False, eps=2/255,
                            output_folder="output/targeted_trained_baseline_same_class", sample_id= 104)
    
    fgsm_attack_save_plots(model, testloader, device, testset_classes, targeted_attack = False, eps=2/255,
                            output_folder="output/targeted_trained_baseline_different_class", sample_id= 104)
    
    # targeted adversarial success rate after training for targeted class
    
    print("Targeted Attack (Cat) Success Rate after training Cifar10_ResCNN:")
    tasr = attack_success_rate(model, testloader, device, testset_classes, eps=2/255, targeted=True, target_label=3)
    
    # Targeted Attack Success Rate before training for each class

    print("Targeted Attack (Cat) Success Rate after training for each class")
    tasr_class = success_rate_each_class(model, testloader, device, testset_classes, class_dict, eps=2/255)
    
    # untargeted adversarial success rate after training
    print("Untargeted attack success rate Cifar10_ResCNN trained on targeted (cat) adv sample:")
    uasr = attack_success_rate(model, testloader, device, testset_classes, eps=2/255, targeted=False)
    
    # Labels Legend
    # 0 airplane
    # 1 automobile
    # 2 bird
    # 3 cat
    # 4 deer
    # 5 dog
    # 6 frog
    # 7 horse
    # 8 ship
    # 9 truck