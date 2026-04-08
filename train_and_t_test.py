import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.nn.modules.batchnorm import _BatchNorm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import numpy as np
from scipy import stats
import time
import gc

from tqdm import tqdm

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# optimizer hyperparams
WD = 1e-5
MT = 0.9
# SAM neighbourhood size
RHO = 0.05

# model
def get_efficientnet_model(model_type, classes_num):
    if model_type == 'b0':
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
    elif model_type == 'b7':
        weights = EfficientNet_B7_Weights.DEFAULT
        model = efficientnet_b7(weights=weights)
    else:
        raise ValueError("Invalid model type")
    
    # replace final layer so output -> correct num classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, classes_num)
    return model

# smooth_cross_entropy.py (from https://github.com/davda54/sam/blob/main/example/model/smooth_cross_entropy.py)
def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)
    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)
    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

# sam.py (from https://github.com/davda54/sam/blob/main/sam.py)
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# bypass_bn.py (from https://github.com/davda54/sam/blob/main/example/utility/bypass_bn.py)
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

# adversarial attack helpers
mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1,3,1,1).to(DEVICE)
std  = torch.tensor((0.2023, 0.1994, 0.2010)).view(1,3,1,1).to(DEVICE)

def normalize_img(x):
    return (x - mean) / std

def unnormalize_img(x):
    return x * std + mean

def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=10, norm="linf"):
    model.eval()
    delta = torch.zeros_like(images).uniform_(-eps, eps) if norm == "linf" else torch.zeros_like(images).normal_()
    
    if norm == "l2":
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(-1,1,1,1)
        delta = delta / (n + 1e-10) * torch.rand_like(n) * eps

    delta = torch.clamp(delta, -images, 1 - images).detach()
    
    for _ in range(steps):
        delta.requires_grad = True
        adv = normalize_img(images + delta)
        outputs = model(adv)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = delta.grad.detach()

        if norm == "linf":
            delta = torch.clamp(delta + alpha * grad.sign(), -eps, eps)
        elif norm == "l2":
            g_flat = grad.view(grad.size(0), -1)
            scaled_g = grad / (g_flat.norm(p=2, dim=1).view(-1,1,1,1) + 1e-10)
            delta = delta + alpha * scaled_g
            d_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1).view(-1,1,1,1)
            delta = delta * torch.min(torch.ones_like(d_norm), eps / (d_norm + 1e-10))
            
        delta = torch.clamp(delta, -images, 1 - images).detach()
        
    return torch.clamp(images + delta, 0, 1)

# training
def train_sgd(model, dataloader, epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MT, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    pbar = tqdm(range(epochs), desc="SGD", leave=True)
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        correct_count = 0
        number_samples = 0

        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = smooth_crossentropy(outputs, labels, smoothing=0.1).mean()
            epoch_loss += loss.item()

            # backward pass/ update model weights
            loss.backward()
            optimizer.step()

            # count correct predictions
            predictions = outputs.argmax(1)
            correct_predictions = (predictions == labels).sum().item()
            correct_count += correct_predictions

            # add size of batch
            number_samples += labels.size(0)

        scheduler.step()

        gc.collect()
    
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = correct_count / number_samples
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
    return model

def train_sam(model, dataloader, epochs):
    optimizer = SAM(model.parameters(), torch.optim.SGD, rho=RHO, lr=LR, momentum=MT, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=epochs)
    
    pbar = tqdm(range(epochs), desc="SAM", leave=True)
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        correct_count = 0
        number_samples = 0

        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Pass 1
            enable_running_stats(model)
            outputs = model(images)
            loss = smooth_crossentropy(outputs, labels, smoothing=0.1).mean()
            epoch_loss += loss.item()

            # backward pass/ update model weights
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # Pass 2
            disable_running_stats(model)
            smooth_crossentropy(model(images), labels, smoothing=0.1).mean().backward()
            optimizer.second_step(zero_grad=True)

            # count correct predictions
            predictions = outputs.argmax(1)
            correct_predictions = (predictions == labels).sum().item()
            correct_count += correct_predictions

            # add size of batch
            number_samples += labels.size(0)

        scheduler.step()

        gc.collect()

        avg_loss = epoch_loss / len(dataloader)
        avg_acc = correct_count / number_samples
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
    return model

def evaluate_all(model, dataloader, test_configs):
    model.eval()
    number_samples = 0
    results = {config["name"]: {"correct": 0} for config in test_configs}

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images_pixel = unnormalize_img(images)
        number_samples += labels.size(0)

        for config in test_configs:
            name = config["name"]
            
            if config["norm"] == "none":
                # regular accuracy
                with torch.no_grad():
                    outputs = model(images)
            else:
                # adversarial accuracy
                adv_pixel = pgd_attack(
                    model, images_pixel, labels,
                    eps=config["eps"], alpha=config["alpha"], 
                    steps=config["steps"], norm=config["norm"]
                )
                with torch.no_grad():
                    outputs = model(normalize_img(adv_pixel))
                    
            results[name]["correct"] += (outputs.argmax(1) == labels).sum().item()

    # convert counts to percent
    return {name: data["correct"] / number_samples for name, data in results.items()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EfficientNet B0 or B7 with SAM and SGD")
    parser.add_argument('--model', type=str, choices=['b0', 'b7'], default='b0', help="Choose efficientnet version: 'b0' or 'b7'")
    args = parser.parse_args()

    if args.model == 'b0':
        BATCH_SIZE = 1024
        LR = 0.016
    elif args.model == 'b7':
        BATCH_SIZE = 256
        LR = 0.004
    
    NUM_RUNS = 10
    NUM_EPOCHS = 100
    CLASSES_NUM = 10      # CIFAR-10

    # adversarial tests
    ADV_TESTS = [
        {"name": "Regular", "eps": 0, "alpha": 0, "norm": "none"},
        {"name": "L-inf 1/255", "norm": "linf", "eps": 1/255, "alpha": 0.25/255, "steps": 10},
        {"name": "L-inf 2/255", "norm": "linf", "eps": 2/255, "alpha": 0.5/255, "steps": 10},
        {"name": "L-2 32/255",  "norm": "l2",   "eps": 32/255, "alpha": 8/255, "steps": 10},
        {"name": "L-2 64/255",  "norm": "l2",   "eps": 64/255, "alpha": 16/255, "steps": 10}
    ]

    # set up datasets
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    # # Take the first N samples from each split
    # N_TRAIN = 10000
    # N_TEST  = 2000

    # train_dataset = Subset(full_train_dataset, indices=range(N_TRAIN))
    # test_dataset  = Subset(test_dataset,  indices=range(N_TEST))

    train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Starting experiment with {NUM_RUNS} runs, {NUM_EPOCHS} epochs each.")
    
    final_results = {
        "SGD": {test["name"]: [] for test in ADV_TESTS},
        "SAM": {test["name"]: [] for test in ADV_TESTS}
    }

    for run in range(NUM_RUNS):
        print(f"\n{'='*40}")
        print(f" RUN {run + 1}/{NUM_RUNS}")
        print(f"{'='*40}")

        print("Training SGD...")
        model_sgd = get_efficientnet_model(args.model, CLASSES_NUM).to(DEVICE)
        t0 = time.time()
        model_sgd = train_sgd(model_sgd, train_loader, NUM_EPOCHS)
        print(f"SGD Training took {time.time() - t0:.2f}s. Evaluating...")
        
        sgd_metrics = evaluate_all(model_sgd, test_loader, ADV_TESTS)
        for name, acc in sgd_metrics.items():
            final_results["SGD"][name].append(acc)
            print(f"  SGD [{name}]: {acc:.4f}")

        print("\nTraining SAM...")
        model_sam = get_efficientnet_model(args.model, CLASSES_NUM).to(DEVICE)
        t0 = time.time()
        model_sam = train_sam(model_sam, train_loader, NUM_EPOCHS)
        print(f"SAM Training took {time.time() - t0:.2f}s. Evaluating...")
        
        sam_metrics = evaluate_all(model_sam, test_loader, ADV_TESTS)
        for name, acc in sam_metrics.items():
            final_results["SAM"][name].append(acc)
            print(f"  SAM [{name}]: {acc:.4f}")
        
    # do a t-test
    print(f"\n\n{'='*50}")
    print(f"FINAL STATISTICAL RESULTS (n={NUM_RUNS})")
    print(f"{'='*50}")

    for config in ADV_TESTS:
        name = config["name"]
        sgd_arr = np.array(final_results["SGD"][name])
        sam_arr = np.array(final_results["SAM"][name])
        
        sgd_mean, sgd_std = sgd_arr.mean(), sgd_arr.std()
        sam_mean, sam_std = sam_arr.mean(), sam_arr.std()
        
        # paired t-test since shared dataset
        t_stat, p_val = stats.ttest_rel(sam_arr, sgd_arr)
        
        print(f"\nTest: {name}")
        print(f"  SGD Mean: {sgd_mean:.4f} ± {sgd_std:.4f}")
        print(f"  SAM Mean: {sam_mean:.4f} ± {sam_std:.4f}")
        print(f"  Paired t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.5f}")
        
        if p_val < 0.05:
            better = "SAM" if sam_mean > sgd_mean else "SGD"
            print(f"  -> Statistically significant difference! ({better} performs better)")
        else:
            print("  -> No statistically significant difference at α=0.05.")
