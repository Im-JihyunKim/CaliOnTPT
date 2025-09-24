import argparse

import time, os, datetime, csv

from copy import deepcopy

from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip_lessctx import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import torch.nn.functional as F

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def plot_reliability_diagram(result_dict:dict, dataset:str, hash_:str=None, n_bins:int=15):
    predicted_label = np.array(result_dict['prediction'])
    targets = np.array(result_dict['label'])
    confidences = np.array(result_dict['max_confidence'])
    accuracies = (predicted_label == targets)

    # Binning
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    acc_values = []
    conf_values = []
    bin_centers = []
    bin_counts = []

    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        n_in_bin = np.sum(in_bin)
        
        if n_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            
            acc_values.append(accuracy_in_bin)
            conf_values.append(avg_confidence_in_bin)
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_counts.append(n_in_bin)
            
            # ECE
            prop_in_bin = n_in_bin / len(confidences)
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    fig = plt.figure(figsize=(10,8))
    bar_width = bin_lowers[1] - bin_lowers[0]
    x_pos = np.array(bin_centers)
    
    # Confidence
    plt.bar(x_pos, conf_values, width=bar_width, 
            label='Gap', alpha=0.37, color='indianred', 
            edgecolor='black', linewidth=1)
    
    # Accuracy
    plt.bar(x_pos, acc_values, width=bar_width, 
            label='Accuracy', alpha=0.7, color='mediumblue', 
            edgecolor='black', linewidth=1)
    
    # perfect calibration
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, 
             label='Perfect calibration')
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Diagram (ECE = {ece*100:.2f})')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.xticks(np.linspace(0, 1, 11))
    save_plot_dir = './results/reliability_diagram'
    os.makedirs(save_plot_dir, exist_ok=True)
    fig_title = f'./results/reliability_diagram/CaliOnTPT_{dataset}_reliability_diagram.png' if hash_ is None else f'./results/reliability_diagram/CaliOnTPT_{dataset}_reliability_diagram_{hash_}.png'
    plt.savefig(fig_title, dpi=300)

def ECE_Loss(num_bins, predictions, confidences, correct):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_accuracy = [0]*num_bins
    bin_confidence = [0]*num_bins
    bin_num_sample = [0]*num_bins

    for idx in range(len(predictions)):
        confidence = confidences[idx]
        bin_idx = -1
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            bin_idx += 1 
            bin_lower = bin_lower.item()
            bin_upper = bin_upper.item()
            if bin_lower < confidence and confidence <= bin_upper:
                bin_num_sample[bin_idx] += 1
                bin_accuracy[bin_idx] += correct[idx]
                bin_confidence[bin_idx] += confidences[idx]
    
    for idx in range(num_bins):
        if bin_num_sample[idx] != 0:
            bin_accuracy[idx] = bin_accuracy[idx]/bin_num_sample[idx]
            bin_confidence[idx] = bin_confidence[idx]/bin_num_sample[idx]

    ece_loss = 0.0
    oe_loss = 0.0
    for idx in range(num_bins):
        temp_abs = abs(bin_accuracy[idx]-bin_confidence[idx])
        ece_loss += (temp_abs*bin_num_sample[idx])/len(predictions)

        # max(conf - acc, 0)
        overconfidence = max(bin_confidence[idx] - bin_accuracy[idx], 0)
        oe_loss += (overconfidence * bin_num_sample[idx]) / len(predictions)

    return ece_loss, oe_loss, bin_accuracy, bin_confidence, bin_num_sample

def Calculator(result_dict): 
    list_max_confidence = result_dict['max_confidence']
    list_prediction = result_dict['prediction']
    list_label = result_dict['label']

    torch_list_prediction = torch.tensor(list_prediction).int()
    torch_list_label = torch.tensor(list_label).int()

    torch_correct = (torch_list_prediction == torch_list_label)
    list_correct = torch_correct.tolist()

    # Identify incorrect predictions using tensor operations
    incorrect_indices = (torch_list_prediction != torch_list_label)
    torch_max_confidence = torch.tensor(list_max_confidence)

    # Extract confidences for incorrect predictions
    incorrect_confidences = torch_max_confidence[incorrect_indices].tolist()

    ece_data = ECE_Loss(20, list_prediction, list_max_confidence, list_correct)
    acc = sum(list_correct)/len(list_correct)

    print('acc: ', acc*100)
    print('ece: ', ece_data[0]*100)
          
    return acc*100, ece_data[0]*100, ece_data[1]*100, incorrect_confidences

def select_confident_samples(logits, top):
    logits = logits

    batch_entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    if batch_entropy.dim() > 1:
        batch_entropy = batch_entropy.mean(-1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(logits.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    outputs = outputs

    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) 
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) 
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def entropy(outputs):
    outputs = outputs
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    return -(logits * logits.exp()).sum(dim=-1)

def test_time_tuning(model, inputs, optimizer, scaler, args, log_file):
    output = None
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    loss = torch.tensor(0.0).cuda()

    for j in range(args.tta_steps):
        with torch.amp.autocast('cuda'):
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output = model(inputs) 

            raw_pred = output[0].detach()
            p_pred = output.view(output.size()[0], args.num_prompts, -1).detach()

            p_ent = entropy(p_pred).mean(0) 
            plpd = p_pred[0].max(-1)[0].unsqueeze(0) - p_pred[1:].max(dim=-1)[0] 
            plpd = plpd.mean(0)

            log_string(log_file, str(model.prompt_learner.ctx_order))
            
            ent_order = p_ent.topk(args.num_prompts)[1]
            init_p_position = model.prompt_learner.ctx_order[0]
            exactnumber = torch.where(ent_order==init_p_position)[0].item()
            plpd_order = plpd.topk(args.num_prompts)[1]
            exactnumberlist1 = ent_order[min(exactnumber + 1, args.num_prompts - 1):]
            exactnumberlist2 = plpd_order[:exactnumber]
            alllist = np.intersect1d(exactnumberlist1.cpu().numpy(), exactnumberlist2.cpu().numpy())
            exactnumber = torch.cat([ent_order[alllist], ent_order[[exactnumber]]], dim=0).cpu().numpy() if set(alllist) else ent_order[exactnumber].cpu().numpy()

            output = output.view(args.batch_size, args.num_prompts, -1)[:, exactnumber]
            exactnumber_list = np.array(exactnumber).reshape(-1).tolist()
            for i in exactnumber_list:
                model.prompt_learner.ctx_order.remove(i)
                model.prompt_learner.ctx_use[i] += 1
                model.prompt_learner.ctx_order.append(i)
            raw_pred = raw_pred.view(args.num_prompts, -1)[exactnumber].mean(0)

            raw_entropy = avg_entropy(raw_pred.unsqueeze(0))
            output, selected_idx = select_confident_samples(output, args.selection_p)

            loss = avg_entropy(output).mean()

            output_probs = F.softmax(output, dim=-1)
            target_probs = output_probs.mean(0, keepdim=True)

            m = 0.5 * (output_probs + target_probs)
            log_probs, log_target, log_m = torch.log(output_probs + 1e-8), torch.log(target_probs + 1e-8), torch.log(m + 1e-8)
            kl_pm = (output_probs * (log_probs - log_m)).sum(dim=-1).mean()
            kl_qm = (target_probs * (log_target - log_m)).sum(dim=-1).mean()
            consistency_loss = 0.5 * (kl_pm + kl_qm)
            loss += args.lambda_term * consistency_loss

        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
        loss = torch.tensor(0.0).cuda()

    if args.cocoop:
        return pgen_ctx

    return exactnumber, raw_pred, raw_entropy

def log_string(log_file, string):
    log_file.write(string + '\n')
    log_file.flush()


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)
    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
        assert args.load is not None
        load_model_weight(args.load, model, 'cpu', args) # to load to cuda: device="cuda:{}".format(args.gpu)
        model_state = deepcopy(model.state_dict())
    else:
        model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init, False, args.num_prompts)
        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner[0].ctx.copy_(pretrained_ctx)
                model.prompt_learner[0].ctx_init_state = pretrained_ctx
        model_state = None

    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define optimizer
    if args.cocoop:
        optimizer = None
        optim_state = None
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.amp.GradScaler('cuda', init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}

    accuracy_data, ece_data, oe_data, confidence_data = {}, {}, {}, {}

    for set_id in datasets:
        accuracy_data[set_id] = []  # Initialize list for each set_id
        ece_data[set_id] = []       # Initialize list for each set_id 
        oe_data[set_id] = []
        confidence_data[set_id] =[]

        if args.tpt:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1,
                                            augmix=len(set_id)>1)
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        if args.arch == 'ViT-B/16':
            model_arch = 'ViT-B-16'
        elif args.arch == 'RN50':
            model_arch = 'RN50'
        elif args.arch == 'ViT-B/32':
            model_arch = 'ViT-B-32'
        prefix = 'logs/caliontpt/'
        if args.onlinetpt: prefix += 'CaliOnTPT_'

        logname = prefix + "{}_lam{}_nump{}_seed{}_{}_{}.txt".format(args.log_date, args.lambda_term, args.num_prompts, args.seed, model_arch, set_id)
        
        ## ?
        os.makedirs(os.path.dirname(logname), exist_ok=True)
        log_file = open(logname, 'w')
        print("evaluating: {}".format(set_id))
        log_string(log_file, "evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model

        if len(set_id) > 1:
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all
        if args.cocoop:           
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.cuda(args.gpu)
        else:
            model.reset_classnames(classnames, args.arch)

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        log_string(log_file, "number of test samples: {}".format(len(val_dataset)))
        print(logname)
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize,
                    shuffle=True,
                    num_workers=args.workers, pin_memory=True)
            
        results[set_id], result_dict, performance_dict = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, log_file, args)
        
        ## save reliability_diagram plot
        plot_reliability_diagram(result_dict, set_id, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        pd.DataFrame(performance_dict).to_csv(f"./results/CaliOnTPT_{set_id}_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.csv")
        
        acc, ece, oe, incorrect_confidences = Calculator(result_dict)
        accuracy_data[set_id].append(acc)
        ece_data[set_id].append(ece)
        oe_data[set_id].append(oe)
        
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
            log_string(log_file, "=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))
            log_string(log_file, "=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")

    log_string(log_file, "======== Result Summary ========")
    log_string(log_file, "params: nstep	lr	bs")
    log_string(log_file, "params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    log_string(log_file, "\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        log_string(log_file, "{}".format(id))
    log_string(log_file, "\n")
    for id in results.keys():
        log_string(log_file, "{:.2f}".format(results[id][0]))
    log_string(log_file, "\n")

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    custom_path = f"./results/dynap_mixup_{current_datetime}"
    file_exists = os.path.isfile(custom_path)

    with open(custom_path, 'a' if file_exists else 'w', newline='') as csvfile:    
        csvwriter = csv.writer(csvfile)

        if not file_exists:
            #csvwriter.writerow(["======== Result Summary ========"])
            csvwriter.writerow(["params: nstep", "lr", "bs"])
            csvwriter.writerow([current_datetime,"params: {} {} {}".format(args.tta_steps, args.lr, args.batch_size)])
            csvwriter.writerow(["", "[set_id]", "Top-1 acc.", "Top-5 acc."])

               
        # Write the dataset ids in the first row
        dataset_ids = list(results.keys())
        csvwriter.writerow(current_datetime)

        # code without text disperssion test
        csvwriter.writerow([""] + dataset_ids)
        
        # Write the Top-1 accuracies
        top1_accs = ["Top-1 acc."] + ["{:.2f}".format(results[id][0]) for id in dataset_ids]
        csvwriter.writerow(top1_accs)

        # Write the ECE
        ece_ = ["ECE."] + ["{:.2f}".format(results[id][1]) for id in dataset_ids]
        csvwriter.writerow(ece_)

        # Write final accuracies
        final_acc = ["Accuracy"] + ["{:.2f}".format(accuracy_data[id][0]) for id in dataset_ids]
        csvwriter.writerow(final_acc)

        # Write the ECE
        ECE = ["ECE."] + ["{:.2f}".format(ece_data[id][0]) for id in dataset_ids]
        csvwriter.writerow(ECE)   

def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, log_file, args):
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    ece = AverageMeter('ECE', ':6.2f', Summary.AVERAGE)
    oe = AverageMeter('OE', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [top1, ece, oe],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()
    model.l2_norm_cal = False

    result_dict = {'max_confidence': [], 'prediction': [], 'label': []}
    performance_dict = {'iter':[],
                        'acc@1':[],
                        'ece': [],
                        'oe': []}

    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]  # 원본
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)
        images = torch.cat(images, dim=0)

        # reset the tunable prompt to its initial state
        if not args.cocoop:
            if args.tta_steps > 0 and not args.onlinetpt:
                with torch.no_grad():
                    # print('___________Prompts are reset___________')
                    model.reset()
            elif args.onlinetpt and args.num_prompts > 1:
                with torch.no_grad():
                    if model.prompt_learner.ctx_use.count(0) == 0 and 'noapp' not in args.log_date:
                        time0 = time.time()
                        log_string(log_file, "All prompts are used, reset prompt {}".format(model.prompt_learner.ctx_order[0]))
                        model.prompt_learner.ctx_use[model.prompt_learner.ctx_order[0]] = 0
                        model.prompt_learner.ctx[model.prompt_learner.ctx_order[0]] = model.prompt_learner.ctx[model.prompt_learner.ctx_order[0]] * 0

            optimizer.load_state_dict(optim_state)
            p_s, raw_pred, raw_ent = test_time_tuning(model, images, optimizer, scaler, args, log_file)

        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
            optimizer = None
            pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args)

        # The actual inference goes here
        if args.tpt:
            if args.cocoop:
                image_feature = image_feature[0].unsqueeze(0)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output = model((image_feature, pgen_ctx))
                else:
                    p_s = p_s.reshape(-1)
                    output = model(image, p_s + 1)
                    output = output.view(output.size()[0], p_s.reshape(-1).shape[0], -1)
                    output = output.mean(1)

                    ent = avg_entropy(output.unsqueeze(0))

        log_string(log_file,
                       "Sample: {}, Raw entropy: {:.2f}, Adapted entropy: {:.2f}, Raw pred: {}, Adapted pred: {}, Target: {}".format(
                           i, raw_ent.item(), ent.item(), raw_pred.argmax(-1).item(), output.argmax(-1).item(),
                           target.item()))
        

        if 'ViT' in args.arch:
            softmax_output =  torch.nn.Softmax(dim=1)(output/temperature_value['ViT']) #softmax(output)
        elif 'RN' in args.arch:
            softmax_output =  torch.nn.Softmax(dim=1)(output/temperature_value['RN'])  #softmax(output)

        max_confidence, max_index = torch.max(softmax_output, 1)
        result_dict['max_confidence'].append(max_confidence.item())
        result_dict['prediction'].append(max_index.item())
        result_dict['label'].append(target.item())    
        acc1, _ = accuracy(output, target, topk=(1, 5))

        #### get ece_score, oe_score ####
        list_correct = (torch.tensor(result_dict['prediction']).int() == torch.tensor(result_dict['label']).int()).tolist()
        n_bins=20
        ece_data = ECE_Loss(n_bins, result_dict['prediction'], result_dict['max_confidence'], list_correct)
        ece_score = 100*ece_data[0]
        oe_score = 100*ece_data[1]

        top1.update(acc1[0], image.size(0))
        ece.update(ece_score, image.size(0))
        oe.update(oe_score, image.size(0))

        # measure elapsed time
        if (i+1) % args.print_freq == 0:
            progress.display(i)
            entries = [progress.prefix + progress.batch_fmtstr.format(i)]
            entries += [str(meter) for meter in progress.meters]
            log_string(log_file, '\t'.join(entries))

            performance_dict['iter'].append(entries[0].split('/')[0].split(' ')[-1])
            performance_dict['acc@1'].append(float(entries[-3].split('( ')[-1].split(')')[0]))
            performance_dict['ece'].append(float(entries[-2].split('( ')[-1].split(')')[0]))
            performance_dict['oe'].append(float(entries[-1].split('( ')[-1].split(')')[0]))

    progress.display_summary()

    return [top1.avg, ece.avg, oe.avg], result_dict, performance_dict


temperature_value = {'ViT': 1.16, 'RN': 1.15}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('--data', metavar='DIR', help='path to dataset root', default='../data')
    parser.add_argument('--test_sets', type=str, default='I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default="a_photo_of_a", type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_date', type=str, default='')
    parser.add_argument('--onlinetpt', action='store_true', help='run online prompt tuning')
    parser.add_argument('--num_prompts', default=10, type=int, help='number of prompts to tune')
    parser.add_argument('--lambda_term', type=float, default=5, help='Coefficient for consistency loss')

    main()