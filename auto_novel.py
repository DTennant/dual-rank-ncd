import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD 
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import BCE, cluster_acc, AverageMeter, seed_torch
from utils import ramps 
from data.imagenetloader import ImageNet100_Loader70, ImageNet100_Loader30, ImageNet100_LoaderMix70_30
from tqdm import tqdm
import numpy as np
import os
from utils.util import setup_logger, get_schedule, compute_bce
from utils.util import compute_part_bce


best_acc = 0.0

kl_crit = nn.KLDivLoss(reduction='batchmean')
def symmetric_kld(p, q):
    # p and q are logits
    skld = kl_crit(F.log_softmax(p, 1), F.softmax(q, 1))
    skld += kl_crit(F.log_softmax(q, 1), F.softmax(p, 1))
    return skld / 2

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_( x, y.t(), beta=1, alpha=-2, )
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def train(model, train_loader, labeled_eval_loader, unlabeled_eval_loader, args, logger=None):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = get_schedule(optimizer, args)
    criterion1 = nn.CrossEntropyLoss() 
    criterion2 = BCE() 

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)

        pbar = tqdm(train_loader)
        for batch_idx, ((x, x_bar),  label, idx) in enumerate(train_loader):
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)

            # __import__("ipdb").set_trace()
            outs = model(x, use_ranking=False)
            outs_bar = model(x_bar, use_ranking=False)
            # {
            # 'output': [
            #     [lb_logit_g, ul_logit_g, f_g], 
            #     [lb_logit_p, ul_logit_p, f_p]
            # ], 
            # 'out_featmap': [fm_p]
            # }

            # use mask to find labeled classes
            mask_lb = label < args.num_labeled_classes
            
            # __import__("ipdb").set_trace()
            # --------------------------------------
            # global bce
            out_g, out_g_bar = outs['output'][0], outs_bar['output'][0]
            ub_logits_g, ub_logits_bar_g = out_g[1], out_g_bar[1]
            prob_ub_g, prob_ub_bar_g = F.softmax(ub_logits_g, dim=1), F.softmax(ub_logits_bar_g, dim=1)
            loss_g_bce, ulb_g_acc, p, r = compute_bce(out_g[2], prob_ub_g, prob_ub_bar_g, mask_lb, criterion2, args, return_acc=True, label=label)
            
            loss = loss_g_bce
            pstr = f'g_bce: {loss_g_bce.item():.2f}'
            pstr += f'p:{p:.4f}'
            pstr += f'r:{r:.4f}'
            # --------------------------------------
            # local bce
            # __import__("ipdb").set_trace()
            out_p, out_p_bar = outs['output'][1], outs['output'][1]
            ub_logits_p, ub_logits_bar_p = out_p[1], out_p_bar[1]
            prob_ub_p, prob_ub_bar_p = F.softmax(ub_logits_p, dim=1), F.softmax(ub_logits_bar_p, dim=1)
            fm_p = outs['out_featmap'][0]
            parts_assign, _ = model.part_dict(fm_p)
            loss_p_bce, ulb_acc, p_pa, r_pa = compute_part_bce(parts_assign, prob_ub_p, prob_ub_bar_p, mask_lb, criterion2, args, return_acc=True, label=label)

            loss += loss_p_bce
            pstr += f'pbce: {loss_p_bce.item():.2f}'
            pstr += f'pp:{p_pa:.4f}'
            pstr += f'pr:{r_pa:.4f}'
            # --------------------------------------
            # global and local mutual
            q_g, q_p = outs['output'][0][2], outs['output'][1][2]
            k_g, k_p = outs_bar['output'][0][2], outs_bar['output'][1][2]
            q_g, q_p = F.normalize(q_g[~mask_lb], dim=1), F.normalize(q_p[~mask_lb], dim=1)
            k_g, k_p = F.normalize(k_g[~mask_lb], dim=1), F.normalize(k_p[~mask_lb], dim=1)
            
            queue_g = model.memory_g.detach().clone()
            queue_p = model.memory_p.detach().clone()
            
            logits_g = model.compute_logits(q_g, k_g, queue_g)
            logits_p = model.compute_logits(q_p, k_p, queue_p)
            
            model.update_memory_g(k_g, model.memory_g)
            model.update_pointer_g(k_g.shape[0])
            model.update_memory_p(k_p, model.memory_p)
            model.update_pointer_p(k_p.shape[0])
            
            mutual_loss = symmetric_kld(logits_g, logits_p)

            loss += mutual_loss
            pstr += f'mutual: {mutual_loss.item():.2f}'
            # --------------------------------------
            # ce and consist
            loss_ce, loss_consist = 0.0, 0.0
            for out, out_bar in zip(outs['output'], outs_bar['output']):
                lb_logits, ub_logits, feat = out
                lb_logits_bar, ub_logits_bar, feat2 = out_bar

                prob_lb, prob_lb_bar = F.softmax(lb_logits, dim=1), F.softmax(lb_logits_bar, dim=1)
                prob_ub, prob_ub_bar = F.softmax(ub_logits, dim=1), F.softmax(ub_logits_bar, dim=1)
            
                loss_ce += criterion1(lb_logits[mask_lb], label[mask_lb])
                loss_consist += F.mse_loss(prob_lb, prob_lb_bar) + F.mse_loss(prob_ub, prob_ub_bar)
                
            loss += loss_ce
            pstr += f'ce: {loss_ce.item():.2f} '
            loss += w * loss_consist
            pstr += f'consist: {w * loss_consist.item():.2f} '
            # --------------------------------------

            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(pstr)
            pbar.update()

        pbar.close()
        logger.info('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        logger.info('test on labeled classes')
        args.head = 'head1'
        test(model, labeled_eval_loader, args, logger=logger)
        logger.info('test on unlabeled classes')
        args.head = 'head2'
        test(model, unlabeled_eval_loader, args, logger=logger)


def test(model, test_loader, args, save_test_res=False, logger=None):
    global best_acc
    model.eval()
    preds, targets, softmax_values = np.array([]), np.array([]), np.array([])
    feats = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        output1, output2, feat = model(x)
        if args.head == 'head1':
            output = output1
        else:
            output = output2
        _, pred = output.max(1)
        softmax_value = torch.softmax(output, 1)

        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())
        
        if len(softmax_values):
            softmax_values = np.append(softmax_values, softmax_value.cpu().detach().numpy(), axis=0)
        else:
            softmax_values = softmax_value.cpu().detach().numpy()
            
        if len(feats):
            feats = np.append(feats, feat.cpu().detach().numpy(), axis=0)
        else:
            feats = feat.cpu().detach().numpy()
    
    if save_test_res:
        np.save(os.path.join(args.model_dir, 'test_targets.npy'), targets)
        np.save(os.path.join(args.model_dir, 'test_preds.npy'), preds)
        np.save(os.path.join(args.model_dir, 'test_softmax.npy'), softmax_values)
        np.save(os.path.join(args.model_dir, 'test_feat.npy'), feats)

    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds) 
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    if args.head == 'head2' and acc > best_acc:
        best_acc = acc
        torch.save(model, os.path.join(args.model_dir, f'best_{args.model_name}.pth'))
        if logger is not None:
            logger.info(f'update best acc {best_acc}')
    if logger is not None:
        logger.info(f'best acc {best_acc}')

    if logger is not None:
        logger.info('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}, set {}'.format(acc, nmi, ari, 'label' if args.head == 'head1' else 'unlabel'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)

    parser.add_argument('--rampup_length', default=150, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=50)
    parser.add_argument('--increment_coefficient', type=float, default=0.05)
    parser.add_argument('--step_size', default=170, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--warmup_model_dir', type=str, default='./data/experiments/pretrain/auto_novel/resnet_rotnet_cifar10.pth')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--IL', action='store_true', default=False, help='w/ incremental learning')
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, svhn')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', type=str, default='train')
    
    parser.add_argument('--use_range_class', action='store_true')
    parser.add_argument('--label_range', type=str, default='0,80')
    parser.add_argument('--unlabel_range', type=str, default='80,100')
    
    parser.add_argument('--custom_run', type=str, default='', help='custom for different save')
    parser.add_argument('--seed_cls', action='store_true', help='using seeding to choice class to train for bias variance calc')
    parser.add_argument('--cls_num', type=int, default=20, help='number of cls to be choosen')

    parser.add_argument('--save_test_result', action='store_true')
    
    parser.add_argument('--method', default='rs', 
        help="choices=['baseline', 'rs', 'moe', 'contrastive', 'contrastive_diversity', 'mutual']")
    parser.add_argument('--net', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'moco_r50'])
    parser.add_argument('--moco_path', type=str, default='')
    
    parser.add_argument('--use_nori', action='store_true')
    parser.add_argument('--label_json_path_train', type=str, default='')
    parser.add_argument('--label_json_path_val', type=str, default='')
    parser.add_argument('--unlabel_json_path_train', type=str, default='')
    parser.add_argument('--unlabel_json_path_val', type=str, default='')
    parser.add_argument('--unlabeled_batch_size', type=int, default=64)

    parser.add_argument('--cls_num_from_json', action='store_true')
    
    parser.add_argument('--input_res', type=int, default=224)
    parser.add_argument('--test_use', default='g')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)
    
    if args.use_nori:
        # nori == imagenet100
        args.num_labeled_classes = 70
        args.num_unlabeled_classes = 30
    
    if args.cls_num_from_json:
        import json
        # __import__("ipdb").set_trace()
        l_json = json.load(open(args.label_json_path_train, 'r')) 
        args.num_labeled_classes = l_json['nr_class']
        u_json = json.load(open(args.unlabel_json_path_train, 'r')) 
        args.num_unlabeled_classes = u_json['nr_class']

    runner_name = os.path.basename(__file__).split(".")[0]
    if len(args.custom_run):
        runner_name = f'{runner_name}_{args.custom_run}'
    model_dir = os.path.join(args.exp_root, runner_name)
    model_dir = f'{model_dir}_{args.net}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    args.save_path = os.path.join(model_dir, f'{args.model_name}.pth')
    args.save_path_wopth = os.path.join(model_dir, f'{args.model_name}')
    args.model_dir = model_dir

    logger = setup_logger('auto_novel', args.model_dir, 0)

    from models.resnet import MoCo_R50
    model = MoCo_R50(args.num_labeled_classes, args.num_unlabeled_classes, moco_path=args.moco_path, test_use=args.test_use).to(device)

    for name, param in model.named_parameters(): 
        if 'head' not in name and 'layer4' not in name:
            param.requires_grad = False
            print(name)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
        print(f'Using {torch.cuda.device_count()} gpus')

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    aug = 'twice'
    mix_train_loader = ImageNet100_LoaderMix70_30(  batch_size=args.batch_size, 
                                                    label_json_path=args.label_json_path_train, 
                                                    unlabel_json_path=args.unlabel_json_path_train, 
                                                    aug=aug,
                                                    unlabeled_batch_size=args.unlabeled_batch_size,
                                                    input_res=args.input_res)
    unlabeled_eval_loader = ImageNet100_Loader30(batch_size=args.batch_size,
                                                 json_path=args.unlabel_json_path_train,
                                                 aug=None, 
                                                 input_res=args.input_res)

    unlabeled_eval_loader_test = ImageNet100_Loader30(batch_size=args.batch_size,
                                                      json_path=args.unlabel_json_path_val,
                                                      aug=None, 
                                                      input_res=args.input_res)
    labeled_eval_loader = ImageNet100_Loader70(batch_size=args.batch_size,
                                               json_path=args.label_json_path_val,
                                               aug=None, 
                                               input_res=args.input_res)

    logger.info(str(args))
     
    if args.mode == 'train':

        train(model, mix_train_loader, labeled_eval_loader, unlabeled_eval_loader, args, logger)

        torch.save(model.state_dict(), args.save_path)
        print("model saved to {}.".format(args.save_path))
    else:
        print("model loaded from {}.".format(args.save_path))
        model.load_state_dict(torch.load(args.save_path))

    print('Evaluating on Head1')
    args.head = 'head1'
    print('test on labeled classes (test split)')
    test(model, labeled_eval_loader, args)


    print('Evaluating on Head2')
    args.head = 'head2'
    print('test on unlabeled classes (train split)')
    test(model, unlabeled_eval_loader, args)
    print('test on unlabeled classes (test split)')
    test(model, unlabeled_eval_loader_test, args, save_test_res=args.save_test_result)
    if logger is not None:
        logger.info(f'best acc on unlabel is {best_acc}')
