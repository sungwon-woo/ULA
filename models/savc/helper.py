# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

from losses import SupContrastive
import logging, os, math
logging.basicConfig(level=logging.INFO, format="%(message)s")

def base_train(model, trainloader, criterion, optimizer, scheduler, epoch, transform, args):
    tl = Averager()
    tl_joint = Averager()
    tl_moco = Averager()
    tl_moco_global = Averager()
    tl_moco_small = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, single_labels = [_ for _ in batch]
        b, c, h, w = data[1].shape
        original = data[0].cuda(non_blocking=True)
        data[1] = data[1].cuda(non_blocking=True)
        data[2] = data[2].cuda(non_blocking=True)
        single_labels = single_labels.cuda(non_blocking=True)
        if len(args.num_crops) > 1:
            data_small = data[args.num_crops[0]+1].unsqueeze(1)
            for j in range(1, args.num_crops[1]):
                data_small = torch.cat((data_small, data[j+args.num_crops[0]+1].unsqueeze(1)), dim=1)
            data_small = data_small.view(-1, c, args.size_crops[1], args.size_crops[1]).cuda(non_blocking=True)
        else:
            data_small = None
        
        data_classify = transform(original)    
        data_query = transform(data[1])
        data_key = transform(data[2])
        data_small = transform(data_small)
        m = data_query.size()[0] // b
        joint_labels = torch.stack([single_labels*m+ii for ii in range(m)], 1).view(-1)
        
        joint_preds, output_global, output_small, target_global, target_small = model(im_cla=data_classify, im_q=data_query, im_k=data_key, labels=joint_labels, im_q_small=data_small)
        loss_moco_global = criterion(output_global, target_global)
        loss_moco_small = criterion(output_small, target_small)
        loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small

        joint_preds = joint_preds[:, :args.base_class*m]
        joint_loss = F.cross_entropy(joint_preds, joint_labels)

        agg_preds = 0
        for i in range(m):
            agg_preds = agg_preds + joint_preds[i::m, i::m] / m

        loss = joint_loss + loss_moco
        total_loss = loss
        
        acc = count_acc(agg_preds, single_labels)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        tl_joint.add(joint_loss.item())
        tl_moco_global.add(loss_moco_global.item())
        tl_moco_small.add(loss_moco_small.item())
        tl_moco.add(loss_moco.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    tl_joint = tl_joint.item()
    tl_moco = tl_moco.item()
    tl_moco_global = tl_moco_global.item()
    tl_moco_small = tl_moco_small.item()
    return tl, tl_joint, tl_moco, tl_moco_global, tl_moco_small, ta


def replace_base_fc(trainset, test_transform, data_transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = test_transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = data_transform(data)
            m = data.size()[0] // b
            labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
            model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(labels.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class*m):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.fc.weight.data[:args.base_class*m] = proto_list

    return model


def update_fc_ft(trainloader, data_transform, model, m, session, args):
    # incremental finetuning
    old_class = args.base_class + args.way * (session - 1)
    new_class = args.base_class + args.way * session 
    new_fc = nn.Parameter(
        torch.rand(args.way*m, model.num_features, device="cuda"),
        requires_grad=True)
    new_fc.data.copy_(model.fc.weight[old_class*m : new_class*m, :].data)
    
    if args.dataset == 'mini_imagenet':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.encoder_q.fc.parameters(), 'lr': 0.05*args.lr_new},
                                     {'params': model.encoder_q.layer4.parameters(), 'lr': 0.001*args.lr_new},],
                                    momentum=0.9, dampening=0.9, weight_decay=0)
        
    if args.dataset == 'cub200':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new}],
                                    momentum=0.9, dampening=0.9, weight_decay=0)
        
    elif args.dataset == 'cifar100':
        optimizer = torch.optim.Adam([{'params': new_fc, 'lr': args.lr_new},
                                      {'params': model.encoder_q.fc.parameters(), 'lr': 0.01*args.lr_new},
                                      {'params': model.encoder_q.layer3.parameters(), 'lr':0.02*args.lr_new}],
                                      weight_decay=0)
        
    criterion = SupContrastive().cuda() 

    with torch.enable_grad():
        for epoch in range(args.epochs_new):
            for batch in trainloader:
                data, single_labels = [_ for _ in batch]
                b, c, h, w = data[1].shape
                origin = data[0].cuda(non_blocking=True)
                data[1] = data[1].cuda(non_blocking=True)
                data[2] = data[2].cuda(non_blocking=True)
                single_labels = single_labels.cuda(non_blocking=True)
                if len(args.num_crops) > 1:
                    data_small = data[args.num_crops[0]+1].unsqueeze(1)
                    for j in range(1, args.num_crops[1]):
                        data_small = torch.cat((data_small, data[j+args.num_crops[0]+1].unsqueeze(1)), dim=1)
                    data_small = data_small.view(-1, c, args.size_crops[1], args.size_crops[1]).cuda(non_blocking=True)
                else:
                    data_small = None
            data_classify = data_transform(origin)    
            data_query = data_transform(data[1])
            data_key = data_transform(data[2])
            data_small = data_transform(data_small)
            joint_labels = torch.stack([single_labels*m+ii for ii in range(m)], 1).view(-1)
            
            old_fc = model.fc.weight[:old_class*m, :].clone().detach()    
            fc = torch.cat([old_fc, new_fc], dim=0)
            features, _ = model.encode_q(data_classify)
            features.detach()
            logits = model.get_logits(features,fc)
            joint_loss = F.cross_entropy(logits, joint_labels)
            _, output_global, output_small, target_global, target_small = model(im_cla=data_classify, im_q=data_query, im_k=data_key, labels=joint_labels, im_q_small=data_small, base_sess=False, last_epochs_new=(epoch==args.epochs_new-1))
            loss_moco_global = criterion(output_global, target_global)
            loss_moco_small = criterion(output_small, target_small)
            loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small 
            loss = joint_loss + loss_moco         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.fc.weight.data[old_class*m : new_class*m, :].copy_(new_fc.data)

# def test(model, testloader, epoch, transform, args, session):
#     test_class = args.base_class + session * args.way
#     model = model.eval()
#     vl = Averager()
#     va = Averager()
#     with torch.no_grad():
#         tqdm_gen = tqdm(testloader)
#         for i, batch in enumerate(tqdm_gen, 1):
#             data, test_label = [_.cuda() for _ in batch]
#             b = data.size()[0]
#             data = transform(data)
#             m = data.size()[0] // b
#             joint_preds = model(data)
#             joint_preds = joint_preds[:, :test_class*m]
            
#             agg_preds = 0
#             for j in range(m):
#                 agg_preds = agg_preds + joint_preds[j::m, j::m] / m
            
#             loss = F.cross_entropy(agg_preds, test_label)
#             acc = count_acc(agg_preds, test_label)

#             vl.add(loss.item())
#             va.add(acc)

#         vl = vl.item()
#         va = va.item()
#     print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

#     return vl,va


@torch.no_grad()
def compute_fnr_fpr_group(lgt: torch.Tensor,
                          lbs: torch.Tensor,
                          base_num: int):
    """
    TEEN 스타일의 FNR/FPR 계산.
    Positive = base classes (label < base_num)
    Negative = incremental classes (label >= base_num)
    lgt: [N, C] logits (already truncated to test_class)
    lbs: [N]    ground-truth labels
    """
    # argmax 기반 predicted class
    preds = torch.argmax(lgt, dim=1)

    # 그룹 단위 마스크
    is_base_true = (lbs < base_num)
    is_base_pred = (preds < base_num)

    TP = (is_base_true & is_base_pred).sum().item()
    FN = (is_base_true & ~is_base_pred).sum().item()
    FP = (~is_base_true & is_base_pred).sum().item()
    TN = (~is_base_true & ~is_base_pred).sum().item()

    eps = 1e-12
    fnr = FN / (FN + TP + eps) * 100.0
    fpr = FP / (FP + TN + eps) * 100.0
    return fnr, fpr

# def test(model, testloader, epoch, transform, args, session):
#     import os
#     import numpy as np
#     import torch
#     import torch.nn.functional as F
#     from tqdm import tqdm

#     test_class = args.base_class + session * args.way
#     model = model.eval()
#     vl = Averager()
#     va = Averager()

#     # logits, labels 저장
#     all_logits = []
#     all_labels = []

#     with torch.no_grad():
#         tqdm_gen = tqdm(testloader)
#         for i, batch in enumerate(tqdm_gen, 1):
#             data, test_label = [_.cuda() for _ in batch]
#             b = data.size(0)
#             data = transform(data)
#             m = data.size(0) // b

#             joint_preds = model(data)
#             joint_preds = joint_preds[:, :test_class * m]

#             agg_preds = 0
#             for j in range(m):
#                 agg_preds = agg_preds + joint_preds[j::m, j::m] / m

#             loss = F.cross_entropy(agg_preds, test_label)
#             acc = count_acc(agg_preds, test_label)

#             vl.add(loss.item())
#             va.add(acc)

#             all_logits.append(agg_preds.cpu())
#             all_labels.append(test_label.cpu())

#         vl = vl.item()
#         va = va.item()

#     print(f'epo {epoch}, test, loss={vl:.4f} acc={va:.4f}')

#     # --- seen / unseen accuracy 계산 ---
#     if session > 0:
#         all_logits = torch.cat(all_logits, dim=0)
#         all_labels = torch.cat(all_labels, dim=0)

#         # confusion matrix 계산
#         save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
#         cm = confmatrix(all_logits, all_labels, save_model_dir)

#         per_class_acc = cm.diagonal()
#         seen_acc   = np.mean(per_class_acc[:args.base_class])
#         unseen_acc = np.mean(per_class_acc[args.base_class:])

#         # === FNR / FPR 추가 ===
#         fnr, fpr = compute_fnr_fpr_group(
#             all_logits, all_labels, base_num=args.base_class
#         )
#         logging.info(f"[Orig] Session {session} ⇒ FNR={fnr:.2f}%, FPR={fpr:.2f}% (pos=base, neg=incr)")
#         # =======================

#         print(f"Seen Acc: {seen_acc:.4f}, Unseen Acc: {unseen_acc:.4f}")
#         return vl, (seen_acc, unseen_acc, va)
#     else:
#         return vl, va


def test(model, testloader, epoch, transform, args, session):
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    import logging

    test_class = args.base_class + session * args.way
    base_num = args.base_class

    model = model.eval()
    vl = Averager()
    va = Averager()

    # logits, labels 저장
    all_logits = []
    all_labels = []

    # ===== [ADD] Top-2 pattern counters (orig) =====
    # 전체
    cnt_A_all = 0   # top1=base, top2=incr
    cnt_B_all = 0   # top1=incr, top2=incr
    N_all = 0
    # GT=incr만
    cnt_A_gtincr = 0
    cnt_B_gtincr = 0
    N_gtincr = 0
    # =============================================

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.size(0)
            data = transform(data)
            m = data.size(0) // b

            joint_preds = model(data)
            joint_preds = joint_preds[:, :test_class * m]

            agg_preds = 0
            for j in range(m):
                agg_preds = agg_preds + joint_preds[j::m, j::m] / m

            # ===== [ADD] compute top-2 patterns on ORIGINAL logits (agg_preds) =====
            top2_idx = torch.topk(agg_preds, k=2, dim=1).indices
            i1 = top2_idx[:, 0]
            i2 = top2_idx[:, 1]

            is_top1_base = (i1 < base_num)
            is_top1_incr = ~is_top1_base
            is_top2_incr = (i2 >= base_num)

            mask_A = is_top1_base & is_top2_incr          # base -> incr
            mask_B = is_top1_incr & is_top2_incr          # incr -> incr (전체)

            # ✅ 추가: top1 정답 여부
            pred1_correct = (i1 == test_label)
            mask_B_wrong = mask_B & (~pred1_correct)      # incr->incr 중 top1 오답만

            # ALL
            cnt_A_all += mask_A.sum().item()
            cnt_B_all += mask_B.sum().item()
            N_all += test_label.size(0)

            # GT=incr
            is_gt_incr = (test_label >= base_num)
            cnt_A_gtincr += (is_gt_incr & mask_A).sum().item()

            # ✅ 변경: B는 "GT=incr & (top1=incr, top2=incr) & top1 오답"
            cnt_B_gtincr += (is_gt_incr & mask_B_wrong).sum().item()

            N_gtincr += is_gt_incr.sum().item()
            # ===================================================================

            loss = F.cross_entropy(agg_preds, test_label)
            acc = count_acc(agg_preds, test_label)

            vl.add(loss.item())
            va.add(acc)

            all_logits.append(agg_preds.cpu())
            all_labels.append(test_label.cpu())

        vl = vl.item()
        va = va.item()

    print(f'epo {epoch}, test, loss={vl:.4f} acc={va:.4f}')

    # ===== [ADD] log pattern ratios (orig) =====
    eps = 1e-12
    if session > 0:
        # 전체 기준
        logging.info(
            f"[Orig] Session {session} Top-2 pattern (ALL): "
            f"A(base→incr)={cnt_A_all} ({cnt_A_all/(N_all+eps):.4f}), "
            f"B(incr→incr)={cnt_B_all} ({cnt_B_all/(N_all+eps):.4f}), "
            f"A/B={(cnt_A_all+eps)/(cnt_B_all+eps):.3f}"
        )
        # GT=incr 기준 (핵심)
        if N_gtincr > 0:
            logging.info(
                f"[Orig] Session {session} Top-2 pattern (GT=incr): "
                f"A(base→incr)={cnt_A_gtincr} ({cnt_A_gtincr/(N_gtincr+eps):.4f}), "
                f"B(incr→incr)={cnt_B_gtincr} ({cnt_B_gtincr/(N_gtincr+eps):.4f}), "
                f"A/B={(cnt_A_gtincr+eps)/(cnt_B_gtincr+eps):.3f}"
            )
    # ===========================================

    # --- seen / unseen accuracy 계산 ---
    if session > 0:
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
        cm = confmatrix(all_logits, all_labels, save_model_dir)

        per_class_acc = cm.diagonal()
        seen_acc   = np.mean(per_class_acc[:args.base_class])
        unseen_acc = np.mean(per_class_acc[args.base_class:])

        fnr, fpr = compute_fnr_fpr_group(all_logits, all_labels, base_num=args.base_class)
        logging.info(f"[Orig] Session {session} ⇒ FNR={fnr:.2f}%, FPR={fpr:.2f}% (pos=base, neg=incr)")

        print(f"Seen Acc: {seen_acc:.4f}, Unseen Acc: {unseen_acc:.4f}")
        return vl, (seen_acc, unseen_acc, va)
    else:
        return vl, va

# -------------------------------------------------
# Adaptive Logit Adjustment (margin + base-entropy)
# + Top-2 pattern counting (ALL / GT=incr) for Orig and ADJ
# + Rebuttal: margin-binned H_base (GT=base vs GT=incr) under the same margin
# -------------------------------------------------
@torch.no_grad()
def adaptive_logit_adjust(model, testloader, epoch, transform, args, session, result_list=None):
    import os
    import math
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    test_class = args.base_class + session * args.way
    base_num   = args.base_class

    model = model.eval()
    vl = Averager()
    va = Averager()

    # logits/labels 누적 (seen/unseen 계산용)
    all_logits = []
    all_labels = []

    # --- 분석용 버퍼 (옵션: 기존 w/margin 분석) ---
    w_all = []
    marg_all = []
    maskA_all = []   # pred1∈base & pred2∈incr & small_margin
    maskB_all = []   # pred1∈incr & pred2∈base & small_margin
    is_base_pred_all = []
    is_incr_pred_all = []

    # -----------------------------
    # Top-2 pattern counters
    # -----------------------------
    # Orig (before adjustment)
    orig_A_all = 0
    orig_B_all = 0
    orig_N_all = 0

    orig_A_incrGT = 0
    orig_B_incrGT = 0
    orig_N_incrGT = 0

    # ADJ (after adjustment)
    adj_A_all = 0
    adj_B_all = 0
    adj_N_all = 0

    adj_A_incrGT = 0
    adj_B_incrGT = 0
    adj_N_incrGT = 0

    # --- Rebuttal: margin-binned base-entropy buffers ---
    Hbase_all = []          # per-sample H_base
    gt_is_base_all = []     # GT base? (label < base_num)
    top2_base_incr_all = [] # condition: (pred1 base, pred2 incr)
    # (선택) small_margin까지 포함한 조건을 쓰고 싶으면 아래 버퍼를 쓰거나,
    # top2_base_incr_all에 small_margin까지 포함된 cond를 넣으면 됨.

    def _update_top2_counters(logits_2d, labels_1d, is_adj: bool):
        """
        logits_2d: [B, C]
        labels_1d: [B]
        Updates A/B counts for ALL and GT=incr.

        A: (pred1 base, pred2 incr)
        B: (pred1 incr, pred2 incr) AND (top1 is WRONG)   <-- changed
        """
        nonlocal orig_A_all, orig_B_all, orig_N_all
        nonlocal orig_A_incrGT, orig_B_incrGT, orig_N_incrGT
        nonlocal adj_A_all, adj_B_all, adj_N_all
        nonlocal adj_A_incrGT, adj_B_incrGT, adj_N_incrGT

        _, top2_idx = torch.topk(logits_2d, k=2, dim=1)  # [B,2]
        i1 = top2_idx[:, 0].long()
        i2 = top2_idx[:, 1].long()
        labels_1d = labels_1d.long()

        pred1_is_base = (i1 < base_num)
        pred2_is_incr = (i2 >= base_num)
        pred1_is_incr = ~pred1_is_base

        # A: base -> incr
        A_mask = pred1_is_base & pred2_is_incr

        # B: incr -> incr (but wrong-top1 only)
        B_mask = pred1_is_incr & pred2_is_incr
        pred1_correct = (i1 == labels_1d)
        B_mask_wrong = B_mask & (~pred1_correct)

        # ALL
        A_all = int(A_mask.sum().item())
        B_all = int(B_mask_wrong.sum().item())  # ✅ (추천) ALL도 wrong-only로 통일
        N_all = int(labels_1d.numel())

        # GT=incr only
        gt_incr = (labels_1d >= base_num)
        A_incr = int((A_mask & gt_incr).sum().item())
        B_incr = int((B_mask_wrong & gt_incr).sum().item())
        N_incr = int(gt_incr.sum().item())

        if not is_adj:
            orig_A_all += A_all; orig_B_all += B_all; orig_N_all += N_all
            orig_A_incrGT += A_incr; orig_B_incrGT += B_incr; orig_N_incrGT += N_incr
        else:
            adj_A_all += A_all; adj_B_all += B_all; adj_N_all += N_all
            adj_A_incrGT += A_incr; adj_B_incrGT += B_incr; adj_N_incrGT += N_incr


    # -----------------------------
    # Hyper / options
    # -----------------------------
    tau          = float(getattr(args, 'adj_tau', 0.06))
    use_entropy  = bool(getattr(args, 'adj_use_entropy', True))
    entropy_mode = str(getattr(args, 'adj_entropy_mode', 'soft'))  # 'soft' or 'hard'
    lam_base     = float(getattr(args, 'adj_lambda_base', 0.1))
    lam_incr     = float(getattr(args, 'adj_lambda_incr', 0.2))
    max_shift    = float(getattr(args, 'adj_max_shift', 0.06))
    enable_B     = bool(getattr(args, 'adj_enable_caseB', False))

    # entropy hyper
    if use_entropy and entropy_mode == 'soft':
        T_B  = float(getattr(args, 'adj_T_base', 0.06))
        a    = float(getattr(args, 'adj_entropy_alpha', 0.1))
        t_th = float(getattr(args, 'adj_entropy_thresh', 0.06))
    elif use_entropy and entropy_mode == 'hard':
        t_hard = float(getattr(args, 'adj_entropy_hard_t', 0.70))
        mode_h = str(getattr(args, 'adj_entropy_hard_mode', 'one_sided'))  # 'one_sided' or 'bi'
    elif use_entropy:
        raise ValueError(f"Unknown entropy_mode: {entropy_mode}")

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]

            b = data.size(0)
            data_aug = transform(data)
            m = data_aug.size(0) // b if data_aug.size(0) % b == 0 else 1

            joint_preds = model(data_aug)  # [m*b, C_all]
            joint_preds = joint_preds[:, :test_class * m] if m > 1 else joint_preds[:, :test_class]

            if m > 1:
                agg_preds = 0
                for j in range(m):
                    agg_preds = agg_preds + joint_preds[j::m, j::m] / m
            else:
                agg_preds = joint_preds

            # -----------------------------
            # Orig logits (before adjustment)
            # -----------------------------
            logits = agg_preds.clone()
            _update_top2_counters(logits, test_label, is_adj=False)

            # 1) top-2 & margin
            top2_vals, top2_idx = torch.topk(logits, k=2, dim=1)  # [B,2]
            z1, z2 = top2_vals[:, 0], top2_vals[:, 1]
            i1, i2 = top2_idx[:, 0].long(), top2_idx[:, 1].long()
            margin = z1 - z2

            pred1_is_base = i1 < base_num
            pred2_is_incr = i2 >= base_num
            pred1_is_incr = ~pred1_is_base
            pred2_is_base = ~pred2_is_incr

            # 3) margin gate h(m)
            small_margin = margin < tau
            h = torch.clamp(tau - margin, min=0) / max(tau, 1e-6)

            # 4) entropy (base-only entropy)
            if use_entropy:
                z_base = logits[:, :base_num] / max(T_B, 1e-6) if entropy_mode == 'soft' else logits[:, :base_num]
                p_base = torch.softmax(z_base, dim=1)
                eps = 1e-12
                H_base = -(p_base * (p_base.clamp_min(eps).log())).sum(dim=1) / math.log(base_num + eps)

                if entropy_mode == 'soft':
                    w = torch.sigmoid(a * (H_base - t_th))  # [B] in (0,1)
                    w_all.append(w.detach().cpu())
                else:
                    incr_flag = (H_base >= t_hard).float()  # [B] {0,1}

                # ---- collect for rebuttal (NOW H_base exists) ----
                # default: Top1=base & Top2=incr
                cond_top2 = (pred1_is_base & pred2_is_incr)
                # If you want to focus on ambiguous cases only, use:
                # cond_top2 = (pred1_is_base & pred2_is_incr & small_margin)
                Hbase_all.append(H_base.detach().cpu())
                gt_is_base_all.append((test_label < base_num).detach().cpu())
                top2_base_incr_all.append(cond_top2.detach().cpu())

            # 기존 분석 버퍼
            marg_all.append(margin.detach().cpu())
            maskA_all.append((pred1_is_base & pred2_is_incr & small_margin).detach().cpu())
            maskB_all.append((pred1_is_incr & pred2_is_base & small_margin).detach().cpu())
            is_base_pred_all.append(pred1_is_base.detach().cpu())
            is_incr_pred_all.append(pred1_is_incr.detach().cpu())

            # -----------------------------
            # Logit adjustment
            # -----------------------------
            calibrated = logits.clone()

            # Case A: pred1 ∈ base, pred2 ∈ incr, margin<τ
            maskA = pred1_is_base & pred2_is_incr & small_margin
            if maskA.any():
                rowsA = torch.nonzero(maskA, as_tuple=False).squeeze(1)
                if use_entropy:
                    if entropy_mode == 'soft':
                        deltaA = (lam_base * h[maskA] * w[maskA]).clamp(max=max_shift)
                        calibrated[rowsA, i1[maskA]] -= deltaA
                        calibrated[rowsA, i2[maskA]] += deltaA
                    else:  # hard
                        hA = h[maskA]
                        lam_incrA = float(getattr(args, 'adj_lambda_base_incr', lam_base))
                        lam_baseA = float(getattr(args, 'adj_lambda_base_base', 0.00 if mode_h == 'one_sided' else 0.05))
                        incr_flag_A = incr_flag[maskA]
                        base_flag_A = 1.0 - incr_flag_A
                        net_delta = (lam_incrA * hA * incr_flag_A - lam_baseA * hA * base_flag_A).clamp(min=-max_shift, max=max_shift)
                        calibrated[rowsA, i1[maskA]] -= net_delta
                        calibrated[rowsA, i2[maskA]] += net_delta
                else:
                    deltaA = (lam_base * h[maskA]).clamp(max=max_shift)
                    calibrated[rowsA, i1[maskA]] -= deltaA
                    calibrated[rowsA, i2[maskA]] += deltaA

            # Case B: pred1 ∈ incr, pred2 ∈ base, margin<τ (optional)
            if enable_B and lam_incr > 0:
                maskB = pred1_is_incr & pred2_is_base & small_margin
                if maskB.any():
                    rowsB = torch.nonzero(maskB, as_tuple=False).squeeze(1)
                    if use_entropy and entropy_mode == 'soft':
                        deltaB = (lam_incr * h[maskB] * (1.0 - w[maskB])).clamp(max=max_shift)
                        calibrated[rowsB, i1[maskB]] -= deltaB
                        calibrated[rowsB, i2[maskB]] += deltaB
                    else:
                        deltaB = (lam_incr * h[maskB]).clamp(max=max_shift)
                        calibrated[rowsB, i1[maskB]] -= deltaB
                        calibrated[rowsB, i2[maskB]] += deltaB

            # ---- 보정된 점수 사용 ----
            logits = calibrated

            # -----------------------------
            # ADJ logits (after adjustment)
            # -----------------------------
            _update_top2_counters(logits, test_label, is_adj=True)

            # 평가
            loss = F.cross_entropy(logits, test_label)
            acc  = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)

            all_logits.append(logits.detach().cpu())
            all_labels.append(test_label.detach().cpu())

    # -----------------------------
    # (옵션) w 분석(기존 유지)
    # -----------------------------
    def _cat_safe(x_list):
        import numpy as np, torch
        if len(x_list) == 0:
            return np.empty((0,), dtype=float)
        first = x_list[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(x_list, dim=0).detach().cpu().view(-1).numpy()
        return np.concatenate(x_list, axis=0)

    def _summarize(name, arr_np):
        import numpy as np, logging
        arr_np = np.asarray(arr_np)
        if arr_np.size == 0:
            logging.info(f"[{session:02d}] w {name} | N=0 (skip)")
            return
        arr_np = arr_np[~np.isnan(arr_np)]
        if arr_np.size == 0:
            logging.info(f"[{session:02d}] w {name} | N=0 after NaN filter (skip)")
            return
        n = arr_np.size
        mean = float(arr_np.mean()); std = float(arr_np.std())
        amin = float(arr_np.min());  amax = float(arr_np.max())
        probs = [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]
        try:
            q = np.quantile(arr_np, probs)
        except Exception:
            q = np.percentile(arr_np, [1,5,10,25,50,75,90,95,99])
        logging.info(
            f"[{session:02d}] w {name} | N={n} "
            f"mean={mean:.4f}, std={std:.4f}, min={amin:.4f}, max={amax:.4f}, "
            f"q01={q[0]:.4f}, q05={q[1]:.4f}, q10={q[2]:.4f}, q25={q[3]:.4f}, "
            f"q50={q[4]:.4f}, q75={q[5]:.4f}, q90={q[6]:.4f}, q95={q[7]:.4f}, q99={q[8]:.4f}"
        )

    marg_arr      = _cat_safe(marg_all)
    A_arr         = _cat_safe(maskA_all).astype(bool)
    B_arr         = _cat_safe(maskB_all).astype(bool)
    pred_base_arr = _cat_safe(is_base_pred_all).astype(bool)
    pred_incr_arr = _cat_safe(is_incr_pred_all).astype(bool)

    if use_entropy and entropy_mode == 'soft' and len(w_all) > 0:
        w_arr = _cat_safe(w_all)

        _summarize("ALL", w_arr)
        if A_arr.any(): _summarize("maskA(pred1∈base, pred2∈incr, small_margin)", w_arr[A_arr])
        if B_arr.any(): _summarize("maskB(pred1∈incr, pred2∈base, small_margin)", w_arr[B_arr])
        if pred_base_arr.any(): _summarize("pred_base", w_arr[pred_base_arr])
        if pred_incr_arr.any(): _summarize("pred_incr", w_arr[pred_incr_arr])

        if w_arr.size > 0:
            bins_w = np.linspace(0.0, 1.0, 11)
            hist_all, _ = np.histogram(w_arr, bins=bins_w)
            import logging
            logging.info(f"[{session:02d}] w hist(ALL, bins=0..1 step .1): {hist_all.tolist()}")
            if A_arr.any():
                hist_A, _ = np.histogram(w_arr[A_arr], bins=bins_w)
                logging.info(f"[{session:02d}] w hist(maskA): {hist_A.tolist()}")

        if w_arr.size > 1 and marg_arr.size == w_arr.size:
            corr = np.corrcoef(w_arr, marg_arr)[0, 1]
            import logging
            logging.info(f"[{session:02d}] corr(w, margin) = {corr:.4f}")
    else:
        import logging
        logging.info(f"[{session:02d}] entropy disabled / non-soft / no w collected; skip w-distribution analysis.")

    # =========================================================
    # Rebuttal analysis: Margin-binned H_base separation
    # Condition: Top1=base & Top2=incr (optionally + small_margin)
    # =========================================================
    import logging
    if use_entropy and len(Hbase_all) > 0:
        m_np    = marg_arr
        h_np    = _cat_safe(Hbase_all)
        gtB_np  = _cat_safe(gt_is_base_all).astype(bool)
        cond_np = _cat_safe(top2_base_incr_all).astype(bool)

        # lengths should match
        n_min = min(m_np.shape[0], h_np.shape[0], gtB_np.shape[0], cond_np.shape[0])
        m_np, h_np, gtB_np, cond_np = m_np[:n_min], h_np[:n_min], gtB_np[:n_min], cond_np[:n_min]

        # apply condition
        m_np   = m_np[cond_np]
        h_np   = h_np[cond_np]
        gtB_np = gtB_np[cond_np]

        bins_m = getattr(args, "adj_margin_bins", [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 1e6])

        logging.info(f"[Session {session}] Margin-binned H_base (Top1=base, Top2=incr)")
        for b0, b1 in zip(bins_m[:-1], bins_m[1:]):
            in_bin = (m_np >= b0) & (m_np < b1)
            n_bin = int(in_bin.sum())
            if n_bin < 30:
                continue

            hb_base = h_np[in_bin & gtB_np]        # GT=base
            hb_incr = h_np[in_bin & (~gtB_np)]     # GT=incr
            if hb_base.size < 10 or hb_incr.size < 10:
                continue

            mean_base = float(hb_base.mean())
            mean_incr = float(hb_incr.mean())
            diff = mean_incr - mean_base

            logging.info(
                f"  margin∈[{b0:.3f},{b1:.3f}) N={n_bin} | "
                f"H_base(GT=base)={mean_base:.4f} (n={hb_base.size}) | "
                f"H_base(GT=incr)={mean_incr:.4f} (n={hb_incr.size}) | "
                f"Δ={diff:+.4f}"
            )
    else:
        logging.info(f"[Session {session}] Skip margin-binned H_base analysis (entropy off or empty).")

    vl = vl.item()
    va = va.item()
    print(f'epo {epoch}, test (after logit adj), loss={vl:.4f} acc={va:.4f}')

    # -----------------------------
    # Print Top-2 pattern summaries
    # -----------------------------
    def _fmt_line(tag, A, B, N, name):
        pA = (A / max(N, 1e-12))
        pB = (B / max(N, 1e-12))
        ratio = (A / max(B, 1e-12))
        return f"[{tag:4s}] Session {session} Top-2 pattern ({name}): A(base→incr)={A} ({pA:.4f}), B(incr→incr)={B} ({pB:.4f}), A/B={ratio:.3f}"

    line1 = _fmt_line("Orig", orig_A_incrGT, orig_B_incrGT, orig_N_incrGT, "GT=incr")
    line2 = _fmt_line("ADJ",  adj_A_incrGT,  adj_B_incrGT,  adj_N_incrGT,  "GT=incr")
    line3 = _fmt_line("Orig", orig_A_all,    orig_B_all,    orig_N_all,    "ALL")
    line4 = _fmt_line("ADJ",  adj_A_all,     adj_B_all,     adj_N_all,     "ALL")

    import logging
    logging.info(line1); logging.info(line2); logging.info(line3); logging.info(line4)
    if result_list is not None:
        result_list.append(line1); result_list.append(line2); result_list.append(line3); result_list.append(line4)

    # --- seen / unseen accuracy 계산 ---
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if session > 0:
        save_model_dir = os.path.join(args.save_path, f'session{session}_confusion_matrix_after ULA')
        cm = confmatrix(all_logits, all_labels, save_model_dir)
        per_class_acc = cm.diagonal()
        seen_acc   = np.mean(per_class_acc[:args.base_class])
        unseen_acc = np.mean(per_class_acc[args.base_class:])

        # === FNR / FPR 추가 ===
        fnr, fpr = compute_fnr_fpr_group(all_logits, all_labels, base_num=args.base_class)
        logging.info(f"[Adj ] Session {session} ⇒ FNR={fnr:.2f}%, FPR={fpr:.2f}% (pos=base, neg=incr)")

        print(f"[After Logit Adjustment] Seen Acc: {seen_acc:.4f}, Unseen Acc: {unseen_acc:.4f}")
        if result_list is not None:
            result_list.append(
                f"[After Logit Adjustment] Session {session} ==> "
                f"Seen Acc: {seen_acc*100:.3f}, "
                f"Unseen Acc: {unseen_acc*100:.3f}, "
                f"Avg Acc: {va*100:.3f}, "
                f"Harmonic Mean: { (2*seen_acc*unseen_acc/(seen_acc+unseen_acc) if (seen_acc+unseen_acc)>0 else 0.0)*100:.3f}\n"
            )
        return vl, (seen_acc, unseen_acc, va)
    else:
        return vl, va




# # -------------------------------------------------
# # Adaptive Logit Adjustment (margin + base-entropy)
# # -------------------------------------------------
# @torch.no_grad()
# def adaptive_logit_adjust(model, testloader, epoch, transform, args, session, result_list=None):
#     import os
#     import math
#     import numpy as np
#     import torch
#     import torch.nn.functional as F
#     from tqdm import tqdm

#     test_class = args.base_class + session * args.way
#     base_num   = args.base_class

#     model = model.eval()
#     vl = Averager()
#     va = Averager()

#     # logits/labels 누적 (seen/unseen 계산용)
#     all_logits = []
#     all_labels = []

#     # --- 분석용 버퍼 (옵션) ---
#     w_all = []
#     marg_all = []
#     maskA_all = []   # pred1∈base & pred2∈incr & small_margin
#     maskB_all = []   # pred1∈incr & pred2∈base & small_margin
#     is_base_pred_all = []
#     is_incr_pred_all = []

#     # 하이퍼/옵션
#     tau         = float(getattr(args, 'adj_tau', 0.06))
#     use_entropy = bool(getattr(args, 'adj_use_entropy', True))
#     entropy_mode = str(getattr(args, 'adj_entropy_mode', 'soft'))  # 'soft' or 'hard'
#     lam_base    = float(getattr(args, 'adj_lambda_base', 0.1))  
#     lam_incr    = float(getattr(args, 'adj_lambda_incr', 0.2))    
#     max_shift   = float(getattr(args, 'adj_max_shift', 0.06))
#     enable_B    = bool(getattr(args, 'adj_enable_caseB', False))

#     if use_entropy and entropy_mode == 'soft':
#         T_B   = float(getattr(args, 'adj_T_base', 0.06))
#         a     = float(getattr(args, 'adj_entropy_alpha', 0.1))
#         t_th  = float(getattr(args, 'adj_entropy_thresh', 0.06))
#     elif use_entropy and entropy_mode == 'hard':
#         t_hard = float(getattr(args, 'adj_entropy_hard_t', 0.70))
#         mode_h = str(getattr(args, 'adj_entropy_hard_mode', 'one_sided'))  # 'one_sided' or 'bi'

#     with torch.no_grad():
#         tqdm_gen = tqdm(testloader)
#         for i, batch in enumerate(tqdm_gen, 1):
#             data, test_label = [_.cuda() for _ in batch]

#             b = data.size(0)
#             data_aug = transform(data)          
#             m = data_aug.size(0) // b if data_aug.size(0) % b == 0 else 1

#             joint_preds = model(data_aug)          # [m*b, C_all]
#             joint_preds = joint_preds[:, :test_class * m] if m > 1 else joint_preds[:, :test_class]

#             if m > 1:
#                 agg_preds = 0
#                 for j in range(m):
#                     agg_preds = agg_preds + joint_preds[j::m, j::m] / m
#             else:
#                 agg_preds = joint_preds

#             logits = agg_preds.clone()  

#             # 1) top-2 & margin
#             top2_vals, top2_idx = torch.topk(logits, k=2, dim=1)  # [B,2]
#             z1, z2 = top2_vals[:, 0], top2_vals[:, 1]
#             i1, i2 = top2_idx[:, 0].long(), top2_idx[:, 1].long()
#             margin = z1 - z2


#             pred1_is_base = i1 < base_num
#             pred2_is_incr = i2 >= base_num
#             pred1_is_incr = ~pred1_is_base
#             pred2_is_base = ~pred2_is_incr


#             # 3) margin gate h(m)
#             small_margin = margin < tau
#             h = torch.clamp(tau - margin, min=0) / max(tau, 1e-6)

#             # 4) entropy
#             if use_entropy:
#                 z_base = logits[:, :base_num] / max(T_B, 1e-6) if entropy_mode == 'soft' else logits[:, :base_num]
#                 #z_base = logits[:, :test_class] / max(T_B, 1e-6) if entropy_mode == 'soft' else logits[:, :base_num]
#                 #z_base = logits[:, base_num:test_class] / max(T_B, 1e-6) if entropy_mode == 'soft' else logits[:, :base_num]
#                 p_base = torch.softmax(z_base, dim=1)
#                 eps = 1e-12
#                 H_base = -(p_base * (p_base.clamp_min(eps).log())).sum(dim=1) / math.log(base_num + eps)
#                 if entropy_mode == 'soft':
#                     w = torch.sigmoid(a * (H_base - t_th))  # [B] in (0,1)
#                     w_all.append(w.detach().cpu())
#                 else:
#                     incr_flag = (H_base >= t_hard).float()  # [B] {0,1}


#             marg_all.append(margin.detach().cpu())
#             maskA_all.append((pred1_is_base & pred2_is_incr & small_margin).detach().cpu())
#             maskB_all.append((pred1_is_incr & pred2_is_base & small_margin).detach().cpu())
#             is_base_pred_all.append(pred1_is_base.detach().cpu())
#             is_incr_pred_all.append(pred1_is_incr.detach().cpu())

#             # === Case A: pred1 ∈ base, pred2 ∈ incr, margin<τ
#             calibrated = logits.clone()
#             maskA = pred1_is_base & pred2_is_incr & small_margin
#             # w_th = float(getattr(args, 'adj_w_thresh', 0.0))  

#             # mask_hw =  (w >= w_th)
#             # maskA_hw = maskA & mask_hw
#             #maskA = pred1_is_incr & pred2_is_base & small_margin
#             #maskA = pred1_is_base & pred2_is_incr
#             if maskA.any():
#                 rowsA = torch.nonzero(maskA, as_tuple=False).squeeze(1)
#                 #rowsA = torch.nonzero(maskA_hw, as_tuple=False).squeeze(1)
#                 if use_entropy:
#                     if entropy_mode == 'soft':
#                         # deltaA = (lam_base * h[maskA_hw]).clamp(max=max_shift)  
#                         # calibrated[rowsA, i1[maskA_hw]] -= deltaA
#                         # calibrated[rowsA, i2[maskA_hw]] += deltaA
#                         deltaA = (lam_base * h[maskA] * w[maskA]).clamp(max=max_shift)
#                         #deltaA = (lam_base * w[maskA]).clamp(max=max_shift)
#                         #deltaA = (lam_base * h[maskA]).clamp(max=max_shift)
#                         calibrated[rowsA, i1[maskA]] -= deltaA
#                         calibrated[rowsA, i2[maskA]] += deltaA
#                        # calibrated[rowsA, i1[maskA]] += deltaA

#                     else:  # 'hard'
#                         hA = h[maskA]
#                         lam_incrA = float(getattr(args, 'adj_lambda_base_incr', lam_base))
#                         lam_baseA = float(getattr(args, 'adj_lambda_base_base', 0.00 if mode_h == 'one_sided' else 0.05))
#                         incr_flag_A = incr_flag[maskA]
#                         base_flag_A = 1.0 - incr_flag_A
#                         net_delta = (lam_incrA * hA * incr_flag_A - lam_baseA * hA * base_flag_A).clamp(min=-max_shift, max=max_shift)
#                         calibrated[rowsA, i1[maskA]] -= net_delta
#                         calibrated[rowsA, i2[maskA]] += net_delta
#                 else:
#                     deltaA = (lam_base * h[maskA]).clamp(max=max_shift)
#                     calibrated[rowsA, i1[maskA]] -= deltaA
#                     calibrated[rowsA, i2[maskA]] += deltaA

#             #=== Case B: pred1 ∈ incr, pred2 ∈ base, margin<τ
#             if enable_B and lam_incr > 0:
#                 maskB = pred1_is_incr & pred2_is_base & small_margin
#                 if maskB.any():
#                     rowsB = torch.nonzero(maskB, as_tuple=False).squeeze(1)
#                     if use_entropy and entropy_mode == 'soft':
#                         deltaB = (lam_incr * h[maskB] * (1.0 - w[maskB])).clamp(max=max_shift)
#                         calibrated[rowsB, i1[maskB]] -= deltaB  # top1(incr) ↓
#                         calibrated[rowsB, i2[maskB]] += deltaB  # top2(base) ↑
#                     else:
#                         deltaB = (lam_incr * h[maskB]).clamp(max=max_shift)
#                         calibrated[rowsB, i1[maskB]] -= deltaB
#                         calibrated[rowsB, i2[maskB]] += deltaB

#             # ---- 보정된 점수 사용 ----
#             logits = calibrated

#             # 평가
#             loss = F.cross_entropy(logits, test_label)
#             acc  = count_acc(logits, test_label)

#             vl.add(loss.item())
#             va.add(acc)

#             all_logits.append(logits.detach().cpu())
#             all_labels.append(test_label.detach().cpu())


#     def _cat_safe(x_list):
#         import numpy as np, torch
#         if len(x_list) == 0:
#             return np.empty((0,), dtype=float)
#         first = x_list[0]
#         if isinstance(first, torch.Tensor):
#             return torch.cat(x_list, dim=0).detach().cpu().numpy()
#         return np.concatenate(x_list, axis=0)

#     def _summarize(name, arr_np):
#         import numpy as np
#         arr_np = np.asarray(arr_np)
#         if arr_np.size == 0:
#             logging.info(f"[{session:02d}] w {name} | N=0 (skip)")
#             return
#         arr_np = arr_np[~np.isnan(arr_np)]
#         if arr_np.size == 0:
#             logging.info(f"[{session:02d}] w {name} | N=0 after NaN filter (skip)")
#             return
#         n = arr_np.size
#         mean = float(arr_np.mean()); std = float(arr_np.std())
#         amin = float(arr_np.min());  amax = float(arr_np.max())
#         probs = [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]
#         try:
#             q = np.quantile(arr_np, probs)
#         except Exception:
#             q = np.percentile(arr_np, [1,5,10,25,50,75,90,95,99])
#         logging.info(
#             f"[{session:02d}] w {name} | N={n} "
#             f"mean={mean:.4f}, std={std:.4f}, min={amin:.4f}, max={amax:.4f}, "
#             f"q01={q[0]:.4f}, q05={q[1]:.4f}, q10={q[2]:.4f}, q25={q[3]:.4f}, "
#             f"q50={q[4]:.4f}, q75={q[5]:.4f}, q90={q[6]:.4f}, q95={q[7]:.4f}, q99={q[8]:.4f}"
#         )

#     marg_arr       = _cat_safe(marg_all)
#     A_arr          = _cat_safe(maskA_all).astype(bool)
#     B_arr          = _cat_safe(maskB_all).astype(bool)
#     pred_base_arr  = _cat_safe(is_base_pred_all).astype(bool)
#     pred_incr_arr  = _cat_safe(is_incr_pred_all).astype(bool)

#     if use_entropy and entropy_mode == 'soft' and len(w_all) > 0:
#         w_arr = _cat_safe(w_all)

#         # 전체/조건부 요약
#         _summarize("ALL", w_arr)
#         if A_arr.any(): _summarize("maskA(pred1∈base, pred2∈incr, small_margin)", w_arr[A_arr])
#         if B_arr.any(): _summarize("maskB(pred1∈incr, pred2∈base, small_margin)", w_arr[B_arr])
#         if pred_base_arr.any(): _summarize("pred_base", w_arr[pred_base_arr])
#         if pred_incr_arr.any(): _summarize("pred_incr", w_arr[pred_incr_arr])

#         # 히스토그램
#         import numpy as np
#         if w_arr.size > 0:
#             bins = np.linspace(0.0, 1.0, 11)
#             hist_all, _ = np.histogram(w_arr, bins=bins)
#             logging.info(f"[{session:02d}] w hist(ALL, bins=0..1 step .1): {hist_all.tolist()}")
#             if A_arr.any():
#                 hist_A, _ = np.histogram(w_arr[A_arr], bins=bins)
#                 logging.info(f"[{session:02d}] w hist(maskA): {hist_A.tolist()}")

#         # 상관
#         if w_arr.size > 1 and marg_arr.size == w_arr.size:
#             import numpy as np
#             corr = np.corrcoef(w_arr, marg_arr)[0, 1]
#             logging.info(f"[{session:02d}] corr(w, margin) = {corr:.4f}")
#     else:
#         logging.info(f"[{session:02d}] entropy disabled / non-soft / no w collected; skip w-distribution analysis.")

#     vl = vl.item()
#     va = va.item()
#     print(f'epo {epoch}, test (after logit adj), loss={vl:.4f} acc={va:.4f}')

#     # --- seen / unseen accuracy 계산 ---
#     all_logits = torch.cat(all_logits, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)

#     if session > 0:
#         save_model_dir = os.path.join(args.save_path, f'session{session}_confusion_matrix_after ULA')
#         cm = confmatrix(all_logits, all_labels, save_model_dir)
#         per_class_acc = cm.diagonal()
#         seen_acc   = np.mean(per_class_acc[:args.base_class])
#         unseen_acc = np.mean(per_class_acc[args.base_class:])

#         # === FNR / FPR 추가 ===
#         fnr, fpr = compute_fnr_fpr_group(
#             all_logits, all_labels, base_num=args.base_class
#         )
#         logging.info(f"[Adj ] Session {session} ⇒ FNR={fnr:.2f}%, FPR={fpr:.2f}% (pos=base, neg=incr)")
#         # =======================

#         print(f"[After Logit Adjustment] Seen Acc: {seen_acc:.4f}, Unseen Acc: {unseen_acc:.4f}")
#         if result_list is not None:
#             result_list.append(
#                 f"[After Logit Adjustment] Session {session} ==> "
#                 f"Seen Acc: {seen_acc*100:.3f}, "
#                 f"Unseen Acc: {unseen_acc*100:.3f}, "
#                 f"Avg Acc: {va*100:.3f}, "
#                 f"Harmonic Mean: { (2*seen_acc*unseen_acc/(seen_acc+unseen_acc) if (seen_acc+unseen_acc)>0 else 0.0)*100:.3f}\n"
#             )
#         return vl, (seen_acc, unseen_acc, va)
#     else:
#         return vl, va



def compute_margin_entropy_per_class(model, testloader, transform, args, session, logger):
    """
    SAVC-style evaluation (multi-view transform) + margin & entropy computation

    계산 항목:
      - mean margin (top1 - top2)
      - 6가지 entropy (base/incr × base-only/incr-only/all-class)
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    import numpy as np

    eps = 1e-8
    model.eval()
    device = next(model.parameters()).device

    base_num = args.base_class
    test_class = args.base_class + session * args.way

    margin_list = []
    label_list  = []
    ent_all_list = []
    ent_base_cond_list = []
    ent_incr_cond_list = []

    logger.info(f"[Session {session}] Computing SAVC-style margin & entropy...")

    with torch.no_grad():
        for batch in tqdm(testloader):
            data, label = [_.to(device, non_blocking=True) for _ in batch]
            b = data.size(0)

            # SAVC multi-view transform
            data = transform(data)
            m = data.size(0) // b

            # Forward
            joint_preds = model(data)                         # (B*m, C)
            joint_preds = joint_preds[:, :test_class * m]     # truncate to valid classes

            # Aggregation across m views
            agg_preds = 0
            for j in range(m):
                agg_preds += joint_preds[j::m, j::m] / m
            logits = agg_preds                                 # (B, C)

            probs_all = F.softmax(logits, dim=-1)
            C = logits.shape[1]

            # --- margin ---
            top2_vals, _ = torch.topk(logits, k=2, dim=-1)
            margin = top2_vals[:, 0] - top2_vals[:, 1]      # (B,)

            # --- all-class entropy ---
            ent_all = -(probs_all * (probs_all + eps).log()).sum(dim=-1)

            # --- base-only (conditional) entropy ---
            if base_num > 0:
                p_base_mass = probs_all[:, :base_num].sum(dim=1, keepdim=True)
                probs_base_cond = probs_all[:, :base_num] / (p_base_mass + eps)
                ent_base_cond = -(probs_base_cond * (probs_base_cond + eps).log()).sum(dim=-1)
            else:
                ent_base_cond = torch.full_like(ent_all, float('nan'))

            # --- incr-only (conditional) entropy ---
            if base_num < C:
                p_incr_mass = probs_all[:, base_num:].sum(dim=1, keepdim=True)
                probs_incr_cond = probs_all[:, base_num:] / (p_incr_mass + eps)
                ent_incr_cond = -(probs_incr_cond * (probs_incr_cond + eps).log()).sum(dim=-1)
            else:
                ent_incr_cond = torch.full_like(ent_all, float('nan'))

            # Accumulate (CPU)
            margin_list.append(margin.cpu())
            ent_all_list.append(ent_all.cpu())
            ent_base_cond_list.append(ent_base_cond.cpu())
            ent_incr_cond_list.append(ent_incr_cond.cpu())
            label_list.append(label.cpu())

    # --- 결합 ---
    margins = torch.cat(margin_list)
    ent_all = torch.cat(ent_all_list)
    ent_base = torch.cat(ent_base_cond_list)
    ent_incr = torch.cat(ent_incr_cond_list)
    labels = torch.cat(label_list).long()

    base_mask = labels < base_num
    incr_mask = labels >= base_num

    def mean_mask(x, m):
        return x[m].mean().item() if m.any() else float('nan')

    # margin
    margin_base = mean_mask(margins, base_mask)
    margin_incr = mean_mask(margins, incr_mask)

    # --- entropy (6가지) ---
    entropy_baseSample_baseOnly = mean_mask(ent_base, base_mask)
    entropy_incrSample_baseOnly = mean_mask(ent_base, incr_mask)
    entropy_baseSample_incrOnly = mean_mask(ent_incr, base_mask)
    entropy_incrSample_incrOnly = mean_mask(ent_incr, incr_mask)
    entropy_baseSample_allClass = mean_mask(ent_all, base_mask)
    entropy_incrSample_allClass = mean_mask(ent_all, incr_mask)

    logger.info(f"[{session}] Margin → base:{margin_base:.6f} | incr:{margin_incr:.6f}")
    logger.info(f"[{session}] Entropy (all-class) → base:{entropy_baseSample_allClass:.6f} | incr:{entropy_incrSample_allClass:.6f}")
    logger.info(f"[{session}] Entropy (base-only, cond) → base:{entropy_baseSample_baseOnly:.6f} | incr:{entropy_incrSample_baseOnly:.6f}")
    logger.info(f"[{session}] Entropy (incr-only, cond) → base:{entropy_baseSample_incrOnly:.6f} | incr:{entropy_incrSample_incrOnly:.6f}")

    return {
        "margin_base": margin_base,
        "margin_incr": margin_incr,
        "entropy_baseSample_baseOnly": entropy_baseSample_baseOnly,
        "entropy_incrSample_baseOnly": entropy_incrSample_baseOnly,
        "entropy_baseSample_allClass":  entropy_baseSample_allClass,
        "entropy_incrSample_allClass":  entropy_incrSample_allClass,
        "entropy_baseSample_incrOnly":  entropy_baseSample_incrOnly,
        "entropy_incrSample_incrOnly":  entropy_incrSample_incrOnly,
    }


@torch.no_grad()
def plot_margin_by_group(
    model, testloader, base_num, session,
    save_dir, bins=40
):
    """
    한 번의 forward로 margin 분포만 시각화.
    - GT=Base vs GT=Incr margin 분포 비교
    저장: {save_dir}/S{session}_margin_only.png/pdf
    """
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    # ⚠️ 기존 코드와 동일한 방식 유지 (위험하지만 그대로 둠)
    test_class = base_num + session * getattr(
        model, 'args', getattr(testloader.dataset, 'args', None)
    ).way

    all_margins, all_labels = [], []

    for data, label in testloader:
        data  = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logits = model(data)[:, :test_class]

        # margin = top1 - top2
        top2_vals, _ = torch.topk(logits, k=2, dim=1)
        margin = top2_vals[:, 0] - top2_vals[:, 1]

        all_margins.append(margin.detach().cpu())
        all_labels.append(label.detach().cpu())

    margins = torch.cat(all_margins).numpy()
    labels  = torch.cat(all_labels).long().numpy()

    base_mask = labels < base_num
    incr_mask = labels >= base_num

    m_base = margins[base_mask]
    m_incr = margins[incr_mask]

    # ===== Plot (single panel) =====
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)

    ax.hist(m_base, bins=bins, alpha=0.6, density=True, label="Base")
    ax.hist(m_incr, bins=bins, alpha=0.6, density=True, label="Incr")

    ax.set_xlabel("Margin")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.legend()

    png_path = os.path.join(save_dir, f"S{session}_margin_only.png")
    pdf_path = os.path.join(save_dir, f"S{session}_margin_only.pdf")

    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()

    return {
        "margins_base": m_base,
        "margins_incr": m_incr,
        "png": png_path,
        "pdf": pdf_path,
    }