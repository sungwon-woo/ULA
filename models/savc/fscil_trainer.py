from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *
from losses import SupContrastive
from augmentations import fantasy
import os
from datetime import datetime
import logging

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        
        if args.fantasy is not None:
            self.transform, self.num_trans = fantasy.__dict__[args.fantasy]()
        else:
            self.transform = None
            self.num_trans = 0

        self.model = self.model = MYNET(self.args, mode=self.args.base_mode, trans=self.num_trans)
#         self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader
        
    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]
        all_acc = []
        all_acc_calib = []

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            self.model.load_state_dict(self.best_model_dict)

            if session == 0:
                logging.info("Skipping base training (session 0). Using pretrained model from --model_dir.")
                assert args.model_dir is not None, "Provide --model_dir (checkpoint with {'params': ...})."
                vl, va = test(self.model, testloader, 0, self.transform, args, session)
                all_acc.append([va * 100.0])
                all_acc_calib.append([va * 100.0])

                ratio = int(args.base_class / args.way)
                g_acc0, auc0 = get_gacc(ratio, all_acc)
                logging.info(f"Session 0 ⇒ average Acc = {va:.4f}")
                logging.info(f"Session 0 ⇒ Generalized AUC = {auc0:.4f}")
                result_list.append(f"Session {session} ⇒ average Acc = {va:.4f}")
                result_list.append(f"Session {session} ⇒ Generalized AUC = {auc0:.4f}")
                continue

            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.mode = self.args.new_mode
                self.model.eval()
                train_transform = trainloader.dataset.transform
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.update_fc(trainloader, np.unique(train_set.targets), self.transform, session)
                if args.incft:
                    trainloader.dataset.transform = train_transform
                    train_set.multi_train = True
                    update_fc_ft(trainloader, self.transform, self.model, self.num_trans, session, args) 

                tsl, (seenac, unseenac, tsa) = test(self.model, testloader, 0, self.transform, args, session)

                all_acc.append([seenac * 100, unseenac * 100, tsa * 100])
                alpha, gacc_per_session, auc_per_session = compute_gacc_session_style(
                    all_acc, base_num=60, incr_num=5, alpha_points=12
                )
                gacc_last = gacc_per_session[-1]
                auc  = auc_per_session[-1]
                logging.info(f"Session {session} ⇒ Generalized AUC = {auc:.4f}")
                result_list.append(f"Session {session} ⇒ Generalized AUC = {auc:.4f}")

                if 'seen_acc' not in self.trlog:
                    self.trlog['seen_acc'] = []
                if 'unseen_acc' not in self.trlog:
                    self.trlog['unseen_acc'] = []
                self.trlog['seen_acc'].append(float('%.3f' % (seenac * 100)))
                self.trlog['unseen_acc'].append(float('%.3f' % (unseenac * 100)))
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                self.best_model_dict = deepcopy(self.model.state_dict())

                base_acc = seenac * 100.0
                incr_acc = unseenac * 100.0
                avg_acc  = tsa * 100.0
                harm_acc = (2 * base_acc * incr_acc / (base_acc + incr_acc)) if (base_acc + incr_acc) > 0 else 0.0

                logging.info('Saving model to :%s' % save_model_dir)
                logging.info('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))
                logging.info(f"Session {session} ==> Seen Acc:{self.trlog['seen_acc'][-1]} , "
                    f"Unseen Acc:{self.trlog['unseen_acc'][-1]} Avg Acc:{self.trlog['max_acc'][session]}, "
                    f"Harmonic Mean: {harm_acc:.3f}")

                result_list.append(
                    f"Session {session} ==> "
                    f"Seen Acc: {self.trlog['seen_acc'][-1]:.3f}, "
                    f"Unseen Acc: {self.trlog['unseen_acc'][-1]:.3f}, "
                    f"Avg Acc: {self.trlog['max_acc'][session]:.3f}, "
                    f"Harmonic Mean: {harm_acc:.3f}\n"
                )


                tsl, (seenac, unseenac, avgac) = adaptive_logit_adjust(
                    self.model, testloader, 0, self.transform, args, session, result_list=result_list,
                )

                all_acc_calib.append([seenac * 100, unseenac * 100, tsa * 100])
                alpha, gacc_per_session, auc_per_session = compute_gacc_session_style(
                    all_acc_calib, base_num=60, incr_num=5, alpha_points=12
                )
                gacc_last = gacc_per_session[-1]
                auc  = auc_per_session[-1]
                logging.info(f"Session {session} ⇒ C. Generalized AUC = {auc:.4f}")
                result_list.append(f"Session {session} ⇒ C. Generalized AUC = {auc:.4f}")

                base_acc = seenac * 100.0
                incr_acc = unseenac * 100.0
                avg_acc  = avgac * 100.0
                harm_acc = (2 * base_acc * incr_acc / (base_acc + incr_acc)) if (base_acc + incr_acc) > 0 else 0.0
                self.trlog['seen_acc'].append(float('%.3f' % (seenac * 100)))
                self.trlog['unseen_acc'].append(float('%.3f' % (unseenac * 100)))
                self.trlog['max_acc'][session] = float('%.3f' % (avgac * 100))
                logging.info('[After Logit Adjustment]  test acc={:.3f}'.format(self.trlog['max_acc'][session]))
                logging.info(f"Session {session} ==> Seen Acc:{self.trlog['seen_acc'][-1]} , "
                    f"Unseen Acc:{self.trlog['unseen_acc'][-1]} Avg Acc:{self.trlog['max_acc'][session]}, "
                    f"Harmonic Mean: {harm_acc:.3f}")

                result_list.append(
                    f"[After Logit Adjustment] Session {session} ==> "
                    f"Seen Acc: {self.trlog['seen_acc'][-1]:.3f}, "
                    f"Unseen Acc: {self.trlog['unseen_acc'][-1]:.3f}, "
                    f"Avg Acc: {self.trlog['max_acc'][session]:.3f}, "
                    f"Harmonic Mean: {harm_acc:.3f}\n"
                )

                print(f"{avg_acc:.3f}")
                print(f"{base_acc:.3f}")
                print(f"{incr_acc:.3f}")
                print(f"{auc:.3f}")
                print(f"{harm_acc:.3f}")


        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        # logging.info(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        # logging.info('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        # logging.info('Total time used %.2f mins' % total_time)
        
    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Cosine-Epo_%d-Lr_%.4f' % (
                self.args.epochs_base, self.args.lr_base)
            
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)
        self.args.save_path = self.args.save_path + f'-fantasy_{self.args.fantasy}'
        self.args.save_path = self.args.save_path + '-alpha_%.2f-beta_%.2f' % (self.args.alpha, self.args.beta)
        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.args.save_path = os.path.join('checkpoint', f"{self.args.save_path}_{current_time}")

        ensure_path(self.args.save_path)
        return None