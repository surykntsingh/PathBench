import os
from abc import abstractmethod
from datetime import datetime

import time
import torch
import pandas as pd
from numpy import inf
from tqdm import tqdm
import torch.distributed as dist

from utils.utils import write_json_file


class BaseTrainer(object):
    def __init__(self, args, model, criterion, metric_ftns, optimizer=None, lr_scheduler=None, train_dataloader=None, val_dataloader=None,
                 test_dataloader=None):

        self.args = args

        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.model = model

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.max_epochs
        self.epochs_val = self.args.epochs_val
        self.start_val = self.args.start_val
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric

        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = args.save_dir
        # if args.resume is not None:
        #     self._resume_checkpoint(args.resume)

        self.best_recorder = {self.mnt_metric: self.mnt_best}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self, rank):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            print("epoch: ", epoch)
            self.train_dataloader.sampler.set_epoch(epoch)
            result = self._train_epoch(rank)
            best = False

            if epoch % self.epochs_val == 0 and epoch > self.start_val:
                val_result = self._val_epoch(rank, result)
                # save logged informations into log dict
                log = {'epoch': epoch}
                log.update(val_result)
                self._record_best(log)

                # print logged informations to the screen
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best

                if self.mnt_mode != 'off':
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1
                        best = False

                    if not_improved_count > self.early_stop:
                        print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                            self.early_stop))
                        break

            if epoch % self.save_period == 0 and rank == 0:
                self._save_checkpoint(epoch, save_best=best)
        if rank == 0:
            self._print_best()
            self._print_best_to_file('val')

    def test(self, rank):
        log, predictions = self._test_epoch(rank,{})

        if rank == 0:
            self._print_best()
            self._print_best_to_file('test')

        print('Results in test set')
        for key, value in log.items():
            print('\t{:15s}: {}'.format(str(key), value))

        self._save_predictions(predictions)

    def _print_best_to_file(self, stage):

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)

        date = datetime.now()
        self.best_recorder['date'] = date.strftime("%Y-%m-%d %H:%M:%S")
        # self.best_recorder['seed'] = self.args.seed
        metrics_df = pd.DataFrame([self.best_recorder])
        metrics_df.to_csv(f'{self.args.record_dir}/results_{stage}.csv', mode='a')

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _save_predictions(self, predictions):
        os.makedirs(self.args.record_dir, exist_ok=True)
        results_path = f'{self.args.record_dir}/predictions.json'
        write_json_file(predictions, results_path)

    def _record_best(self, log):

        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder[
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder[self.mnt_metric])
        if improved_val:
            self.best_recorder.update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder.items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, args, model, criterion, metric_ftns, optimizer, lr_scheduler, train_dataloader=None, val_dataloader=None,
                 test_dataloader=None):
        super(Trainer, self).__init__(args, model, criterion, metric_ftns, optimizer, lr_scheduler, train_dataloader,
                                      val_dataloader,
                                      test_dataloader)

    def _train_epoch(self, rank):
        dist.barrier()
        train_loss = 0
        # print(f'train_loss: {train_loss}')
        self.model.train()
        for batch_idx, (slide_ids, features, reports_ids, reports_masks) in enumerate(
                tqdm(self.train_dataloader, desc='Training')):
            # print(f'features: {features.shape}')
            reports_ids, reports_masks = reports_ids.cuda(), reports_masks.cuda()
            output, weights = self.model(features, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks, weights, self.args.g_lambda)
            train_loss += loss.item()
            # print(train_loss)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        avg_train_loss = train_loss / len(self.train_dataloader)
        avg_train_loss = self._sync_scalar(avg_train_loss, rank)
        log = {'train_loss': avg_train_loss}
        print('train_loss: ', avg_train_loss)
        self.lr_scheduler.step(avg_train_loss)

        return log

    def _sync_scalar(self, value, rank):
        value_tensor = torch.tensor(value, device=f'cuda:{rank}', dtype=torch.float32)
        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        value_tensor /= dist.get_world_size()
        return value_tensor.item()

    def _val_epoch(self, rank, log):
        dist.barrier()
        self.model.eval()
        with torch.no_grad():
            val_gts_ids, val_res_ids = [], []
            for batch_idx, (slide_ids, features, reports_ids, reports_masks) in enumerate(
                    tqdm(self.val_dataloader, desc='Validating')):
                reports_ids, reports_masks = reports_ids.cuda(), reports_masks.cuda()
                # print('start eval...')
                # print("repo: ",reports_ids.shape)
                output, _ = self.model(features, mode='sample')

                val_res_ids.append(output)  # predict
                val_gts_ids.append(reports_ids)  # ground truth

            val_res_ids = distributed_concat(torch.cat(val_res_ids, dim=0),
                                             len(self.val_dataloader.dataset)).cpu().numpy()
            val_gts_ids = distributed_concat(torch.cat(val_gts_ids, dim=0),
                                             len(self.val_dataloader.dataset)).cpu().numpy()

            val_gts, val_res = self.model.module.tokenizer.decode_batch(
                val_gts_ids[:, 1:]), self.model.module.tokenizer.decode_batch(val_res_ids)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            print("val_met: ", val_met)
        return log

    def _test_epoch(self, rank, log):
        dist.barrier()
        self.model.eval()
        with torch.no_grad():
            slides, test_gts_ids, test_res_ids = [], [], []
            for batch_idx, (slide_ids, features, reports_ids, reports_masks) in enumerate(
                    tqdm(self.test_dataloader, desc='Testing')):
                reports_ids, reports_masks = reports_ids.cuda(), reports_masks.cuda()
                output, _ = self.model(features, mode='sample')

                test_res_ids.append(output)  # predict
                test_gts_ids.append(reports_ids)  # ground truth
                slides.append(slide_ids)
            test_res_ids = distributed_concat(torch.cat(test_res_ids, dim=0),
                                              len(self.test_dataloader.dataset)).cpu().numpy()
            test_gts_ids = distributed_concat(torch.cat(test_gts_ids, dim=0),
                                              len(self.test_dataloader.dataset)).cpu().numpy()

            test_gts, test_res = self.model.module.tokenizer.decode_batch(
                test_gts_ids[:, 1:]), self.model.module.tokenizer.decode_batch(test_res_ids)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print("test_met: ", test_met)

            predictions = []
            for i,slide_id in enumerate(slides):
                predictions.append({
                    'slide_id': slide_id,
                    'prediction': test_res[i],
                    'ground_truth': test_gts[i]
                })

        return log, predictions


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]
