import os 
import sys
import argparse
import time 
import json
import datetime
import torch

# import src.misc.dist as dist 
from src.core import YAMLConfig
from src.misc import dist
from src.solver.det_solver import DetSolver 
from src.solver.det_engine import *
from src.solver import TASKS
from src.data import get_coco_api_from_dataset
from argparse import Namespace

class CustomedDetSolver(DetSolver):
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            # if self.output_dir:
            #     checkpoint_paths = [self.output_dir / 'checkpoint.pth']
            #     # extra checkpoint before LR drop and every 100 epochs
            #     if (epoch + 1) % args.checkpoint_step == 0:
            #         checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
            #     for checkpoint_path in checkpoint_paths:
            #         dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # TODO 
            for k in test_stats.keys():
                checkpoint_path = self.output_dir / f'last.pth'
                best_path = self.output_dir / f'best.pth'
                dist.save_on_master(self.state_dict(epoch), checkpoint_path)
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    # best_stat[k] = max(best_stat[k], test_stats[k][0])
                    if best_stat[k] < test_stats[k][0]:
                        best_stat[k] = test_stats[k][0]
                        if self.output_dir:
                            dist.save_on_master(self.state_dict(epoch), best_path)
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
                    if self.output_dir:
                        dist.save_on_master(self.state_dict(epoch), best_path)
            print('best_stat: ', best_stat)


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    
    def val_with_query(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate_with_query(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
    
if __name__ == '__main__':
    # download pretrained model: https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth
    args = Namespace(
                    config='configs/rtdetr/rtdetr_r18vd_6x_coco.yml', 
                    # config='configs/rtdetr/rtdetr_r18vd_6x_coco.yml', 
                    resume=None, 
                    tuning='rtdetr_r18vd_dec3_6x_coco_from_paddle.pth', 
                    test_only=False, 
                    amp=True, 
                    seed=None)
    # dist.init_distributed()
    # if args.seed is not None:
    #     dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )
    cfg.epoches = 300
    # print(list(vars(cfg).keys()))
    solver = CustomedDetSolver(cfg)
    solver.train()

    # for batch in solver.train_dataloader:
    #     input = batch[0].to(solver.device)
    #     targets = [{k: v.to(solver.device) for k, v in t.items()} for t in batch[1]]
    #     break    
    # out = solver.model(input, targets)

    if args.test_only:
        solver.val_with_query()
    else:
        solver.fit()