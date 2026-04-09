import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import resample

from .base import BaseTrainer
from .hooks.base_hook import HookContainer
from utils.stat_tracker import RuntimeTracker
from utils.timer import Timer
import utils.eval_meters as eval_meters
import optim as optim
import types
from collections import Counter
from transformers.models.bert.tokenization_bert import BertTokenizer


def seqcls_batch_to_device(batched):
    uids = batched[0]
    input_ids, golds, attention_mask, token_type_ids = map(
        lambda x: x.cuda(), batched[1:]
    )
    return (
        uids,
        golds,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }, 
        token_type_ids
    )

def bootstrap_ci(preds, golds, n_bootstrap=1000, ci=95):
    preds = preds.detach().cpu().numpy() 
    golds = golds.detach().cpu().numpy() 
    scores = []
    for _ in range(n_bootstrap):
        idx = resample(range(len(preds)), replace=True)
        p = [preds[i] for i in idx]
        g = [golds[i] for i in idx]
        scores.append(sum(pi == gi for pi, gi in zip(p, g)) / len(p))
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return lower, upper

class BertFinetuner(BaseTrainer):
    def __init__(self, conf, logger, data_iter):
        super(BertFinetuner, self).__init__(conf, logger)
        self.trn_dl, self.val_dl, self.tst_dl = None, None, None
        self.task_metrics = data_iter.metrics
        self.batch_to_device = seqcls_batch_to_device
        self._wrap_datasplits(data_iter)

        # logging tools.
        self.tracker = RuntimeTracker(metrics_to_track=[])
        self.timer = Timer(
            verbosity_level=1 if conf.track_time else 0,
            log_fn=logger.log_metric,
            on_cuda=True,
        )
        self.model_ptl = conf.ptl
        self.results = {}
        self.bounds = {}

    def train(self, model, masker, hooks=None):
        # init the model for the training.
        opt, model = self._init_training(model)
        self.model = model  # set the model for hooks
        results = []
        self.model.train()

        # init the masker for the training.
        self.masker = masker  # set the masker for hooks
        self.masker_scheduler = masker.masker_scheduler if masker is not None else None
        self._init_sparsity_control()


        # init the hook for the training.
        hook_container = HookContainer(world_env={"trainer": self}, hooks=hooks)

        num_batch = len(self.trn_dl)
        self.log_fn(f"[INFO]: start training for task: {self.conf.task}")
        hook_container.on_train_begin()
        for epoch in range(1, self.conf.num_epochs + 1):
            for batched in self.trn_dl:
                self._batch_step += 1
                self._epoch_step = self._batch_step / num_batch

                with self.timer("validation", epoch=self._epoch_step):
                    if self._batch_step % self.conf.eval_every_batch == 0:
                        eval_res = self.evaluate()
                        hook_container.on_validation_end(eval_res=eval_res)

                with self.timer("load_data", epoch=self._epoch_step):
                    uids, golds, inputs, _ = seqcls_batch_to_device(batched)

                # forward for "pretrained model+classifier".
                with self.timer("forward_pass", epoch=self._epoch_step):
                    if self.conf.model_scheme == "postagging":
                        logits, bert_out, *_ = self._model_forward(**inputs)
                    else:
                        output = self._model_forward(**inputs)
                        logits = output.logits
                    # logits, *_ = self._model_forward(**inputs)
                    # the cross entropy by default uses reduction='mean'
                    loss = self.criterion(logits, golds)

                # backward for "pretrained model+classifier".
                with self.timer("backward_pass", epoch=self._epoch_step):
                    self.tracker.update_metrics(
                        metric_stat=[loss.item()], n_samples=len(logits)
                    )
                    loss.backward()

                # try to control the mask sparsity via the lagrangian loss.
                with self.timer("control_sparsity_backward", epoch=self._epoch_step):
                    current_sparsity = (
                        self.masker_scheduler.get_sparsity_over_whole_model(
                            self.model, self.masker
                        )
                        if self.masker_scheduler is not None
                        else torch.tensor(0)
                    )
                    if (
                        self.masker_scheduler is not None
                        and self.masker_scheduler.is_skip is False
                    ):
                        _, target_sparsity, _ = self.masker_scheduler.step(
                            cur_epoch=self._epoch_step
                        )
                        lagrangian_loss = self.lambda_ * (
                            (target_sparsity - current_sparsity) ** 2
                        )
                        lagrangian_loss.backward()

                with self.timer("perform_update", epoch=self._epoch_step):
                    opt.step()
                    if (
                        self.masker_scheduler is not None
                        and self.masker_scheduler.is_skip is False
                    ):
                        self.sparsity_optimizer.step()
                        self.sparsity_optimizer.zero_grad()
                    opt.zero_grad()

                # logging.
                self.log_fn_json(
                    name="training",
                    values={
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "step": self._batch_step,
                        "epoch": self._epoch_step,
                        "loss": self.tracker.stat["loss"].val,
                        "avg_loss": self.tracker.stat["loss"].avg,
                        "current_sparsity": current_sparsity.item(),
                        "target_sparsity": target_sparsity
                        if self.masker_scheduler is not None
                        and self.masker_scheduler.is_skip is False
                        else -1,
                    },
                    tags={"split": "train"},
                    display=True,
                )
                hook_container.on_batch_end()

                # early stopping.
                self._best_epoch_step = hook_container.hooks[0].best_step / num_batch
                if (
                    self.conf.early_stop is not None
                    and self._epoch_step - self._best_epoch_step > self.conf.early_stop
                ):
                    self.log_fn(
                        f"Early-stopping: current epoch={self._epoch_step},"
                        f"best_epoch={self._best_epoch_step}."
                    )
                    self.logger.save_json()
                    hook_container.on_train_end()
                    # testing once on model end training
                    #self.evaluate(eval_name="tst_dl")
                    return

                # display the timer info.
                if (
                    self.conf.track_time
                    and self._batch_step % self.conf.summary_freq == 0
                ):
                    print(self.timer.summary())

            self._epoch_step += 1
            self.tracker.reset()
            self.logger.save_json()
            torch.cuda.empty_cache()
        hook_container.on_train_end()
        # testing once on model end training
        #self.evaluate(eval_name="tst_dl")
        return results

    def evaluate(self, eval_name = "val_dl"):
        self.model.eval()
        eval_res = {}
        message = ""
        with torch.no_grad():
                eval_dl = getattr(self, eval_name)
                if not eval_dl:
                    message += f" skip evaluation on {eval_name}."
                    eval_res[eval_name] = None
                message += f" finished evaluation on {eval_name}."
                all_losses, all_golds, all_preds = [], [], []
                all_bounds = []
                all_golds_ner, all_preds_ner = [], []
                for batched in eval_dl:
                    # golds is used for compute loss, _golds used for i2t convertion
                    uids, golds, batched, _golds = self.batch_to_device(batched)
                    with torch.no_grad():
                        if self.conf.model_scheme == "postagging":
                            logits, bert_out, *_ = self._model_forward(**batched)
                        else:
                            output = self._model_forward(**batched)
                            logits = output.logits
                        loss = self.criterion(logits, golds).mean().item()
                        preds = torch.argmax(logits, dim=-1, keepdim=False)
                        all_losses.append(loss)
                        all_preds.extend(preds.detach().cpu().numpy())
                        all_golds.extend(golds.detach().cpu().numpy())

                # returning upper, lower bounds for the predictions due to smaller datasets
                all_preds_tensor = torch.tensor(all_preds)
                all_golds_tensor = torch.tensor(all_golds)
                lower, upper = bootstrap_ci(all_preds_tensor, all_golds_tensor)
                    
                eval_res[eval_name] = {}
                for task_metric in self.task_metrics:
                    eval_fn = getattr(eval_meters, task_metric)
                    if len(all_golds_ner) == 0:
                        eval_res[eval_name][task_metric] = eval_fn(all_preds, all_golds)
                        self.log_fn(
                            f"[INFO]: gold distribution on {eval_name}: {Counter(all_golds)}"
                        )
                    else:
                        eval_res[eval_name][task_metric] = eval_fn(
                            all_preds_ner, all_golds_ner
                        )
                
                eval_res[eval_name]["ci_lower"] = lower
                eval_res[eval_name]["ci_upper"] = upper

                # logging.
                self.log_fn(f"[INFO] Finished evaluation: {message}")
                self.log_fn_json(
                    name="evaluation",
                    values={
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "step": self._batch_step,
                        "epoch": self._epoch_step,
                        "loss": loss,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        **eval_res[eval_name],
                    },
                    tags={"split": eval_name},
                    display=True,
                )
                self.log_fn(
                    f"[INFO] Eval results on {eval_name} @ batch_step {self._batch_step}, "
                    f"avg loss: {np.mean(all_losses)}."
                )
                for m_name, m_score in eval_res[eval_name].items():
                    self.log_fn(
                        f"[INFO] Eval results on {eval_name} @ batch_step {self._batch_step},"
                        f"{m_name}: {m_score:.4f}."
                    )
        # save results, bounds in json to be called later        
        self.results[f"results_{self.batch_step}"] = eval_res
        self.bounds[f"bounds_{self._batch_step}"] = {
                split: {"ci_lower": eval_res[split]["ci_lower"], 
                        "ci_upper": eval_res[split]["ci_upper"]}
                for split in ("val_dl", "tst_dl") 
                if eval_res.get(split) is not None
            }
        self.log_fn(
                f"[INFO] 95% CI @ batch_step {self._batch_step}: "
                f"val [{eval_res.get('val_dl', {}).get('ci_lower', 'N/A'):.3f}, "
                f"{eval_res.get('val_dl', {}).get('ci_upper', 'N/A'):.3f}]"
            )
        if eval_name == "val_dl":
            self.model.train()
        

    def _model_forward(self, **kwargs):
        if (self.model_ptl == "roberta" or self.model_ptl == "distilbert") and "token_type_ids" in kwargs:
            kwargs.pop("token_type_ids")
        output = self.model(**kwargs)
        return output

    def _init_training(self, model):
        model = self._parallel_to_device(model)

        # split params into encoder and classifier groups
        encoder_lr = getattr(self.conf, 'lr_encoder', self.conf.lr)
        
        params = [
            {
                "params": [value],
                "name": key,
                "weight_decay": self.conf.weight_decay,
                "param_size": value.size(),
                "nelement": value.nelement(),
                "lr": self.conf.lr_for_mask
                    if self.conf.lr_for_mask is not None and "mask" in key
                    else self.conf.lr  # classifier lr
                    if "classifier" in key
                    else encoder_lr,  # encoder lr
            }
            for key, value in model.named_parameters()
            if value.requires_grad
        ]

        # create the optimizer.
        if self.conf.optimizer == "adam":
            opt = optim.Adam(
                params,
                lr=self.conf.lr,
                betas=(self.conf.adam_beta_1, self.conf.adam_beta_2),
                eps=self.conf.adam_eps,
                weight_decay=self.conf.weight_decay,
            )
        elif self.conf.optimizer == "sgd":
            opt = torch.optim.SGD(
                params,
                lr=self.conf.lr,
                momentum=self.conf.momentum_factor,
                weight_decay=self.conf.weight_decay,
                nesterov=self.conf.use_nesterov,
            )
        elif self.conf.optimizer == "signsgd":
            opt = optim.SignSGD(
                params,
                lr=self.conf.lr,
                momentum=self.conf.momentum_factor,
                weight_decay=self.conf.weight_decay,
                nesterov=self.conf.use_nesterov,
            )
        else:
            raise NotImplementedError("this optimizer is not supported yet.")
        opt.zero_grad()
        model.zero_grad()
        self.log_fn(f"Initialize the optimizer: {self.conf.optimizer}")
        return opt, model

    def _init_sparsity_control(self):
        if self.masker_scheduler is not None and self.masker_scheduler.is_skip is False:
            self.log_fn(f"Initialize to control the sparsity.")
            # init the parameters.
            self.lambda_ = nn.Parameter(torch.tensor(0.0).cuda())

            # init the optimizer for the lambda2.
            # if we also include lambda1 then the optimization becomes unstable.
            self.sparsity_optimizer = torch.optim.Adam([self.lambda_], weight_decay=0)
            self.sparsity_optimizer.param_groups[0]["lr"] = (
                -1.0
                if "lambdas_lr" not in self.conf.masking_scheduler_conf_
                else -self.conf.masking_scheduler_conf_["lambdas_lr"]
            )
    