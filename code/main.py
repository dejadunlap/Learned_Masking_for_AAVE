# -*- coding: utf-8 -*-
import os
import copy
import torch
from datetime import datetime

from parameters import get_args
from trainers.hooks import EvaluationRecorder, SparsityRecorder
from trainers.finetuner import BertFinetuner
import configs.task_configs as task_configs
import csv
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.tokenization_bert import BertTokenizer

import predictors.linear_predictors as linear_predictors
import predictors.random_reinit as random_reinit
import masking.maskers as maskers
import masking.sparsity_control as sp_control
import utils.checkpoint as checkpoint
import utils.logging as logging
import utils.param_parser as param_parser

config = dict(
    ptl="bert",
    model="bert-base-uncased",
    task="aave_mask",
    model_scheme="vector_cls_sentence",
    experiment="debug",
    max_seq_len=128,
    lr=1e-3,
    world="0",
    batch_size=32,
    eval_every_batch=60,
    num_epochs=10,
    do_BL=False,
    do_MS=True,
    ptl_req_grad=False,
    classifier_req_grad=False,
    mask_classifier=True,
    mask_ptl=True,
    layers_to_mask="5,6,7,8",
    train_fast=True,
    num_snapshots=10,
    masking_scheduler_conf="lambdas_lr=0,sparsity_warmup=automated_gradual_sparsity,final_sparsity=0.05,sparsity_warmup_interval_epoch=0.1,init_epoch=0,final_epoch=1,initial_sparsity=0.8",
    checkpoint="./data/cached/",
)


def init_task(conf):
    tokenizer = BertTokenizer.from_pretrained(conf.model)
    data_iter = task_configs.task2dataiter[conf.task](conf.task, conf.model, tokenizer, conf.max_seq_len)
    conf.logger.log(f"Creating and loading pretrained {conf.ptl.upper()} model.")
    
    model = BertForSequenceClassification.from_pretrained(
        conf.model,
        num_labels=2,
        cache_dir=conf.pretrained_weight_path,
    )

    # we can have the choice to randomly (and partially) initialize the ptl.
    random_reinit.random_init_ptl(conf, model)
    return model, data_iter


def confirm_experiment(conf, model):
    # we will first put the model on the device (to reduce the initialization time).
    model = model.cuda()

    # this fn does not override conf so the path etc should be fine
    for name, param in model.named_parameters():
        param.requires_grad = False

    if conf.do_BL:
        assert (not conf.do_MS) and (not conf.mask_classifier)
        for param_name, param in model.named_parameters():
            if conf.ptl in param_name and conf.ptl_req_grad:
                param.requires_grad = True
            # note, roberta has one extra linear layer in the ``classifier'' namespace
            if "classifier" in param_name and conf.classifier_req_grad:
                param.requires_grad = True
            conf.logger.log("{} -> {}".format(param_name, param.requires_grad))
        return (model, None)

    elif conf.do_MS:
        assert (not conf.do_BL) and (not conf.ptl_req_grad)
        # for do_masking, the top classification layer is either
        # (i) masked without explicit training (ii) explict training
        if conf.classifier_req_grad:
            for _, param in model.classifier.named_parameters():
                param.requires_grad = True
            assert not conf.mask_classifier
        # else:
        # assert conf.mask_classifier
        masker = init_masker(conf, model)
        return (model, masker)
    raise ValueError("one of do_BL, do_MS must be true!")


def init_masker(conf, model):
    # init the masker scheduler.
    masker_scheduler = sp_control.MaskerScheduler(conf)

    # init the masker.
    masker = maskers.Masker(
        masker_scheduler=masker_scheduler,
        log_fn=conf.logger.log,
        mask_biases=conf.mask_biases,
        structured_masking_info={
            "structured_masking": conf.structured_masking,
            "structured_masking_types": conf.structured_masking_types_,
            "force_masking": conf.force_masking,
        },
        threshold=conf.threshold,
        init_scale=conf.init_scale,
        which_ptl=conf.ptl,
        controlled_init=conf.controlled_init,
    )

    # assuming mask all stuff in one transformer block, absorb bert.pooler directly
    weight_types = ["K", "Q", "V", "AO", "I", "O", "P"]

    # parse the get the names of layers to be masked.
    names_tobe_masked = set()
    if conf.mask_ptl:
        names_tobe_masked = maskers.chain_module_names(
            conf.ptl, conf.layers_to_mask_, weight_types
        )
    if conf.mask_classifier:
        if conf.ptl == "bert" or conf.ptl == "distilbert":
            names_tobe_masked.add("classifier")
        elif conf.ptl == "roberta":
            if (
                conf.model_scheme == "postagging"
                or conf.model_scheme == "multiplechoice"
            ):
                names_tobe_masked.add("classifier")
            elif conf.model_scheme == "vector_cls_sentence":
                names_tobe_masked.add("classifier.dense")
                names_tobe_masked.add("classifier.out_proj")

    # patch modules.
    masker.patch_modules(
        model=model,
        names_tobe_masked=names_tobe_masked,
        name_of_masker=conf.name_of_masker,
    )
    return masker


def init_recorders(conf, masker):
    state_recorder = EvaluationRecorder(
        init_state_where_=os.path.join(conf.checkpoint_root, "init_state"),
        where_=os.path.join(conf.checkpoint_root, "best_state"),
        which_metric=["accuracy"],
    )

    if masker is not None:
        sparsity_recorder = SparsityRecorder(
            where_=f"{os.path.join(conf.checkpoint_root, f'accuracy_sparsities')}",
            init_masks=masker.init_masks,
        )
        return [state_recorder, sparsity_recorder]
    return [state_recorder]


def finetune_on_fixed_masks(conf, model, masker, data_iter):
    _conf = copy.deepcopy(conf)
    _model = copy.deepcopy(model)
    assert _conf.do_MS

    # turn off the gradients for masks.
    for name, param in _model.named_parameters():
        if "mask" in name:
            param.requires_grad = False

    # remove the masks for classifier.
    if "classifier_wo_mask" in _conf.do_tuning_on_MS_scheme_:
        _model.classifier = maskers.MaskedLinear0(
            weight=_model.classifier.weight, bias=_model.classifier.bias,
        )
    # turn on the gradients for some layers of the berts for fine-tuning.
    if "ptl_req_grad" in _conf.do_tuning_on_MS_scheme_:
        for name, param in getattr(_model, conf.ptl).named_parameters():
            if "mask" not in name:
                param.requires_grad = True
    if "classifier_req_grad" in _conf.do_tuning_on_MS_scheme_:
        for name, param in _model.classifier.named_parameters():
            if "mask" not in name:
                param.requires_grad = True

    # specify the optimizer
    if "adam" in _conf.do_tuning_on_MS_scheme_:
        _conf.optimizer = "adam"
    else:
        raise NotImplementedError("please specify the optimizer")

    # init the recorders.
    recorder_hooks = init_recorders(_conf, masker=None)

    # init the trainer (i.e. finetuner.)
    _conf.logger.log(
        "Finetuning on the fixed masks: Initialized tasks, masks, recorders, and initing the trainer."
    )
    trainer = BertFinetuner(_conf, logger=_conf.logger, data_iter=data_iter)

    # training/tuning.
    trainer.evaluate(model=_model, masker=None, hooks=recorder_hooks)

def experiment_with_mask(conf, model, inverse, baseline=False):
    _conf = copy.deepcopy(conf)
    _model = None

    if not baseline:
        _model = copy.deepcopy(model)
        masked_count = 0
        for name, module in _model.named_modules():
            # check all possible mask attribute names
            for mask_attr in ['mask', 'weight_mask', 'bias_mask']:
                if hasattr(module, mask_attr):
                    mask_tensor = getattr(module, mask_attr)
                    if not isinstance(mask_tensor, torch.Tensor):
                        continue
                    with torch.no_grad():
                        if inverse:
                            binary_mask = (mask_tensor > conf.threshold).float()
                            mask_tensor.data = 1.0 - binary_mask
                            conf.logger.log(f"[INVERTED MASK] {name}.{mask_attr}")
                        else:
                            conf.logger.log(f"[KEEPING MASK] {name}.{mask_attr}")
                    mask_tensor.requires_grad = False
                    masked_count += 1

        conf.logger.log(f"[INFO] Total masked tensors found and frozen: {masked_count}")
        if masked_count == 0:
            conf.logger.log("[WARN] No masked tensors found — check maskers.py for attribute name")
    else:
        # baseline has no masks — leave everything trainable
        _model = BertForSequenceClassification.from_pretrained(
            _conf.model,
            num_labels=2,
            cache_dir=conf.pretrained_weight_path,
        )

    # move to cpu to save on space
    _model = _model.cpu()
    tokenizer = BertTokenizer.from_pretrained(_conf.model)

    # making sure we are letting the classifier update w/o masks
    _model.classifier = maskers.MaskedLinear0(weight=_model.classifier.weight, bias=_model.classifier.bias,)

    results = {}
    for task in ["boolQ", "sst2", "multirc", "copa"]:
        for dialect in ["sae", "aave"]:
            task_name = f"{task}_{dialect}"
            conf.logger.log(f"[INFO] Evaluating inverse={inverse} mask on {task_name}, baseline={baseline}")

            # setting parameters for the task training
            _conf.task = task_name
            _conf.drop_rate = 0.1
            _conf.weight_decay = 0.01
            _conf.max_seq_len = 512 if task in ["boolQ", "multirc"] else 128
            _conf.eval_every_batch = 10
            _conf.lr = 2e-5
            _conf.lr_encoder = 5e-6

            # load data
            data_iter_cls = task_configs.task2dataiter[task_name]
            task_data_iter = data_iter_cls(
                task_name, _conf.model, tokenizer, _conf.max_seq_len
            )
 
            # unfreeze everything except masks
            for name, param in _model.named_parameters():
                if "mask" not in name:
                    param.requires_grad = True
            
            _conf.checkpoint_root = os.path.join(
                conf.checkpoint_root,
                f"{'baseline' if baseline else ('inv' if inverse else 'learned')}_mask",
                task_name
            )
            os.makedirs(_conf.checkpoint_root, exist_ok=True)

            recorder_hooks = init_recorders(_conf, masker=None)
            trainer = BertFinetuner(_conf, logger=_conf.logger, data_iter=task_data_iter)
            trainer.train(model=_model, masker=None, hooks=recorder_hooks)
            results[task_name] = trainer.results

            # pull back to CPU and clear GPU between tasks
            _model = _model.cpu()
            del trainer
            del task_data_iter
            del recorder_hooks
            torch.cuda.empty_cache()

    del _model
    torch.cuda.empty_cache()
    return results
    

def main(conf):
    # general init.
    if conf.override:
        for name, value in config.items():
            assert type(getattr(conf, name)) == type(value), f"{name} {value}"
            setattr(conf, name, value)
    init_config(conf)

    # experiementing with messing with which layers are masked
    which_layers = {
        "2-11": "2,3,4,5,6,7,8,9,10,11",
        "2-4": "2,3,4",
        "5-8": "5,6,7,8",
        "9-11": "9,10,11"
    }
    which_layers = {
        "9-11": "9,10,11"
    }

    times = {}

    for index, (key, value) in enumerate(which_layers.items()):
        # timing how long experiments take
        start_time = datetime.now()

        # experiment with how we chose which layers to mask to see how that effects performance
        conf.layers_to_mask = value

        # should always start with aave/sae classification task
        conf.task = "aave_mask"

        # init the task.
        model, data_iter = init_task(conf)

        # init the mask.
        model, masker = confirm_experiment(conf, model)

        # init the recorders.
        recorder_hooks = init_recorders(conf, masker)    
        conf.logger.log("Initialized tasks, masks, recorders, and initing the trainer.")

        # init the trainer (i.e. finetuner.)
        trainer = BertFinetuner(conf, logger=conf.logger, data_iter=data_iter)

        # training/tuning aave/sae classification task.
        conf.logger.log("Starting training/validation for AAVE/SAE Classification.")
        start_aave = datetime.now()
        trainer.train(model, masker, hooks=recorder_hooks)
        end_aave = datetime.now()

        # saving results
        conf.logger.log("Finishing training/validation for SAE/AAVE Classification")
        aave_sae_results = trainer.results
        conf.logger.log(f"Results from AAVE-SAE Classification task: {aave_sae_results}")
        conf.logger.log(f"Saving AAVE/SAE Classification Results: results/results_aave_{datetime.now()}")
        save_results(aave_sae_results, "aave_sae", which_layers=key)
        
        # load best checkpoint so experiments use best model, not last step
        best_state_path = os.path.join(conf.checkpoint_root, "best_state")
        if os.path.exists(best_state_path):
            conf.logger.log("[INFO] Loading best checkpoint for mask experiments.")
            model.load_state_dict(torch.load(best_state_path))
        else:
            conf.logger.log("[WARN] No best checkpoint found, using final training state.")

        # run mask experiments before finalizing logs
        conf.logger.log("Starting mask experiments.")
        """

        """
        # experiment #1 - inverted mask (shut off AAVE/SAE-distinguishing features)
        conf.logger.log("[EXPERIMENT 1] Inverted mask on NLU tasks.")
        start_inverse = datetime.now()
        results_inverse = experiment_with_mask(conf, model, inverse=True)
        end_inverse = datetime.now()
        conf.logger.log(f"results from inverted experiments {results_inverse}")
        conf.logger.log(f"Saving Experimental Results to Directory: results/results_{datetime.now()}")
        save_results(results_inverse, "inverse", which_layers=key)

        # experiment #2 - learned mask applied to NLU tasks (do features transfer?)
        conf.logger.log("[EXPERIMENT 2] Learned mask on NLU tasks.")
        start_transfer = datetime.now()
        results_transfer = experiment_with_mask(conf, model, inverse=False)
        end_transfer = datetime.now()
        save_results(results_transfer, "transfer", which_layers=key)

        # experiment #3 - unpatched baseline model, no mask
        conf.logger.log("[EXPERIMENT 3] Baseline (no mask) on NLU tasks.")
        start_baseline = datetime.now()
        results_baseline = experiment_with_mask(conf, model, inverse=False, baseline=True)
        end_baseline = datetime.now()
        save_results(results_baseline, f"baseline", which_layers=key)

        end_time = datetime.now()
        overall = end_time - start_time
        inverse = end_inverse - start_inverse
        aave = end_aave - start_aave
        transfer = end_transfer - start_transfer
        baseline = end_baseline - start_baseline
        times[key] = {
            f"{key}_overall" : overall,
            f"{key}_aave" : aave,
            f"{key}_inverse": inverse,
            f"{key}_transfer": transfer,
            f"{key}_baseline": baseline
        }
        

    # saving the times from the experiements
    conf.logger.log(f"Times for the Experiments {times}")
    with open(f"results/times.csv_{datetime.now()}", "w+") as f: 
        writer = csv.writerow(f)
        header = ["Task", "Time Elapsed"]

        writer.writerow(header)
        for key, value in times:
            for  key, v in value:
                writer.writerow(key, v)

    # update the status.
    conf.logger.log("Finished with Experiments.")
    conf.is_finished = True
    logging.save_arguments(conf)
    os.system(f"echo {conf.checkpoint_root} >> {conf.job_id}")

def save_results(res, experiment, which_layers="2-11"):
    os.makedirs(f"results/{which_layers}", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    with open(f"results/{which_layers}/results_{timestamp}_{experiment}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)

        if experiment == "aave_sae":
            writer.writerow(["Batch", "Val Accuracy", "Val CI Lower", "Val CI Upper", 
                             "Test Accuracy", "Test CI Lower", "Test CI Upper"])
            for key, value in res.items():
                batch = int(key.split("_")[1])
                val = value.get("val_dl", {}) or {}
                tst = value.get("tst_dl", {}) or {}
                writer.writerow([
                    batch,
                    val.get("accuracy", ""),
                    val.get("ci_lower", ""),
                    val.get("ci_upper", ""),
                    tst.get("accuracy", ""),
                    tst.get("ci_lower", ""),
                    tst.get("ci_upper", ""),
                ])

        else:
            for task, task_data in res.items():
                # write task header row
                writer.writerow([task])

                is_multirc = "multirc" in task
                if is_multirc:
                    writer.writerow([
                        "Batch",
                        "Val Accuracy", "Val F1", "Val CI Lower", "Val CI Upper",
                        "Test Accuracy", "Test F1", "Test CI Lower", "Test CI Upper"
                    ])
                else:
                    writer.writerow([
                        "Batch",
                        "Val Accuracy", "Val CI Lower", "Val CI Upper",
                        "Test Accuracy", "Test CI Lower", "Test CI Upper"
                    ])

                for key, value in task_data.items():
                    try:
                        batch = int(key.split("_")[1])
                    except (IndexError, ValueError):
                        continue

                    val = value.get("val_dl", {}) or {}
                    tst = value.get("tst_dl", {}) or {}

                    if is_multirc:
                        writer.writerow([
                            batch,
                            val.get("accuracy", ""),
                            val.get("f1", ""),
                            val.get("ci_lower", ""),
                            val.get("ci_upper", ""),
                            tst.get("accuracy", ""),
                            tst.get("f1", ""),
                            tst.get("ci_lower", ""),
                            tst.get("ci_upper", ""),
                        ])
                    else:
                        writer.writerow([
                            batch,
                            val.get("accuracy", ""),
                            val.get("ci_lower", ""),
                            val.get("ci_upper", ""),
                            tst.get("accuracy", ""),
                            tst.get("ci_lower", ""),
                            tst.get("ci_upper", ""),
                        ])

                # blank row between tasks for readability
                writer.writerow([])


def init_config(conf):
    conf.is_finished = False
    assert conf.ptl in conf.model

    # configure the training device.
    assert conf.world is not None, "Please specify the gpu ids."
    conf.world = (
        [int(x) for x in conf.world.split(",")]
        if "," in conf.world
        else [int(conf.world)]
    )
    conf.n_sub_process = len(conf.world)

    # init the masking scheduler.
    conf.masking_scheduler_conf_ = (
        param_parser.dict_parser(conf.masking_scheduler_conf)
        if conf.masking_scheduler_conf is not None
        else None
    )
    if conf.masking_scheduler_conf is not None:
        for k, v in conf.masking_scheduler_conf_.items():
            setattr(conf, f"masking_scheduler_{k}", v)

    # init the layers to mask.
    assert conf.layers_to_mask is not None, "Please specify which BERT layers to mask."
    conf.layers_to_mask_ = (
        [int(x) for x in conf.layers_to_mask.split(",")]
        if "," in conf.layers_to_mask
        else [int(conf.layers_to_mask)]
    )


    # init the params for structure pruning.
    if (
        conf.structured_masking is not None
        and conf.structured_masking_types is not None
    ):
        conf.structured_masking_types_ = conf.structured_masking_types.split(",")
    else:
        conf.structured_masking_types_ = None

    # init the params for do_tuning_on_MS_scheme
    if conf.do_tuning_on_MS:
        assert conf.do_tuning_on_MS_scheme is not None
        conf.do_tuning_on_MS_scheme_ = conf.do_tuning_on_MS_scheme.split(",")

    # re-configure batch_size if sub_process > 1.
    if conf.n_sub_process > 1:
        conf.batch_size = conf.batch_size * conf.n_sub_process

    # configure cuda related.
    assert torch.cuda.is_available()
    torch.manual_seed(conf.manual_seed)
    torch.cuda.manual_seed(conf.manual_seed)
    torch.cuda.set_device(conf.world[0])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True if conf.train_fast else False

    # define checkpoint for logging.
    checkpoint.init_checkpoint(conf)

    # display the arguments' info.
    logging.display_args(conf)

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_root)


if __name__ == "__main__":
    conf = get_args()

    main(conf)