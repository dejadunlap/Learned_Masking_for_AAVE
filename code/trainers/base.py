import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader


class BaseTrainer(object):
    """a basic trainer"""

    def __init__(self, conf, logger, criterion=torch.nn.CrossEntropyLoss()):
        self.conf = conf
        self.logger = logger
        self.log_fn_json = logger.log_metric
        self.log_fn = logger.log
        self.criterion = criterion

        # counter.
        self._batch_step = 0
        self._epoch_step = 0

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def _parallel_to_device(self, model):
        model = model.cuda()
        if len(self.conf.world) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.conf.world)
        return model

    @property
    def batch_step(self):
        return self._batch_step

    @property
    def epoch_step(self):
        return self._epoch_step

    def _wrap_datasplits(self, data_iter):
        for attr_name in ["trn_dl", "val_dl", "tst_dl"]:
            try:
                ds = getattr(data_iter, attr_name)
            except AttributeError:
                self.log_fn(f"[WARN]: skip ``None`` split: {attr_name}")
                continue
            if attr_name == "trn_dl":
                sampler = RandomSampler
            elif attr_name == "val_dl" or attr_name == "tst_dl":
                sampler = SequentialSampler
            else:
                raise ValueError
            setattr(
                self,
                attr_name,
                DataLoader(ds, sampler=sampler(ds), batch_size=self.conf.batch_size),
            )