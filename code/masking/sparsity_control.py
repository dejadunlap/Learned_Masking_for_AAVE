import functools

from .maskers import MaskedLinearX

# sparsity controller
def automated_gradual_sparsity(init_sparsity, final_sparsity, interval_epoch, init_epoch, final_epoch):
    def f(current_epoch, current_sparsity):
        if current_epoch > final_epoch: 
            return final_sparsity
        
        # estimate sparsity
        span = final_epoch - init_epoch


        if span != 0: 
            target_sparsity = (final_sparsity + (init_sparsity - final_sparsity) * (1.0 - (1.0 * (current_epoch - init_epoch) / span)) ** 3)
        else:
            target_sparsity = final_sparsity

        return target_sparsity
    return f

class MaskerScheduler(object):
    def __init__(self, conf):
        self.conf = conf
        self.masking_scheduler_conf_ = conf.masking_scheduler_conf_
        self._current_sparsity = 0

        self.init_sparsity = self.masking_scheduler_conf_["initial_sparsity"]
        self.final_sparsity = self.masking_scheduler_conf_["final_sparsity"]
        self.get_sparsity_fn = automated_gradual_sparsity(
            init_sparsity = self.init_sparsity,
            final_sparsity = self.final_sparsity, 
            interval_epoch = self.masking_scheduler_conf_["sparsity_warmup_interval_epoch"],
            init_epoch = self.masking_scheduler_conf_["init_epoch"], 
            final_epoch = self.masking_scheduler_conf_["final_epoch"]
        )
    
    def step(self, cur_epoch):
        self.cur_epoch = cur_epoch

        # derive target sparsity under init and final sparsity constraints
        _target_sparsity = self.get_sparsity_fn(cur_epoch, self.current_sparsity)

        if self.masking_scheduler_conf_["final_sparsity"] > self.init_sparsity: 
            min_sparsity = self.init_sparsity
            max_sparsity = self.final_sparsity
        else: 
            min_sparsity = self.final_sparsity
            max_sparsity = self.init_sparsity
        
        self.target_sparsity = min(max_sparsity, max(_target_sparsity, min_sparsity))

        # get incremental sparsity ratio w curr sparsity
        _incremental_sparsity = (self.target_sparsity - self.current_sparsity) / (
            1 - self._current_sparsity
        )

        return  _incremental_sparsity, self.target_sparsity, self.is_sparsity_change()

    
    def is_meet_sparsity(self):
        return self.target_sparsity >= self.final_sparsity

    def is_sparsity_change(self):
        if self._current_sparsity == self.target_sparsity: 
            return False
        else: 
            self._current_sparsity = self.target_sparsity
            return True

            
    def get_sparsity_over_whole_model(self, model, masker, trainable=True):
        def get_modified_linear_modules(my_module):
            modules = []
            for m in my_module.children():
                if isinstance(m, MaskedLinearX):
                    modules.append(m)
                else:
                    modules.extend(get_modified_linear_modules(m))
            return modules

        def get_info_from_one_layer(masks, tensor_name, info_type):
            mask = masks[0 if tensor_name == "weight" else 1]
            if mask is not None:
                if info_type == "nnz":
                    return mask.sum()
                elif info_type == "tot":
                    return mask.numel()
                else:
                    raise NotImplementedError(
                        f"the info_type={info_type} is not supported yet."
                    )
            else:
                return None

        # get modified linear modules
        modified_linear_modules = get_modified_linear_modules(model)

        # get the masks as well as the corresponding information in these modified linear modules.
        masks = [module.get_masks() for module in modified_linear_modules]
        nnz_info = [
            get_info_from_one_layer(mask, tensor_name="weight", info_type="nnz")
            for mask in masks
        ] + [
            get_info_from_one_layer(mask, tensor_name="bias", info_type="nnz")
            for mask in masks
        ]
        tot_info = [
            get_info_from_one_layer(mask, tensor_name="weight", info_type="tot")
            for mask in masks
        ] + [
            get_info_from_one_layer(mask, tensor_name="bias", info_type="tot")
            for mask in masks
        ]

        # evaluate the overall sparsity.
        total_nnz = sum([x for x in nnz_info if x is not None])
        total_tot = sum([x for x in tot_info if x is not None])
        return 1 - total_nnz / total_tot
           


