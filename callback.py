from TrainingViz.callback import CallbackViz
import torch


class CallbackVDP(CallbackViz):
    def __init__(self, x_name='xn', relocation=True):
        super().__init__()
        self.x_name = x_name
        self.relocation = relocation
        self.mean = 0
        self.overfit_data = None
        self.dist = []

    def end_epoch(self, trainer, output):
        super().end_epoch(trainer, output)
        if self.overfit_data is None:
            self.overfit_data = {
                    "mean_train_loss": [output["mean_train_loss"].detach()],
                    "mean_dev_loss": [output["mean_dev_loss"]]}
        else:
            {key: val.append(output[key].detach()) for key, val in
             self.overfit_data.items()}

    def end_eval(self, trainer, output, i):
        self.dist.append(trainer.train_data.dataset.datadict['xn'])
        if i == 0:
            self.mean = torch.zeros(output[f'train_{self.x_name}'].shape[-1])
        elif (i + 1) % 1 == 0 and self.relocation:
            min_loss = torch.tensor(-torch.inf)
            save_data = None
            for d_batch in trainer.dev_data:
                eval_output = trainer.model(d_batch)
                loss = eval_output[trainer.dev_metric]
                if loss > min_loss:
                    min_loss = loss
                    save_data = d_batch
            self.mean = torch.squeeze(save_data[self.x_name])
            shape = trainer.train_data.dataset.datadict[self.x_name].shape
            new_train = None
            if self.relocation:
                new_train = torch.stack([torch.normal(x, 0.5, shape[0:-1])
                                         for x in self.mean], axis=-1)
            else:
                new_train = torch.randn(shape)
            trainer.train_data.dataset.datadict[self.x_name] = new_train


class CallbackTT(CallbackViz):
    def __init__(self, x_name='xn', relocation=True):
        super().__init__()
        self.x_name = x_name
        self.relocation = relocation
        self.mean_r = 0
        self.mean = 0

    def end_epoch(self, trainer, output):
        super().end_epoch(trainer, output)

    def end_eval(self, trainer, output, i):
        if i == 0:
            self.mean = torch.zeros(output[f'train_{self.x_name}'].shape[-1])
            self.mean = torch.zeros(output['train_r'].shape[-1])
        elif (i + 1) % 5 == 0 and self.relocation:
            min_loss = torch.tensor(-torch.inf)
            save_data = None
            for d_batch in trainer.dev_data:
                eval_output = trainer.model(d_batch)
                loss = eval_output[trainer.dev_metric]
                if loss > min_loss:
                    min_loss = loss
                    save_data = d_batch
            self.mean += torch.squeeze(save_data[self.x_name]) * 0.05
            self.r_mean += torch.mean(save_data['r']) * 0.05
            shape = trainer.train_data.dataset.datadict[self.x_name].shape
            new_train = None
            new_r = None
            r_shape = trainer.train_data.dataset.datadict['r'].shape

            if self.relocation:
                new_train = torch.stack([torch.normal(x, 0.5, shape[0:-1])
                                         for x in self.mean], axis=-1)
                new_r = [torch.normal(self.r_mean, 0.5,
                                      (1, 1)) * torch.ones(r_shape[1:])
                         for k in range(r_shape[0])]
            else:
                new_train = torch.randn(shape)
                new_r = [torch.rand(1, 1) * torch.ones(r_shape[1:])
                         for k in range(r_shape[0])]

            trainer.train_data.dataset.datadict[self.x_name] = new_train
            new_r = torch.cat(new_r)
            new_r = new_r.reshape(r_shape)
            trainer.train_data.dataset.datadict['r'] = new_r
