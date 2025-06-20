from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import hydra
from omegaconf import DictConfig
import torch

from dataset import init_dataset, compute_input_output_dims
from extra_features import ExtraFeatures, DummyExtraFeatures
from diffusion_discrete import DiscreteDenoisingDiffusion
from analysis.spectre_utils import CrossDomainSamplingMetrics
from analysis.iker_spectre_utils import TFG_iker_SamplingMetrics
import utils
import os

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    hydra_path = hydra.utils.get_original_cwd()
    ####### START IKER #######
    os.environ["MASTER_PORT"] = str(10000 + (os.getpid() % 10000))  # Use a different port per process
    print("HI")
    print(os.path.abspath(os.getcwd()))
    ####### END IKER #######

    data_loaders, num_classes, max_n_nodes, nodes_dist, edge_types, node_types, n_nodes = init_dataset(cfg.dataset.name, cfg.train.batch_size, hydra_path, cfg.dataset.sample, cfg.general.num_train)

    extra_features = ExtraFeatures(cfg.model.extra_features, max_n_nodes)
    domain_features = DummyExtraFeatures()

    input_dims, output_dims = compute_input_output_dims(data_loaders['train'], extra_features, domain_features)

    # use my own metrics
    #sampling_metrics = CrossDomainSamplingMetrics(data_loaders)
    sampling_metrics = TFG_iker_SamplingMetrics(data_loaders)

    model = DiscreteDenoisingDiffusion(cfg, input_dims, output_dims, nodes_dist, node_types, edge_types, extra_features, domain_features, data_loaders, sampling_metrics) 


    num_params = sum(p.numel() for p in model.parameters())
    param_size_bytes = num_params * 4 # For float32
    param_size_mb = param_size_bytes / (1024 * 1024)
    print(f"Parameter size: {param_size_mb:.2f} MB")

    # Set the Model and Trainer
    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=3,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)


    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)


    use_gpu = 1 > 0 and torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50,
                      logger = [],
                      accumulate_grad_batches=cfg.train.accumulate_grad_batches)

    if cfg.general.setting == 'train_scratch':
        trainer.fit(model, train_dataloaders = data_loaders['train'], val_dataloaders = data_loaders['val'])
    elif cfg.general.setting == 'train_from_pretrained':
        trainer.fit(model, train_dataloaders = data_loaders['train'], val_dataloaders = data_loaders['val'], ckpt_path = cfg.general.ckpt_path)
    elif cfg.general.setting == 'test':
        try:
            print(f"cfg.general.ckpt_path = {cfg.general.ckpt_path} ({type(cfg.general.ckpt_path)})")

            trainer.test(model, dataloaders=data_loaders['test'], ckpt_path=cfg.general.ckpt_path)
        except Exception as e:
            import traceback
            print(" EXCEPCIÓN ORIGINAL:")
            traceback.print_exc()
            print(" exception.args:", e.args)
            raise

    elif cfg.general.setting == 'test2':
        try:
            print(f"cfg.general.ckpt_path = {cfg.general.ckpt_path} ({type(cfg.general.ckpt_path)})")

            trainer.test(model, dataloaders=data_loaders['train'], ckpt_path=cfg.general.ckpt_path)
        except Exception as e:
            import traceback
            print(" EXCEPCIÓN ORIGINAL:")
            traceback.print_exc()
            print(" exception.args:", e.args)
            raise

if __name__ == '__main__':
    main()
