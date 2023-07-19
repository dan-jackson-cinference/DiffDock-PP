import resource
import time

import hydra

import torch

from hydra.core.config_store import ConfigStore
from torch_geometric.loader import DataLoader

from confidence import evaluate_confidence
from data.data_factory import load_data
from data.load_data import get_datasets, get_loaders, split_data
from data.dataset import BindingDataset, SamplingDataset
from evaluate import evaluate_all_predictions

from geom_utils.transform import NoiseSchedule, NoiseTransform
from model.factory import load_confidence_model, load_score_model, to_cuda

# from helpers import WandbLogger, TensorboardLogger
from sample import Sampler, initialize_random_positions
from seed import set_seed
from config import DiffDockCfg

from utils import printt

cs = ConfigStore.instance()
cs.store(name="test_config", node=DiffDockCfg)


@hydra.main(version_base=None, config_path="../config/", config_name="config")
def main(cfg: DiffDockCfg):
    """test mode: load up all replicates from checkpoint directory
    and evaluate by sampling from reverse diffusion process"""
    printt("Starting Inference")
    set_seed(cfg.training_cfg.seed)
    inf_cfg = cfg.inference_cfg
    log_cfg = cfg.logging_cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.hub.set_dir("torchhub")

    start_time = time.time()
    # load raw data

    # needs to be set if DataLoader does heavy lifting
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # needs to be set if sharing resources
    if cfg.training_cfg.num_workers >= 1:
        torch.multiprocessing.set_sharing_strategy("file_system")

    processed_data, data_params = load_data(
        cfg.run_cfg.experiment_root_dir, cfg.data_cfg, cfg.model_cfg.lm_embed_dim
    )
    printt("finished loading and processing data")
    printt("running inference")

    fold = 0
    # load and convert data to DataLoaders
    noise_schedule = NoiseSchedule.from_config(cfg.diffusion_cfg)
    noise_transform = NoiseTransform(noise_schedule)
    # data_split = split_data(
    #     processed_data,
    #     fold,
    #     cfg.training_cfg.num_folds,
    #     cfg.training_cfg.mode,
    # )
    # datasets = get_datasets(data_split, noise_transform, BindingDataset)
    # dataloaders = get_loaders(
    #     datasets,
    #     cfg.training_cfg.batch_size,
    #     cfg.training_cfg.num_workers,
    #     cfg.training_cfg.num_gpu,
    #     cfg.training_cfg.mode,
    # )
    printt("finished creating data splits")

    # get model and load checkpoint, if relevant
    score_model = load_score_model(
        cfg.diffusion_cfg,
        cfg.model_cfg.score_model_cfg,
        noise_schedule,
        cfg.run_cfg.score_model_path,
        cfg.model_cfg.lm_embed_dim,
        num_atoms=23,
        num_gpu=0,
        fold=fold,
    )
    score_model = to_cuda(score_model, cfg.training_cfg.gpu, cfg.training_cfg.gpu)

    confidence_model = load_confidence_model(
        cfg.model_cfg.confidence_model_cfg,
        noise_schedule,
        cfg.run_cfg.filtering_model_path,
        cfg.model_cfg.lm_embed_dim,
        num_atoms=23,
        fold=fold,
    )
    confidence_model = to_cuda(
        confidence_model, cfg.training_cfg.gpu, cfg.training_cfg.gpu
    )
    printt("finished loading model")
    # MAIN RUN
    # run reverse diffusion process
    if inf_cfg.temp_cfg is not None:
        print(f"temp_sampling: {inf_cfg.temp_cfg.temp_sampling}")

    processed_data[str(cfg.data_cfg.pred_cfg.pdb_id)].write_to_pdb(
        f"{cfg.data_cfg.pred_cfg.pdb_id}_orig.pdb"
    )

    initial_positions = initialize_random_positions(
        processed_data[str(cfg.data_cfg.pred_cfg.pdb_id)].graph,
        cfg.run_cfg.num_samples,
        cfg.diffusion_cfg.tr_s_max,
        no_torsion=True,
    )

    sampler = Sampler(
        score_model,
        noise_transform,
        inf_cfg.num_steps,
        batch_size=5,
        no_final_noise=inf_cfg.no_final_noise,
        no_random=inf_cfg.no_random,
        device=device,
    )

    sampling_dataset = SamplingDataset(initial_positions)
    samples = sampler.sample(sampling_dataset, inf_cfg.ode, inf_cfg.temp_cfg)
    samples_loader = DataLoader(samples[-1], batch_size=cfg.run_cfg.num_samples)
    confidence = evaluate_confidence(
        confidence_model, samples_loader, device
    )  # TODO -> maybe list inside

    print(confidence)
    samples_list = [sample for sample in samples[-1]]
    results = sorted(list(zip(samples_list, confidence)), key=lambda x: -x[1])
    printt("Finished Complex!")

    printt(f"Finished run {log_cfg.run_name}")
    print(f"Total time spent: {time.time()-start_time}")
    print(results[0])

    for i, (sample, confidence_val) in enumerate(results[:10]):
        processed_data[str(cfg.data_cfg.pred_cfg.pdb_id)].update_proteins_from_graph(
            sample
        )
        processed_data[str(cfg.data_cfg.pred_cfg.pdb_id)].write_to_pdb(
            f"sample_{i}_confidence_{confidence_val:.3f}.pdb"
        )


if __name__ == "__main__":
    main()

    # if args.wandb_sweep:

    #     def try_params():
    #         run = wandb.init()
    #         args.temp_sampling = wandb.config.temp_sampling
    #         args.temp_psi = wandb.config.temp_psi
    #         args.temp_sigma_data_tr = wandb.config.temp_sigma_data_tr
    #         args.temp_sigma_data_rot = wandb.config.temp_sigma_data_rot

    #         print(
    #             f"running run with: {args.temp_sampling, args.temp_psi, args.temp_sigma_data_tr, args.temp_sigma_data_rot}"
    #         )
    #         printt("Running sequentially without confidence model")
    #         complex_rmsd_lt5 = []
    #         complex_rmsd_lt2 = []
    #         for i in tqdm(range(5)):
    #             try:
    #                 samples_list = sample(loaders["val"], model, args)
    #             except RuntimeError as e:
    #                 print(e)
    #                 print(traceback.format_exc())
    #                 raise e

    #             meter = evaluate_all_rmsds(loaders["val"], samples_list)
    #             (
    #                 ligand_rmsd_summarized,
    #                 complex_rmsd_summarized,
    #                 interface_rmsd_summarized,
    #             ) = meter.summarize(verbose=False)
    #             complex_rmsd_lt5.append(complex_rmsd_summarized["lt5"])
    #             complex_rmsd_lt2.append(complex_rmsd_summarized["lt2"])
    #             printt(f"Finished {i}-th sweep over the data")
    #         complex_rmsd_lt5 = np.array(complex_rmsd_lt5)
    #         complex_rmsd_lt2 = np.array(complex_rmsd_lt2)
    #         print(f"Average CRMSD < 5: {complex_rmsd_lt5.mean()}")
    #         print(f"Average CRMSD < 2: {complex_rmsd_lt2.mean()}")
    #         wandb.log(
    #             {
    #                 "complex_rmsd_lt5": complex_rmsd_lt5.mean(),
    #                 "complex_rmsd_lt2": complex_rmsd_lt2.mean(),
    #             }
    #         )

    #     def try_params_with_confidence_model():
    #         wandb_key = "INSERT_YOUR_WANDB_KEY"
    #         wandb.login(key=wandb_key, relogin=True)
    #         run = wandb.init()
    #         args.temp_sampling = wandb.config.temp_sampling
    #         args.temp_psi = wandb.config.temp_psi
    #         args.temp_sigma_data_tr = wandb.config.temp_sigma_data_tr
    #         args.temp_sigma_data_rot = wandb.config.temp_sigma_data_rot

    #         print(
    #             f"running run with confidence model with: {args.temp_sampling, args.temp_psi, args.temp_sigma_data_tr, args.temp_sigma_data_rot}"
    #         )

    #         args.num_samples = 5

    #         # run reverse diffusion process
    #         try:
    #             loaders_repeated, results = generate_loaders(
    #                 loaders["val"], args
    #             )  # TODO adapt sample size

    #             for i, loader in tqdm(
    #                 enumerate(loaders_repeated), total=len(loaders_repeated)
    #             ):
    #                 samples_list = sample(
    #                     loader, model, args
    #                 )  # TODO: should work on data loader
    #                 samples_loader = DataLoader(
    #                     samples_list, batch_size=args.batch_size
    #                 )
    #                 pred_list = evaluate_confidence(
    #                     model_confidence, samples_loader, args
    #                 )  # TODO -> maybe list inside
    #                 results[i] = results[i] + sorted(
    #                     list(zip(samples_list, pred_list)), key=lambda x: -x[1]
    #                 )
    #                 printt("Finished Complex!")
    #         except Exception as e:
    #             print(e)
    #             print(traceback.format_exc())
    #             raise e

    #         printt(f"Finished run {args.run_name}")

    #         meter = evaluate_all_predictions(results)
    #         (
    #             ligand_rmsd_summarized,
    #             complex_rmsd_summarized,
    #             interface_rmsd_summarized,
    #         ) = meter.summarize(verbose=False)
    #         complex_rmsd_lt5 = complex_rmsd_summarized["lt5"]
    #         complex_rmsd_lt2 = complex_rmsd_summarized["lt2"]

    #         print(f"Average CRMSD < 5: {complex_rmsd_lt5}")
    #         print(f"Average CRMSD < 2: {complex_rmsd_lt2}")
    #         wandb.log(
    #             {
    #                 "complex_rmsd_lt5": complex_rmsd_lt5,
    #                 "complex_rmsd_lt2": complex_rmsd_lt2,
    #             }
    #         )

    #     def try_actual_steps_with_confidence_model():
    #         wandb_key = "INSERT_YOUR_WANDB_KEY"
    #         wandb.login(key=wandb_key, relogin=True)
    #         run = wandb.init()
    #         args.actual_steps = wandb.config.actual_steps

    #         print(f"Running with actual steps: {args.actual_steps}")

    #         args.num_samples = 10

    #         # run reverse diffusion process
    #         try:
    #             loaders_repeated, results = generate_loaders(
    #                 loaders["val"], args
    #             )  # TODO adapt sample size

    #             for i, loader in tqdm(
    #                 enumerate(loaders_repeated), total=len(loaders_repeated)
    #             ):
    #                 samples_list = sample(
    #                     loader, model, args
    #                 )  # TODO: should work on data loader
    #                 samples_loader = DataLoader(
    #                     samples_list, batch_size=args.batch_size
    #                 )
    #                 pred_list = evaluate_confidence(
    #                     model_confidence, samples_loader, args
    #                 )  # TODO -> maybe list inside
    #                 results[i] = results[i] + sorted(
    #                     list(zip(samples_list, pred_list)), key=lambda x: -x[1]
    #                 )
    #                 printt("Finished Complex!")
    #         except Exception as e:
    #             print(e)
    #             print(traceback.format_exc())
    #             raise e

    #         printt(f"Finished run {args.run_name}")

    #         meter = evaluate_all_predictions(results)
    #         (
    #             ligand_rmsd_summarized,
    #             complex_rmsd_summarized,
    #             interface_rmsd_summarized,
    #         ) = meter.summarize(verbose=False)
    #         complex_rmsd_lt5 = complex_rmsd_summarized["lt5"]
    #         complex_rmsd_lt2 = complex_rmsd_summarized["lt2"]

    #         print(f"Average CRMSD < 5: {complex_rmsd_lt5}")
    #         print(f"Average CRMSD < 2: {complex_rmsd_lt2}")
    #         wandb.log(
    #             {
    #                 "complex_rmsd_lt5": complex_rmsd_lt5,
    #                 "complex_rmsd_lt2": complex_rmsd_lt2,
    #             }
    #         )

    # sweep_configuration = {
    #     'method': 'grid',
    #     'name': 'sweep',
    #     'metric': {'goal': 'maximize', 'name': 'complex_rmsd_lt2'},
    #     'parameters':
    #     {
    #         'actual_steps': {'values': [30, 32, 34, 36, 38, 40]},
    #     }
    # }

    # sweep_configuration = {
    #     "method": "bayes",
    #     "name": "sweep",
    #     "metric": {"goal": "maximize", "name": "complex_rmsd_lt2"},
    #     "parameters": {
    #         "temp_sampling": {"max": 4.0, "min": 0.0},
    #         "temp_psi": {"max": 2.0, "min": 0.0},
    #         "temp_sigma_data_tr": {"max": 1.0, "min": 0.0},
    #         "temp_sigma_data_rot": {"max": 1.0, "min": 0.0},
    #     },
    # }
    # sweep_id = wandb.sweep(
    #     sweep=sweep_configuration,
    #     project="DIPS optimize low temp with LRMSD conf model",
    # )

    # wandb.agent(sweep_id, function=try_params_with_confidence_model, count=20)
    # return


# if args.run_inference_without_confidence_model:
#     printt("Running sequentially without confidence model")
#     # loaders["test"].data = sorted(loaders["test"].data, key=lambda x:x['receptor_xyz'].shape[0] + x['ligand_xyz'].shape[0])

#     # list_bs_32 = loaders["test"][0:64]
#     # print(f'list_bs_32: {list_bs_32}')
#     # list_bs_16 = loaders["test"][64:80]
#     # print(f'list_bs_16: {list_bs_16}')
#     # list_bs_8 = loaders["test"][80:88]
#     # print(f'list_bs_8: {list_bs_8}')
#     # list_bs_4 = loaders["test"][88:]
#     # print(f'list_bs_4: {list_bs_4}')

#     full_list = [loaders["val"]]
#     complex_rmsd_lt5 = []
#     complex_rmsd_lt2 = []
#     time_to_load_data = time.time() - start_time
#     print(f"time_to_load_data: {time_to_load_data}")
#     start_time = time.time()
#     for i in tqdm(range(1)):
#         # print(f'bs: {32}')
#         # samples_list_bs_32 = sample(list_bs_32, model, args, in_batch_size=32)
#         # print(f'bs: {16}')
#         # samples_list_bs_16 = sample(list_bs_16, model, args, in_batch_size=16)
#         # print(f'bs: {8}')
#         # samples_list_bs_8 = sample(list_bs_8, model, args, in_batch_size=8)
#         # print(f'bs: {4}')
#         # samples_list_bs_4 = sample(list_bs_4, model, args, in_batch_size=4)

#         # samples_list = samples_list_bs_32 + samples_list_bs_16 + samples_list_bs_8 + samples_list_bs_4
#         samples_list = sample(
#             loaders["val"],
#             model,
#             args,
#             visualize_first_n_samples=args.visualize_n_val_graphs,
#             visualization_dir=args.visualization_path,
#         )
#         full_list.append(samples_list)
#         meter = evaluate_all_rmsds(loaders["val"], samples_list)
#         (
#             ligand_rmsd_summarized,
#             complex_rmsd_summarized,
#             interface_rmsd_summarized,
#         ) = meter.summarize(verbose=True)
#         complex_rmsd_lt5.append(complex_rmsd_summarized["lt5"])
#         complex_rmsd_lt2.append(complex_rmsd_summarized["lt2"])
#         printt(f"Finished {i}-th sweep over the data")

#     end_time = time.time()
#     print(f"Total time spent processing 5 times: {end_time-start_time}")
#     print(f"time_to_load_data: {time_to_load_data}")

#     complex_rmsd_lt5 = np.array(complex_rmsd_lt5)
#     complex_rmsd_lt2 = np.array(complex_rmsd_lt2)
#     print(f"Average CRMSD < 5: {complex_rmsd_lt5.mean()}")
#     print(f"Average CRMSD < 2: {complex_rmsd_lt2.mean()}")
#     dump_predictions(args, full_list)
#     printt("Dumped data!!")
#     return
