import wandb

from lib.data import DataloaderConfig
from lib.modules import Module
from .train import *
from lib.utils import *


# a basic objective factory function, can be copied and modified i.e. for different eval method
def make_objective(
    Model: Module,
    model_config: Config,
    loader: Callable,
    trainer_config: TrainerConfig,
    objective_metric: Metric,
    loader_config: DataloaderConfig = DataloaderConfig(),
):
    def run():
        run = wandb.init()  # noqa: F841 https://docs.wandb.ai/ref/python/run/
        wandb.define_metric(
            "*", step_metric="epoch"
        )  # add custom steps for mfs if needed

        # if vb(7):
        #     print("wandb.config:", wandb.config)

        trainer = Trainer(trainer_config.update(wandb.config, logger=wandb.log))

        dataloader = loader(**loader_config.update(wandb.config).dict())
        model = Model(model_config.update(wandb.config))

        trainer.init(model, dataloader, loss_board=False)
        best_loss = trainer.fit()

        objective_mf = MetricsFrame(
            [objective_metric],
            logger=wandb.log,
            flush_every=0,  # implied, but explicit is better
            xlabel=None,
        )

        if vb(7):
            preds = trainer.eval(objective_mf, pred=True)
            print(preds)
        else:
            trainer.eval(objective_mf)

        result = {k: v[-1] for k, v in objective_mf.dict.items()}
        result["loss"] = best_loss
        result["epoch"] = len(dataloader)

        if vb(5):
            print(result)

        wandb.log(result)

        return result

    return run


def tune(
    sweep_configuration: Dict,
    objective: Callable,
    max_runs=1,
    project_name=None,
):
    project_name = project_name or get_project_name()

    sweep_id = wandb.sweep(sweep_configuration, project=project_name)
    print(f"Sweep ID: {sweep_id}")

    wandb.agent(sweep_id, function=objective, count=max_runs)

    # when to teardown()? https://docs.wandb.ai/guides/sweeps/start-sweep-agents/
