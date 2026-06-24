import torch


def save_model(
        model,
        path,
        mean,
        std,
        grid_x,
        grid_y,
        input_steps,
        metrics=None
):
    """
    保存ConvLSTM模型

    参数
    ----------
    model : ConvLSTM

    path : str

    mean : float

    std : float

    grid_x : ndarray

    grid_y : ndarray

    input_steps : int

    metrics : dict
    """

    save_dict = {

        "model_state_dict":
            model.state_dict(),

        "mean":
            float(mean),

        "std":
            float(std),

        "grid_x":
            grid_x,

        "grid_y":
            grid_y,

        "input_steps":
            input_steps

    }

    if metrics is not None:

        save_dict["metrics"] = metrics

    torch.save(
        save_dict,
        path
    )

    print("\n" + "=" * 50)
    print("模型保存成功")
    print("=" * 50)
    print(f"保存路径: {path}")

    if metrics is not None:

        print(
            f"R²   : {metrics['r2']:.6f}"
        )

        print(
            f"RMSE : {metrics['rmse']:.6e}"
        )

    print("=" * 50)


def load_model_info(path):
    """
    读取保存信息
    """

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    print("\n模型信息")

    print(
        f"input_steps: "
        f"{checkpoint['input_steps']}"
    )

    print(
        f"mean: "
        f"{checkpoint['mean']}"
    )

    print(
        f"std: "
        f"{checkpoint['std']}"
    )

    return checkpoint