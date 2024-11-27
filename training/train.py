
from ultralytics import YOLO
from typing import Optional, Union
import fire





def main(
        model_ckpt: str = 'yolo11n.pt',
        dataset: str = 'C:/Users/admin/maicon/Jimyeong/dataset/testdataset/data.yaml',
        epochs: int = 100,
        patience: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        save: bool = True,
        save_period: int = -1,
        device: Optional[Union[str, int]] = 0,
        workers: int = 4,
        run_name: Optional[str] = None,
        optimizer: str = 'auto',
        seed: int = 0,
        cos_lr: bool = True,
        resume: bool = False,
        lr0: float = 0.001,
        warmup_epochs: float = 3.0,
        cls_weight: float = 0.5,
        dropout: float = 0.0,
        val: bool = True,
        plots: bool = True
):
    
    model = YOLO(model_ckpt)
    model.train(
        data=dataset,
        epochs=epochs,
        patience=patience,
        batch=batch_size,
        imgsz=img_size,
        save=save,
        save_period=save_period,
        device=device,
        workers=workers,
        name=run_name,
        optimizer=optimizer,
        seed=seed,
        cos_lr=cos_lr,
        resume=resume,
        lr0=lr0,
        warmup_epochs=warmup_epochs,
        cls=cls_weight,
        dropout=dropout,
        val=val,
        plots=plots,
    )



if __name__ == '__main__':
    fire.Fire(main)


