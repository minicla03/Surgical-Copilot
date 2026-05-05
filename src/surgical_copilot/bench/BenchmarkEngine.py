import time
import copy
import numpy as np
import torch
import wandb
from tqdm import tqdm

from surgical_copilot.bench.perturbation import PerturbationPipelines

class BenchmarkEngine:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, loss_fn, scaler, cfg, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.scaler = scaler
        self.cfg = cfg
        self.device = device
        
        self._print_model_info()

        self.history = {
            "train_loss": [],
            "clean_dice": [],
            "fps": []
        }

    def run(self):
        max_epochs = self.cfg.trainer.trainer.max_epochs
        
        for epoch in range(max_epochs):
            print(f"\n{'='*40}\nEpoch {epoch+1}/{max_epochs}\n{'='*40}")
            
            # 1. Training Phase
            train_loss = self._train()
            
            # 2. Validation Phase
            if (epoch + 1) % self.cfg.trainer.trainer.val_interval == 0:
                eval_metrics = self.eval(epoch)
                
                # History Update locale
                self.history["train_loss"].append(train_loss)
                self.history["clean_dice"].append(eval_metrics["clean_dice"])
                self.history["fps"].append(eval_metrics["fps"]["fps"])
                
                # --- Report a Terminale Arricchito ---
                print(f"\n[ Risultati Epoca {epoch+1} ]")
                print(f"Train Loss:   {train_loss:.4f}")
                print(f"Clean Dice:   {eval_metrics['clean_dice']:.4f}")
                print(f"FPS (Img/s):  {eval_metrics['fps']['fps']:.2f}")
                
                print("\n--- Robustness Drop Analysis ---")
                clean_score = eval_metrics['clean_dice']
                for scenario, score in eval_metrics['robust_dice'].items():
                    if scenario != "clean":
                        # Calcola quanto crolla la performance rispetto al clean
                        drop_pct = ((clean_score - score) / clean_score * 100) if clean_score > 0 else 0
                        print(f"  > {scenario.ljust(25)}: {score:.4f} (Drop: -{drop_pct:.1f}%)")
                print("="*40 + "\n")

                # 3. Logging Centrale su Weights & Biases
                self._log_metrics_to_wandb(epoch, train_loss, eval_metrics)

    def _train(self):
        self.model.train()
        train_loss = []

        pbar = tqdm(self.train_loader, desc=f"Training", leave=False, dynamic_ncols=True)

        for batch in pbar:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type):
                out = self.model(x)
                loss = self.loss_fn(out, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss.append(loss.item())
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        self.scheduler.step()
        return np.mean(train_loss)

    def eval(self, epoch):
        print("[*] Avvio Stress Test & Valutazione...")
        self.model.eval()
        metrics = {"robust_dice": {}}
        eval_scenarios = PerturbationPipelines.get_eval_scenarios()

        with torch.no_grad():
            for scenario_name, pipeline in eval_scenarios.items():
                scores = []
                
                pbar = tqdm(self.val_loader, desc=f"Eval [{scenario_name}]", leave=False, dynamic_ncols=True)
                
                for batch_idx, batch in enumerate(pbar):
                    
                    # SICUREZZA DATI: Creiamo una copia profonda per non inquinare la cache di MONAI
                    batch_safe = copy.deepcopy(batch)
                    batch_pert = pipeline(batch_safe)
                    
                    x = batch_pert["image"].to(self.device)
                    y = batch_pert["label"].to(self.device)

                    out = self.model(x)
                    preds = (torch.sigmoid(out) > 0.5).float()
                    scores.append(self.compute_dsc(preds, y))

                    # Log visivo SOLO del primo batch, ma ora per TUTTI gli scenari!
                    # Cosi su WandB vedi l'effetto del rumore sull'input e sull'output
                    if batch_idx == 0:
                        self._log_masks_to_wandb(x, y, preds, epoch, scenario_name)

                mean_score = np.mean(scores)
                metrics["robust_dice"][scenario_name] = mean_score
                
                if scenario_name == "clean":
                    metrics["clean_dice"] = mean_score

        metrics["fps"] = self.evaluate_fps(self.model, self.val_loader, self.device)
        return metrics

    def _log_metrics_to_wandb(self, epoch, train_loss, eval_metrics):
        if wandb.run is None:
            return

        clean_score = eval_metrics["clean_dice"]
        wandb_dict = {
            "Epoch": epoch + 1,
            "Loss/Train": train_loss,
            "Metrics/Clean_Dice": clean_score,
            "System/FPS": eval_metrics["fps"]["fps"],
            "System/Learning_Rate": self.optimizer.param_groups[0]['lr']
        }
        
        # Log sia dei punteggi assoluti che della percentuale di drop
        for scenario, score in eval_metrics['robust_dice'].items():
            if scenario != "clean":
                drop_pct = ((clean_score - score) / clean_score * 100) if clean_score > 0 else 0
                wandb_dict[f"Robustness_Dice/{scenario}"] = score
                wandb_dict[f"Robustness_Drop%/{scenario}"] = drop_pct
                
        wandb.log(wandb_dict)

    def _log_masks_to_wandb(self, x, y, preds, epoch, scenario_name):
        """Logga le maschere e una mappa degli errori visiva."""
        if wandb.run is None:
            return

        # Prendiamo solo la prima immagine del batch
        img_tensor = x[0].cpu().numpy()  
        gt_tensor = y[0].cpu().squeeze().numpy().astype(np.uint8)     
        pred_tensor = preds[0].cpu().squeeze().numpy().astype(np.uint8) 

        if img_tensor.shape[0] == 3:
            img_tensor = np.transpose(img_tensor, (1, 2, 0))
        elif img_tensor.shape[0] == 1:
            img_tensor = img_tensor.squeeze()

        # Calcolo della Error Map (XOR tra Ground Truth e Prediction)
        # Dove è 1, il modello ha sbagliato (Falso Positivo o Falso Negativo)
        error_map = np.abs(gt_tensor - pred_tensor).astype(np.uint8)

        class_labels = {
            0: "Background/Tissue",
            1: "Blood Area"
        }
        error_labels = {
            0: "Correct",
            1: "Error (FP/FN)"
        }

        wandb_img = wandb.Image(
            img_tensor,
            masks={
                "Ground Truth": {
                    "mask_data": gt_tensor,
                    "class_labels": class_labels
                },
                "Prediction": {
                    "mask_data": pred_tensor,
                    "class_labels": class_labels
                },
                "Error Map": { # NUOVO: La mappa degli errori rossi
                    "mask_data": error_map,
                    "class_labels": error_labels
                }
            },
            caption=f"Epoca {epoch+1} | Scenario: {scenario_name}"
        )

        wandb.log({f"Visuals/{scenario_name}": wandb_img}, commit=False)

    @staticmethod
    def evaluate_fps(model, loader, device, max_batches=20):
        """Calcola gli FPS per singola immagine, non per batch."""
        model.eval()
        use_cuda = device.type == "cuda"
        
        total_time_s = 0.0
        total_images = 0

        pbar = tqdm(loader, desc="Benchmarking FPS", leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for i, batch in enumerate(pbar):
                if i >= max_batches: break
                x = batch["image"].to(device)
                batch_size = x.shape[0]

                if i == 3 and use_cuda:  # Warmup
                    torch.cuda.synchronize()

                if use_cuda:
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    starter.record()
                    _ = model(x)
                    ender.record()
                    torch.cuda.synchronize()
                    elapsed = starter.elapsed_time(ender) / 1000.0
                else:
                    start = time.perf_counter()
                    _ = model(x)
                    end = time.perf_counter()
                    elapsed = end - start

                total_time_s += elapsed
                total_images += batch_size

        fps = total_images / total_time_s if total_time_s > 0 else 0
        mean_latency = total_time_s / total_images if total_images > 0 else 0
        
        return {"mean_latency_s": mean_latency, "fps": fps}

    @staticmethod
    def compute_dsc(pred, target, eps=1e-6):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return ((2. * intersection + eps) / (union + eps)).item()

    def _print_model_info(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("SURGICAL CO-PILOT - BENCHMARK ENGINE")
        print("=" * 60)
        print(f"Model Class:       {self.model.__class__.__name__}")
        print(f"Device:            {self.device}")
        print(f"Total Params:      {n_params:,}")
        print(f"Trainable Params:  {trainable:,}")
        print("=" * 60 + "\n")