# Person-Search using SigLIP

## Setup

1. Clone the repository

```bash
git clone https://github.com/hungphongtrn/PERSON_RLF.git
```

2. Install uv package manager and sync the dependencies

```bash
cd PERSON_RLF
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

3. Download the `siglip-base-patch16-256-multilingual` checkpoints

```bash
uv run prepare_checkpoints.py
```

4. Put the CUHK-FULL dataset in the root folder
   Here is the sample structure of the project

```bash
.
|-- clip_checkpoints
|-- config
|-- CUHK-PEDES          # This is the dataset folder for CUHK-PEDES
|-- VN3K                # This is the dataset folder for VN3K
|-- data
|-- experiments
|-- lightning_data.py
|-- lightning_models.py
|-- model
|-- m_siglip_checkpoints
|-- outputs
|-- prepare_checkpoints.py
|-- __pycache__
|-- pyproject.toml
|-- README.md
|-- requirements.txt
|-- run.sh
|-- siglip_checkpoints
|-- solver
|-- trainer.py
|-- utils
|-- uv.lock
|-- ...
```

5. Log in to the Weights & Biases

```bash
uv run wandb login <API_KEY>
```

### Run the experiments

1. CUHK-FULL dataset

```bash
# With m-SigLIP
# Run the training with TBPS method
uv run trainer.py -cn m_siglip img_size_str="'(256,256)'" dataset=cuhk_pedes dataset.sampler=random loss.softlabel_ratio=0.0 trainer.max_epochs=60 optimizer=tbps_clip_no_decay optimizer.param_groups.default.lr=1e-5
# Run the training with IRRA method
uv run trainer.py -cn m_siglip img_size_str="'(256,256)'" dataset=cuhk_pedes dataset.sampler=identity dataset.num_instance=1 loss=irra loss.softlabel_ratio=0.0 trainer.max_epochs=60 optimizer=irra_no_decay optimizer.param_groups.default.lr=1e-5
```

### Federated Learning Loss Functions

Des fonctions de perte pour le Federated Learning ont été ajoutées dans `model/federated_losses.py` :

```python
from model.federated_losses import (
    binary_cross_entropy_sigmoid,
    federated_bce_loss,
    federated_focal_loss,
    triplet_loss,
    contrastive_loss,
)

# Utilisation de BCELoss pour classification binaire
logits = torch.randn(100)
labels = torch.randint(0, 2, (100,)).float()
loss = binary_cross_entropy_sigmoid(logits, labels)

# Utilisation de Focal Loss pour données déséquilibrées
loss = federated_focal_loss(logits, labels, alpha=0.25, gamma=2.0)

# Triplet Loss pour REID
embeddings = torch.randn(64, 128)
labels = torch.randint(0, 2, (64,))
loss, metrics = triplet_loss(embeddings, labels)

# BCELoss avec métriques
loss, metrics = federated_bce_loss(logits, labels)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Structure du projet pour FL

```
PERSON_RLF/
├── model/
│   ├── federated_losses.py    # Nouvelles fonctions FL
│   ├── objectives.py          # Pertes SigLIP
│   └── reid_objectives.py     # Pertes REID
├── solver/
│   └── lr_scheduler.py        # Learning rate schedules
└── utils/
    └── metrics.py             # Métriques
```
