from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model, Model
from barriernn_p.input_generation import create_meta_dataset_predictions
from tqdm.autonotebook import tqdm

#%% ----- Parameters -----
#########
datasets = ['dataset_2208_synth', 'dataset_2208_traj']
base_name_dir = "2208_full_new_enc"
model_glob = "*"
#########

data_root = Path("/hits/basement/mbm/riedmiki/structures/KR0008/")
model_root = Path("/hits/fast/mbm/riedmiki/nn/barrier_gnn_out/logs_2.1")

#%% ----- Load models -----
if base_name_dir is not None:
    model_root = model_root / base_name_dir

model_dirs = np.array(list(model_root.glob(f"{model_glob}/*.tf")))
model_names = np.array([m.parent.name for m in model_dirs])
model_types = np.array([m.parent.name.split("_")[0] for m in model_dirs])

models = []
for model_dir in tqdm(model_dirs, "Loading model"):
    model: Model = load_model(model_dir)
    models.append(model)
models = np.array(models)
print("Models loaded!")

#%% ----- Load Datasets, make predictions -----
test_files = [
    i for d in datasets for i in data_root.glob(f"{d}/test/[0-9]*_*[0-9].npz")
]
assert len(test_files) != 0, "Dataset not recognized, searching meta files.."
print(f"Found {len(test_files)} meta files")

print("Loading Data..")
test_ds, energies, scale_t, meta_ds, metas_masked = create_meta_dataset_predictions(
    meta_files=test_files,
    batch_size=128,
    opt=False
)
print("Data loaded!")

predictions = []
for model in tqdm(models, "Predicting"):
    predictions.append(model.predict(test_ds).squeeze())
predictions = np.array(predictions)

print("Done, mae of all test data:")
mae = np.mean(np.abs(predictions - energies))
print(f'MAE = {mae:.2f}')

#%%
# Masking
type_map = np.unique(model_types)
type_broad = np.broadcast_to(type_map, (len(models), len(type_map)))
type_masks = np.apply_along_axis(lambda t: model_types == t, 0, type_broad).T

