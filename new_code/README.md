## 🚀 Running the Project

### 🏋️‍♂️ Train the Model

Run the following command to train the depth estimation model:

```bash
python3 new_model.py \
  --image_dir [PATH TO RGB TRAINING IMAGES] \
  --depth_dir [PATH TO TRAINING DEPTH CHART IMAGES] \
  --epochs [NUMBER OF EPOCHS] \
  --batch_size [BATCH SIZE]

python3 eval_model.py \
  --model_path [PATH TO SAVED MODEL] \
  --image_dir [PATH TO RGB EVAL IMAGES] \
  --depth_dir [PATH TO EVAL DEPTH CHART IMAGES] \
  --output_dir [PATH TO RESULTS DIRECTORY]

