# Ditto avatar service (WSL2 or Linux GPU)

Ditto's CUDA and TensorRT stack is built and tested on Linux, so run it in WSL2 on Windows. Anything listening in WSL2 is reachable from Windows at `localhost`.

## 1. WSL2 with GPU (Windows, once)

```powershell
wsl --install -d Ubuntu-22.04     # reboot afterwards, then open "Ubuntu" from the Start menu
```

Install the normal **Windows** NVIDIA driver only. Don't install a Linux driver inside WSL. Check from Ubuntu:

```bash
nvidia-smi
sudo apt update && sudo apt install -y git git-lfs ffmpeg python3-pip python3-venv
```

## 2. Ditto

```bash
cd ~
git clone https://github.com/antgroup/ditto-talkinghead
cd ditto-talkinghead
# Create the environment exactly as the repo README describes (it ships a conda environment file).
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```

Test Ditto on its own first, using any wav and a front-facing portrait:

```bash
python inference.py \
  --data_root ./checkpoints/ditto_pytorch \
  --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl \
  --audio_path ./example/audio.wav --source_path ./example/image.png --output_path ./tmp/result.mp4
```

The checkpoint folder names differ between the PyTorch and TensorRT builds; use whatever the README and the `checkpoints` folder show. TensorRT is faster but must match your GPU generation.

## 3. Run the service

Use the Python interpreter from Ditto's environment for `DITTO_PYTHON`. The service itself only needs FastAPI.

```bash
cd /mnt/c/<path-to>/p1-avatar/ditto_service      # or copy this folder into WSL for faster disk access
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export DITTO_REPO=~/ditto-talkinghead
export DITTO_PYTHON=~/miniconda3/envs/ditto/bin/python    # adjust to your Ditto env
export DITTO_SOURCE_IMAGE=~/avatar.png
uvicorn server:app --host 0.0.0.0 --port 8090
```

From Windows: `curl http://localhost:8090/health` should return `{"status":"ok"}`. If it returns a 503, the message says what's missing.

## 4. Measure

Note how many seconds it takes to render 5 seconds of audio. Faster than real time means a streaming avatar (LiveKit) is realistic on your GPU. Slower means you should stay with the turn-based kiosk for the demo.
