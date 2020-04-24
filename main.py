import os
from scipy.io.wavfile import write
from config import ESPNetConfig
import sys
import torch
from argparse import Namespace
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.utils.dynamic_import import dynamic_import
import numpy as np
from tacotron_cleaner.cleaners import custom_english_cleaners
from g2p_en import G2p
import yaml
import parallel_wavegan.models
import nltk
import time
import argparse


esp_config = ESPNetConfig.from_json_file("config.json")

# add path
sys.path.append("espnet/egs/ljspeech/tts1/local")
sys.path.append("espnet")

# define device
device = torch.device(esp_config.device)


idim, odim, train_args = get_model_conf(esp_config.model_path)
model_class = dynamic_import(train_args.model_module)
model = model_class(idim, odim, train_args)
torch_load(esp_config.model_path, model)
model = model.eval().to(device)
inference_args = Namespace(**{
    "threshold": 0.5, "minlenratio": 0.0, "maxlenratio": 10.0,
    "use_attention_constraint": True,
    "backward_window": 1, "forward_window": 3,
})

# define neural vocoder
with open(esp_config.vocoder_conf) as f:
    config = yaml.load(f, Loader=yaml.Loader)
vocoder_class = config.get("generator_type", "ParallelWaveGANGenerator")
vocoder = getattr(parallel_wavegan.models, vocoder_class)(**config["generator_params"])
vocoder.load_state_dict(torch.load(esp_config.vocoder_path, map_location="cpu")["model"]["generator"])
vocoder.remove_weight_norm()
vocoder = vocoder.eval().to(device)

# define text frontend

with open(esp_config.dict_path) as f:
    lines = f.readlines()
lines = [line.replace("\n", "").split(" ") for line in lines]
char_to_id = {c: int(i) for c, i in lines}
g2p = G2p()


def frontend(text):
    """Clean text and then convert to id sequence."""
    text = custom_english_cleaners(text)

    if esp_config.trans_type == "phn":
        text = filter(lambda s: s != " ", g2p(text))
        text = " ".join(text)
        print(f"Cleaned text: {text}")
        charseq = text.split(" ")
    else:
        print(f"Cleaned text: {text}")
        charseq = list(text)
    idseq = []
    for c in charseq:
        if c.isspace():
            idseq += [char_to_id["<space>"]]
        elif c not in char_to_id.keys():
            idseq += [char_to_id["<unk>"]]
        else:
            idseq += [char_to_id[c]]
    idseq += [idim - 1]  # <eos>
    phonemes = charseq + ["<eos>"]
    return torch.LongTensor(idseq).view(-1).to(device), phonemes

nltk.download('punkt')
print("Now ready to synthesize!")


parser = argparse.ArgumentParser()
parser.add_argument(
        "--input_text",
        default=None,
        type=str,
        required=True
    )

args = parser.parse_args()

input_text = args.input_text
pad_fn = torch.nn.ReplicationPad1d(config["generator_params"].get("aux_context_window", 0))
use_noise_input = vocoder_class == "ParallelWaveGANGenerator"
with torch.no_grad():
    start = time.time()
    x, phonemes = frontend(input_text)
    c, d_out, _ = model.inference(x, inference_args)
    durations = d_out.cpu().squeeze(0).numpy().tolist()
    c = pad_fn(c.unsqueeze(0).transpose(2, 1)).to(device)
    xx = (c,)
    if use_noise_input:
        z_size = (1, 1, (c.size(2) - sum(pad_fn.padding)) * config["hop_size"])
        z = torch.randn(z_size).to(device)
        xx = (z,) + xx
    y = vocoder(*xx).view(-1)
rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
print(f"RTF = {rtf:5f}")

process_time = (time.time() - start)
print(f"process_time = {process_time}")

audio_duration = (len(y)/config["sampling_rate"])*1000

unit_duration = audio_duration/sum(durations)

ends = np.cumsum(durations) * unit_duration
starts = [0] + ends[:-1].tolist()

with open(esp_config.phonemes_file, 'w') as file_writer:
    for phoneme, start, end in zip(phonemes, starts, ends):
        file_writer.write(phoneme + "\t" + str(start) + "\t" + str(end) + "\n")

write(esp_config.voice_file, config["sampling_rate"], y.view(-1).cpu().numpy())