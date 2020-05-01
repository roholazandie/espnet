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
from flask import Flask, request, session, send_file, Response, after_this_request
import uuid
import ast

app = Flask(__name__)

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


cmu_phonemes = ["F", "M", "N", "L", "D", "B", "HH", "P", "T", "S", "R", "AE", "W", "Z", "V", "G", "NG", "DH", "AX",
                "AA", "AH", "AO", "AW", "AXR", "AY", "CH", "EH", "ER", "EY", "IH", "IX", "IY", "JH", "OW", "OY", "SH",
                "TH", "UH", "UW", "Y", "TS", "R", "R", "AH", "AA", "SIL", "IY", "L", "L", "R", "IH", ]

def regulate_phoneme_duration(phoneme, start, end):
    for char in ['0', '1', '2', '3']:
        if char in phoneme:
            phoneme = phoneme.replace(char, '')
    if phoneme not in cmu_phonemes:
        phoneme = "SIL"

    start = int(float(start) / 10) + 10
    end = int(float(end) / 10) + 10
    return phoneme, start, end



def tts(input_text, out_file_name):
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

    audio_duration = (len(y) / config["sampling_rate"]) * 1000

    unit_duration = audio_duration / sum(durations)

    ends = np.cumsum(durations) * unit_duration
    starts = [0] + ends[:-1].tolist()

    lines = []
    phoneme_out = {"phonemes": [], "start": [], "end": []}
    phonemes_file = os.path.join(esp_config.phonemes_dir, out_file_name+".txt")
    with open(phonemes_file, 'w') as file_writer:
        for phoneme, start, end in zip(phonemes, starts, ends):
            phoneme, start, end = regulate_phoneme_duration(phoneme, start, end)
            line = "{:4d} 0    0    0    0  {:4d} {:4s} 0.0000 ".format(start, end, phoneme) +'\n'
            file_writer.write(line)
            lines.append(line)
            phoneme_out["phonemes"].append(phoneme)
            phoneme_out["start"].append(start)
            phoneme_out["end"].append(end)

    wav_file = os.path.join(esp_config.voice_dir, out_file_name+".wav")
    write(wav_file, config["sampling_rate"], y.view(-1).cpu().numpy())

    return {"phonemes": " ".join(lines)}



@app.route('/api/tts', methods=['POST'])
def tts_api():
    data = ast.literal_eval(request.data.decode("utf-8"))
    unique_name = str(uuid.uuid4())
    response = tts(data["input_text"], out_file_name=unique_name)
    response["filename"] = unique_name
    return response


@app.route('/api/download', methods=['POST', 'GET'])
def download():
    try:
        data = ast.literal_eval(request.data.decode("utf-8"))
        wav_file = os.path.join(esp_config.voice_dir, data["filename"]+".wav")
        phoneme_file = os.path.join(esp_config.phonemes_dir, data["filename"] + ".txt")
        @after_this_request
        def add_header(response):
            os.remove(wav_file)
            os.remove(phoneme_file)
            return response

        return send_file(wav_file)

        # data = ast.literal_eval(request.data.decode("utf-8"))
        # wav_file = os.path.join(esp_config.voice_dir, data["filename"]+".wav")
        #
        # return_data = io.BytesIO()
        # with open(wav_file, 'rb') as fo:
        #     return_data.write(fo.read())
        # # (after writing, cursor will be at last byte, so move it to start)
        # return_data.seek(0)
        #
        # os.remove(wav_file)
        #
        # return send_file(return_data)






    except Exception as e:
        return str(e)

@app.route('/api/delete', methods=['POST', 'GET'])
def delete_api():
    try:
        data = ast.literal_eval(request.data.decode("utf-8"))
        os.remove(os.path.join(esp_config.voice_dir, data["filename"]+".wav"))
        os.remove(os.path.join(esp_config.phonemes_dir, data["filename"]+".txt"))

        return "success"
    except Exception as excep:
        return str(excep)


@app.route('/api/delete_all')
def delete_all_api():
    try:
        wav_filelist = [f for f in os.listdir(esp_config.voice_dir) if f.endswith(".wav")]
        phonemes_filelist = [f for f in os.listdir(esp_config.phonemes_dir) if f.endswith(".txt")]
        for f in wav_filelist:
            os.remove(os.path.join(esp_config.voice_dir, f))

        for f in phonemes_filelist:
            os.remove(os.path.join(esp_config.phonemes_dir, f))

        return "success"
    except Exception as excep:
        return str(excep)


# import json
# from requests_toolbelt import MultipartEncoder
#
# @app.route('/api/download', methods=['GET', 'POST'])
# def download():
#     m = MultipartEncoder({"document": "this is a test", "file": ('filename', open('/home/rohola/codes/espnet/outputs/wav_files/0e550b42-9540-4761-a582-76a0925d8663.wav', 'rb'), 'text/plain')})
#     return Response(m.to_string(), mimetype=m.content_type)


if __name__ == '__main__':
    app.secret_key = 'fhcbnmblhsadf7ew8qw4q'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True, port=esp_config.port)

# ask a question
#curl --header "Content-Type: application/json" --request POST --data '{"input_text":"This is an awesome example."}' http://localhost:3333/api/tts

# download
# curl -o output.wav --header "Content-Type: application/json" --request POST --data '{"filename":"a1b8f97d-97c5-40c8-acd4-cbadca67abdf"}' http://localhost:3333/api/download

#delete
#curl --header "Content-Type: application/json" --request POST --data '{"filename":"a1b8f97d-97c5-40c8-acd4-cbadca67abdf"}' http://localhost:3333/api/delete

# delete all
#curl --header "Content-Type: application/json" http://localhost:3333/api/delete_all
