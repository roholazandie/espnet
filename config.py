import json


class ESPNetConfig:

    def __init__(self,
                 trans_type="",
                 dict_path="",
                 model_path="",
                 vocoder_path="",
                 vocoder_conf="",
                 phonemes_file="",
                 voice_file="",
                 device="cpu",
                 ):
        self.trans_type = trans_type
        self.dict_path = dict_path
        self.model_path = model_path
        self.vocoder_path = vocoder_path
        self.vocoder_conf = vocoder_conf
        self.phonemes_file = phonemes_file
        self.voice_file = voice_file
        self.device = device

    @classmethod
    def from_dict(cls, json_object):
        config = ESPNetConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))

    def __str__(self):
        return str(self.__dict__)