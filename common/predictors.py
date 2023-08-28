import numpy as np
import essentia.standard as es
from essentia.standard import TensorflowPredict
from essentia import Pool
from common.base import Model
from common.mel_spectrogram import MelSpectrogramOpenL3

class Tensorflow2D(Model):
    def __init__(self, graph_path, input_layer = "", output_layer = ""):
       super().__init__(graph_path, input_layer, output_layer)
               
    def compute(self, audio_file=None, embeddings=None):
        data = self.get_audio_or_embeddings(audio_file, embeddings)
        model = es.TensorflowPredict2D(
            graphFilename=str(self.graph_path),
            input=self.input_layer,
            output=self.output_layer
        )
        self.predictions = model(data)
        return self.predictions


class TensorflowFSDSINet(Model):
    def __init__(self, graph_path,input_layer = "", output_layer = ""):
         super().__init__(graph_path, input_layer, output_layer)
               
    def compute(self, audio_file=None, embeddings=None):
        data = self.get_audio_or_embeddings(audio_file, embeddings)
        model = es.TensorflowPredictFSDSINet(
            graphFilename=str(self.graph_path),
            input=self.input_layer,
            output=self.output_layer
        )
        self.predictions = model(data)
        return self.predictions


class TensorflowVGGish (Model):
    def __init__(self, graph_path, json_path, input_layer = "", output_layer = ""):
        super().__init__(graph_path=graph_path, json_path=json_path, input_layer=input_layer, output_layer=output_layer)
    
    def compute(self, audio_file):
        
        audio = self.audio.compute(audio_file)
        model = es.TensorflowPredictVGGish(
            graphFilename=str(self.graph_path),
            inputs=[self.input_layer],
            outputs=[self.output_layer]
        )
        return model(audio)



class OpenL3(Model):
    def __init__(self, graph_path, input_layer="", output_layer="", hop_time=1, batch_size=60, melbands=128):        
        super().__init__(graph_path, input_layer, output_layer)
        self.hop_time = hop_time
        self.batch_size = batch_size
        self.x_size = 199
        self.y_size = melbands
        self.squeeze = False
        self.permutation = [0, 3, 2, 1]
    
        self.mel_extractor = MelSpectrogramOpenL3(hop_time=self.hop_time, sr=self.audio.sr)

        self.model = TensorflowPredict(
            graphFilename=str(self.graph_path),
            inputs=[self.input_layer],
            outputs=[self.output_layer],
            squeeze=self.squeeze,
        )

    def compute(self, audio_file):
        audio = self.audio.load(audio_file)
        mel_spectrogram = self.mel_extractor.compute(audio)
        # in OpenL3 the hop size is computed in the feature extraction level
        hop_size_samples = self.x_size

        batch = self.__melspectrogram_to_batch(mel_spectrogram, hop_size_samples)

        pool = Pool()
        embeddings = []
        nbatches = int(np.ceil(batch.shape[0] / self.batch_size))
        for i in range(nbatches):
            start = i * self.batch_size
            end = min(batch.shape[0], (i + 1) * self.batch_size)
            pool.set(self.input_layer, batch[start:end])
            out_pool = self.model(pool)
            embeddings.append(out_pool[self.output_layer].squeeze())

        return np.vstack(embeddings)

    def __melspectrogram_to_batch(self, melspectrogram, hop_time):
        npatches = int(np.ceil((melspectrogram.shape[0] - self.x_size) / hop_time) + 1)
        batch = np.zeros([npatches, self.x_size, self.y_size], dtype="float32")
        for i in range(npatches):
            last_frame = min(i * hop_time + self.x_size, melspectrogram.shape[0])
            first_frame = i * hop_time
            data_size = last_frame - first_frame

            # the last patch may be empty, remove it and exit the loop
            if data_size <= 0:
                batch = np.delete(batch, i, axis=0)
                break
            else:
                batch[i, :data_size] = melspectrogram[first_frame:last_frame]

        batch = np.expand_dims(batch, 1)
        batch = es.TensorTranspose(permutation=self.permutation)(batch)
        return batch

