import os, json
import essentia.standard as es
from pathlib import Path


class AudioData:
    def __init__(self, sr=48000, resampleQuality=4):
        self.sr = sr
        self.resampleQuality = resampleQuality
    
    def load(self, audio_file):
        """
        Load audio data from a file.
        Args:
            audio_file (str): The path to the audio file.
        Returns:
            np.ndarray: The audio data as a NumPy array.
        """
        return es.MonoLoader(filename=audio_file, sampleRate=self.sr, resampleQuality = self.resampleQuality)()

class MetaData : 
    def __init__(self, json_path):
        self.json_path = json_path
        self.loaded = False
        self.schema = None
        self.data = self.load_data(self.json_path) if self.json_path else ValueError("Json path must be provided.")

    def load_data(self, json_path):
        if(self.loaded):
            return self.data
        with open(json_path, 'r') as json_file:
            self.data = json.load(json_file)
            self.schema = self.get_schema()
            self.loaded = True
            return self.data
        
    def get_metadata(self):
        """
        Get the metadata as a dictionary.
        Returns:
            dict: The metadata as a dictionary.
        """
        return self.data
    
    def get_schema(self):
        """
        Get the schema from the metadata.
        Returns:
            dict: The schema as a dictionary.
        """
        self.schema = self.data['schema']
        return self.schema
    
    def get_layer(self, layer_name, output_purpose="predictions"):
        """
        Get the name of a layer in the schema.
        Args:
            layer_name (str): The name of the layer to be returned. The layer name can be either 'input' or 'output'.
            output_purpose (str): The purpose of the output layer. The output layer can be either 'predictions' or 'embeddings'.
        Returns:
            str: The name of the layer.
        """
        if not layer_name or not self.schema:
            raise ValueError("You must provide a layer name. The layer name must be either 'input' or 'output'.")
        
        # inputs in schema do not have a purpose, so we return the first input layer                       
        return self.schema['inputs'][0]['name'] if layer_name == 'input' else \
            next((layer['name'] for layer in self.schema['outputs'] if layer['output_purpose'] == output_purpose), None) \
            or ValueError(f"Could not find output layer with purpose: {output_purpose}")        

class Model:
    def __init__(self, graph_path, input_layer="", output_layer = ""):
        self.audio = AudioData()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.graph_path = Path(graph_path)
        self.predictions = None
    
    def get_audio_or_embeddings(self, audio_file=None, embeddings=None):
        """
        Check and get the audio data or embeddings.
        Args:
            audio_file (str or None): The path to the audio file. If None, embeddings must be provided.
            embeddings (np.ndarray or None): The embeddings as a NumPy array. If None, audio_file must be provided.
        Returns:
            np.ndarray or None: The audio data as a NumPy array, or None if embeddings are provided.
        """
        # embeddings.all() == None ?
        if(audio_file == None and embeddings == None):
            raise ValueError("You must provide an audio file or a set of embeddings.")
        return embeddings if embeddings is not None else self.audio.load(audio_file)
   
    def evaluate(self, metadata):
        """
        Evaluate the model predictions.
        Args:
            metadata (dict): The metadata as a dictionary.
        Returns:
            dict: The evaluation statistics as a dictionary.
        """
        stats = {}
        for label, probability in zip(metadata['classes'], self.predictions.mean(axis=0)):
            stats[label] = f'{100 * probability:.1f}%'
        return stats
    
class Processor:
    
    def __init__(self, audio_file_path):
        self.audio_file_path = audio_file_path
        self.audio_files = self.create_list(self.audio_file_path)
    
    def create_list(self, path):
        audio_files = []
        if(os.path.isdir(path)):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if(file.endswith(".wav") or file.endswith(".aiff") or file.endswith(".aif") 
                    or file.endswith(".mp3") or file.endswith(".ogg") or file.endswith(".flac")):
                        audio_files.append(os.path.join(root, file))
        else:
            audio_files.append(path)
        
        return audio_files
    
    def classify_audio(self, model, classifier_metadata, embedding_model=None):
        stats = {}
        for audio in self.audio_files:
            # create a new instance of the model
            model = model.__class__(model.graph_path, model.input_layer, model.output_layer)
            embeddings = None
            if embedding_model is not None:
                embedding_model = embedding_model.__class__(embedding_model.graph_path, embedding_model.input_layer, embedding_model.output_layer)
                embeddings = embedding_model.compute(audio_file=audio)

            model.compute(audio_file=audio, embeddings=embeddings)
            stat = model.evaluate(classifier_metadata.data)
            file_name = os.path.basename(audio)
            stats[file_name] = stat
            for label in stat:
                print(file_name, " Label: ", label, " Probability: ", stat[label]) 
        return stats
    
    def export_data(self, stats, stats_folder, stats_parent):
        import subprocess
        folder_name = os.path.basename(os.path.dirname(self.audio_file_path))
        # store one the file names so that subprocess can open the directory in Finder
        data_file = ''
        
        for audio in self.audio_files:
            file_name = os.path.basename(audio)
            file_name_without_ext = os.path.splitext(file_name)[0]
            entry_label = file_name
            entry_values = stats[file_name]
            hierarchy = {stats_folder: {stats_parent: entry_values}}
            
            data_file = os.path.join(os.path.dirname(audio), file_name_without_ext+'_analysis.json')
            
            # check to see if analysis file already exists
            if os.path.isfile(data_file):
                with open(data_file, 'r') as json_file:
                    data = json.load(json_file)

                data.setdefault('classifications', {}).setdefault(stats_folder, {})[stats_parent] = entry_values

                with open(data_file, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
            else:
                data = {'classifications': hierarchy}
                with open(data_file, 'w') as json_file:
                    json.dump(data, json_file, indent=4)

            
        # opens the directory in Finder
        subprocess.run(['open', '-R', data_file])

# need to install tkinter using brew install python-tk@3.9 or pip install tk
def file_dialog():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    dialog = filedialog.askdirectory()
    dialog.wait_for_closed()
    return dialog.selected_path

       
        
