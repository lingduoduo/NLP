# Databricks notebook source
# MAGIC %md
# MAGIC This notebook logs a HuggingFace model with an input example and a model signature and registers it to the Databricks Model Registry.
# MAGIC
# MAGIC After you run this notebook in its entirety, you have a registered model for model serving with Databricks Model Serving ([AWS](https://docs.databricks.com/machine-learning/model-serving/index.html)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/index.html)).

# COMMAND ----------

import transformers
import mlflow
import torch
tokenizer = transformers.BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True, cache_dir='tokenizer_cache')

# COMMAND ----------

class AugmentedBert(torch.nn.Module):
    def __init__(self, output_class_len, base_model, cache_dir, hidden_dim=64):
        super().__init__()
        self.bert_model = transformers.AutoModel.from_pretrained(base_model, cache_dir=cache_dir)
        self.emb_dim = 768
        self.fc1 = torch.nn.Linear(self.emb_dim, self.emb_dim)
        self.tanh = torch.nn.Tanh()
        self.gelu = torch.nn.GELU()

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        output = bert_output["last_hidden_state"][:, 0, :]
        output = self.fc1(output)
        output = self.tanh(output)
        output = self.gelu(output)
        return output


# COMMAND ----------

model = AugmentedBert(10, "vinai/bertweet-base", "model_cache")
with mlflow.start_run():
  mlflow.pytorch.log_model(model, 'pytorch-model', registered_model_name='bert-encoder-pytorch')

# COMMAND ----------

from torch.utils.data import DataLoader
class SampleDatasetWithEncodings(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]).clone().detach()
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def create_data_loader(tokenizer, X, y=None, batch_size=1, input_max_len=64):
    features = tokenizer(
        X,
        max_length=input_max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    if y is not None:
        dataset = SampleDatasetWithEncodings(features, y)
    else:
        dataset = SampleDatasetWithEncodings(features, [0] * features.get("input_ids").shape[0])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# COMMAND ----------

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import os
model_name = 'bert-encoder-pytorch'
model_uri = f"models:/{model_name}/1"
if not os.path.exists('/databricks/driver/pytorch-model-artifacts'):
  os.makedirs('/databricks/driver/pytorch-model-artifacts')
local_path = ModelsArtifactRepository(model_uri).download_artifacts("", dst_path="/databricks/driver/pytorch-model-artifacts") # download model from remote registry
print(local_path)

# COMMAND ----------

import os
import mlflow
import torch
import pandas as pd
import transformers

class ModelPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = torch.load(context.artifacts["torch-weights"])
        self.tokenizer = transformers.BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True, local_files_only=True, cache_dir=context.artifacts["tokenizer_cache"])

    def format_inputs(self, model_input):
        if isinstance(model_input, str):
            model_input = [model_input]
        if isinstance(model_input, pd.Series):
            model_input = model_input.tolist()
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.iloc[:, 0].tolist()
        return model_input

    def prepare_data(self, tokenizer, model_input):
        data_loader = create_data_loader(
            tokenizer,
            model_input
        )
        return data_loader.dataset.encodings

    def format_outputs(self, outputs):
        predictions = (torch.sigmoid(outputs)).data.numpy()
        classes = [
            "class1",
            "class2",
            "class3",
            "class4",
            "class5",
            "class6",
        ]
        return [dict(zip(classes, prediction)) for prediction in predictions]

    def predict(self, context, model_input):
        model_input = self.format_inputs(model_input)
        processed_input = self.prepare_data(self.tokenizer, model_input)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=processed_input.get("input_ids"),
                attention_mask=processed_input.get("attention_mask"),
            )
        return self.format_outputs(outputs)


with mlflow.start_run() as run:
    model = ModelPyfunc()
    mlflow.pyfunc.log_model(
      "model",
      python_model=model,
      artifacts={'torch-weights': "./pytorch-model-artifacts/data/model.pth", "tokenizer_cache": "./tokenizer_cache"},
      input_example=["this is a test", "this is a second test"],
      registered_model_name='bert-encoder'
    )
    
