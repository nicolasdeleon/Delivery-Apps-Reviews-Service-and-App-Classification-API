import transformers
import torch
from torch import nn

class BianryModelInterface:

    def __init__(self, model_bin, device, tokenizer_state, cat):
        self.model = ReviewClassifier(n_classes=1, pre_trained_model='dccuchile/bert-base-spanish-wwm-cased')
        self.model_bin = model_bin
        self.device = device
        self.tokenizer = transformers.BertTokenizer.from_pretrained('tokenizer_state')
        self.classes = [f'Not {cat}', cat]

    def model_ramp_up(self):
        """This loads the model to the cpu or the device. Takes time"""
        self.model.load_state_dict(torch.load(self.model_bin, map_location=torch.device(self.device)))

    def __str__(self):
        # TODO: Print all necessesary things
        print(f'Mounted on: {self.device}')
        print('Model architecture:')
        print(self.model)

    def predict(self, txt):
        encoded_review = self.tokenizer.encode_plus(
            txt,
            max_length=70,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded_review['input_ids'].to(torch.device(self.device))
        attention_mask = encoded_review['attention_mask'].to(torch.device(self.device))
        self.model.eval()
        output = self.model(input_ids, attention_mask)
        prediction = torch.round(output)
        return {
            'input': txt,
            'score': output.tolist()[0][0],
            'category': self.classes[int(prediction.tolist()[0][0])]
        }

class ReviewClassifier(nn.Module):

    def __init__(self, n_classes, pre_trained_model):
        super(ReviewClassifier, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(pre_trained_model)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes) # = Dense en keras
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.sigmoid(self.out(output))

"""
model_service = BianryModelInterface(
    model_bin='binary_service_model.bin',
    device='cpu',
    tokenizer_state='tokenizer_state',
    cat='Service'
)"""

model_product = BianryModelInterface(
    model_bin='binary_product_model.bin',
    device='cpu',
    tokenizer_state='tokenizer_state',
    cat='Product'
)
"""
model_product = BianryModelInterface(
    model_bin='binary_support_model.bin',
    device='cpu',
    tokenizer_state='tokenizer_state',
    cat='Support'
)"""

# model_service.model_ramp_up()
model_product.model_ramp_up()

import time

start = time.time()
print(
 #    model_service.predict(
  #   "Se traba todo el tiempo. No te deja registrarte con facebook ni google"
  #   ),
    model_product.predict(
    "Se traba todo el tiempo. No te deja registrarte con facebook ni google"
   )
  #  model_product.predict(
  #  "Se traba todo el tiempo. No te deja registrarte con facebook ni google"
  #  )
)
end = time.time()

print(end - start)
