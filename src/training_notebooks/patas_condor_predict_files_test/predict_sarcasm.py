"""
Performs model prediction using a saved Pytorch model file.

The command line for launching the script is:
predict_sarcasm.sh <validation set file path> <model file filepath> 

On Patas/Dryas these are:

/home2/droesl/573/balanced_validation_En.csv
/home2/droesl/573/test_model.pth


The script outputs the following files:
model_output.txt - list of probability predictions for each class
model_results.txt - sklearn classification report containing F1 scores

"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report


if __name__ == '__main__':

    validation_filepath = sys.argv[1]
    model_filepath = sys.argv[2]

    # read the validation CSV file
    validation_data = pd.read_csv(validation_filepath)
    X_val = validation_data['tweet']
    y_val = validation_data['sarcastic']

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

    # Create a function to tokenize a set of texts
    def preprocessing_for_bert(data):
        """Perform required preprocessing steps for pretrained BERT.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """
        input_ids = []
        attention_masks = []

        for sent in data:
            encoded_sent = tokenizer.encode_plus(
                text=sent,
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=MAX_LEN,                  # Max length to truncate/pad
                pad_to_max_length=True,         # Pad sentence to max length
                return_attention_mask=True      # Return attention mask
                )
            
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

    # Specify `MAX_LEN`
    MAX_LEN = 100

    # Run function `preprocessing_for_bert` on the validation set
    print('Tokenizing data...')
    val_inputs, val_masks = preprocessing_for_bert(X_val)

    # Convert other data types to torch.Tensor
    val_labels = torch.tensor(y_val.values)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=32)


    class BertClassifier(nn.Module):
        """Bert Model for Classification Tasks.
        """
        def __init__(self, freeze_bert=False):
            """
            @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
            """
            super(BertClassifier, self).__init__()
            # Specify hidden size of BERT, hidden size of our classifier, and number of labels
            D_in, H, D_out = 768, 50, 2

            # Instantiate BERT model
            self.bert = AutoModel.from_pretrained("vinai/bertweet-base")

            # Instantiate an one-layer feed-forward classifier
            self.classifier = nn.Sequential(
                nn.Linear(D_in, H),
                nn.ReLU(),
                nn.Dropout(0.33),
                nn.Linear(H, D_out)
            )

            # Freeze the BERT model
            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
            
        def forward(self, input_ids, attention_mask):
            """
            Feed input to BERT and the classifier to compute logits.
            @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                          max_length)
            @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                          information with shape (batch_size, max_length)
            @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                          num_labels)
            """
            # Feed input to BERT
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
            
            # Extract the last hidden state of the token `[CLS]` for classification task
            last_hidden_state_cls = outputs[0][:, 0, :]

            # Feed input to classifier to compute logits
            logits = self.classifier(last_hidden_state_cls)

            return logits

    def bert_predict(model, test_dataloader):
        """Perform a forward pass on the trained BERT model to predict probabilities
        on the test set.
        """
        # Put the model into the evaluation mode. 
        model.eval()

        all_logits = []

        for batch in test_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
            all_logits.append(logits)
        
        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)

        # Apply softmax to calculate probabilities
        probs = F.softmax(all_logits, dim=1).cpu().numpy()

        all_logits = all_logits.cpu().numpy()

        return probs, all_logits

    device = torch.device("cpu")
    model = torch.load(model_filepath, map_location=torch.device('cpu'))

    # Compute predicted probabilities on the validation set
    probs, all_logits = bert_predict(model, val_dataloader)

    # Get predictions from the probabilities
    preds = np.argmax(probs, axis = 1)


    with open("model_output.txt", mode="w", newline="\n", encoding="utf-8") as output_file:
        output_file.write(np.array2string(all_logits, precision=7, separator=','))

    with open("model_results.txt", mode="w", newline="\n", encoding="utf-8") as results_file:
        results_file.write(classification_report(y_val, preds))


