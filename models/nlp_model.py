import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class NLPModelWrapper(nn.Module):
    """
    A simple wrapper around HuggingFace Sequence Classification models.
    This ensures compatibility with our existing train.py loop which expects
    model(inputs) -> logits, rather than model(**inputs) -> OutputObj.
    """
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        super(NLPModelWrapper, self).__init__()
        print(f"Initializing {model_name} for sequence classification...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, inputs):
        """
        Our dataloader yields (batch_dict, targets). 
        However, train.py does: `outputs = model(inputs)` 
        We need to make sure `inputs` is a dict of tensors (input_ids, attention_mask),
        and we return just the logits.
        """
        # If inputs is a tuple (input_ids, attention_mask) or a dict
        if isinstance(inputs, dict):
            outputs = self.model(**inputs)
        else:
            # Fallback if the loader formats it as a list/tuple
            input_ids, attention_mask = inputs[0], inputs[1]
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        return outputs.logits

