import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
# Make sure to import your actual TransformerBlock
from alchemist_transformerBlock import TransformerBlock

# Load and preprocess the data
with open('dataset.json', 'r') as file:
    data = json.load(file)
    # Adjust based on the actual format of your JSON
    dataset = [entry for entry in data.values()]

# Preprocess the data to extract prompts and responses
prompts = [' '.join(entry['topics']) + ' ' + entry['intent']
           for entry in dataset]
responses = [entry['response']['initial'] for entry in dataset]

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class TextGenerationDataset(Dataset):
    def __init__(self, prompts, responses, tokenizer, max_length):
        self.prompts = prompts
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # Tokenize and encode the prompt and response texts
        encoded_pair = tokenizer.encode_plus(
            self.prompts[idx],
            self.responses[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        return {
            'input_ids': torch.tensor(encoded_pair['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded_pair['attention_mask'], dtype=torch.long),
            # Labels are the input_ids for language modeling
            'labels': torch.tensor(encoded_pair['input_ids'], dtype=torch.long)
        }


# Instantiate dataset and dataloader
max_length = 512  # Maximum sequence length
train_dataset = TextGenerationDataset(
    prompts, responses, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define the Transformer model architecture


class TextGeneratorModel(nn.Module):
    def __init__(self, transformer_block, tokenizer, num_transformer_blocks):
        super(TextGeneratorModel, self).__init__()
        self.tokenizer = tokenizer
        self.transformer_blocks = nn.ModuleList([
            transformer_block for _ in range(num_transformer_blocks)
        ])
        self.vocab_size = len(tokenizer.vocab)
        self.output_layer = nn.Linear(
            transformer_block.hidden_size, self.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = input_ids
        for block in self.transformer_blocks:
            x = block(x)

        logits = self.output_layer(x)
        return logits


# Initialize the model
hidden_size = 768
num_attention_heads = 12
num_embeddings = tokenizer.vocab_size
transformer_block = TransformerBlock(
    hidden_size, num_attention_heads, num_embeddings)
model = TextGeneratorModel(
    transformer_block, tokenizer, num_transformer_blocks=1)

# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * 3)  # Adjust epochs accordingly


# Training loop
def train(model, dataloader, optimizer, criterion, scheduler, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask)
            logits = outputs.view(-1, model.vocab_size)
            loss = criterion(logits, labels.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f'Epoch {epoch} completed. Loss: {loss.item()}')


# Assume we have the same training dataloader
epochs = 3  # Define the number of epochs
train(model, train_dataloader, optimizer, criterion, scheduler, epochs)

# Save the model after training
torch.save(model.state_dict(), 'text_generator_model.pth')
