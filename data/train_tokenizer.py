# data/tokenizer/train_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.processors import RobertaProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os
from datasets import load_from_disk, DatasetDict
from typing import cast

RAW_DATA_PATH = "data/raw"
TOKENIZER_PATH = "data/tokenizer/tokenizer.json"    
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|pad|>", "<|endoftext|>", "<|unk|>"]

def get_training_corpus_in_batches(dataset, batch_size=1000):
    """
    Generator that yields text from the loaded dataset.
    """
    print("Extracting text from the loaded dataset...")
    count = 0
    batch = []
    for example in dataset:
        batch.append(example['text'])
        count +=1
        if len(batch)>=batch_size:
            yield batch
            batch = []

        if count % 20000 == 0:
            print(f"...processed {count} stories...")

    if batch:
        yield batch 
        
def train_bpe_tokenizer():
    """train the byte-pair-encoding tokenizer on the dataset"""
    print("Starting the tokenizer training...")

    # 1. Load the dataset that has been downloaded by running ingestion.py
    ds = cast(DatasetDict, load_from_disk(RAW_DATA_PATH))
    # We dont need the entire dataset to train the tokenizer as it the profits are minimal after a certain size
    train_slice = ds["train"].select(range(100_000))

    # 2. Python Iterator (Memory Efficient)
    # We create a generator so we don't load all text into RAM at once
    # We send 1000 stories at a time (batches)
    def batch_iterator(batch_size=1000):
        for i in range(0, len(train_slice), batch_size):
            yield train_slice[i: i+ batch_size]["text"]

    # 3. Defining the tokenizer components
    # BPE (Byte-Pair Encoding) is a model that starts with individual characters
    # and progressively merges the most frequent pairs of tokens.
    # We also define an "unknown" token for any sequence it can't represent.
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

    # 4. Set the Normalizer.
    # Normalization is a cleanup step that runs *before* pre-tokenization.
    # It handles things like converting different (but visually similar) Unicode
    # characters into a single, canonical form (e.g., standardizing accents, emojis).
    # NFKC (Normalization Form KC) is a good, standard choice.
    tokenizer.normalizer = Sequence([NFKC()]) # type: ignore

    # 5. Set the Pre-Tokenizer.
    # The pre-tokenizer splits the normalized text into initial "words" or "chunks".
    # `ByteLevel` is a popular choice: it treats the text as a stream of raw bytes.
    # This is powerful because it *never* fails (no <unk> at this stage) and can handle
    # all text, including special characters, whitespace, and emojis, without errors.
    # `add_prefix_space=False`: We tell it *not* to add a
    # leading space to the input, which prevents the "ĠOnce" issue where "Once" was tokenized as " Once".
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False) # type: ignore

    # 6. Configure the Trainer.
    # The trainer is responsible for *learning* the BPE merge rules from our corpus.
    # `vocab_size`: We're aiming for 10,000 tokens in our final vocabulary.
    # `special_tokens`: We explicitly tell the trainer to include our special tokens
    # so they are added to the vocabulary with their own IDs (e.g., 0 and 1).
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # 7. Train the Tokenizer.
    # This is the main event. The trainer will process all the text from the
    # iterator and build the vocabulary and merge rules based on token frequency.
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # 6. Set the Post-Processor.
    # The post-processor runs *after* tokenization (during `encode`) and
    # *before* decoding (during `decode`).
    # `RobertaProcessing` is designed to automatically add special tokens like
    # a "class" token (CLS) at the start and a "separator" token (SEP) at the end.
    #
    # Here, we're using <|endoftext|> as both our SEP and CLS token.
    # This is a common setup for GPT-style models.
    tokenizer.post_processor = RobertaProcessing( # type: ignore
        sep=("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
        cls=("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
    )

    # 8. Set the Decoder.
    # The decoder is responsible for converting token IDs back into a readable string.
    # Because our pre-tokenizer was `ByteLevel`, we *must* use a `ByteLevelDecoder`
    # to correctly re-assemble the bytes back into proper Unicode characters.
    # This is what correctly handles the 'Ġ' (space) characters during decoding.
    tokenizer.decoder = ByteLevelDecoder() # type: ignore
    
    # Save the tokenizer
    tokenizer_dir = os.path.dirname(TOKENIZER_PATH)
    os.makedirs(tokenizer_dir, exist_ok=True)
    print("Saving tokenizer...")
    tokenizer.save(TOKENIZER_PATH)
    print("Tokenizer saved.")

    return tokenizer

def test_tokenizer(tokenizer):
    """Test the trained tokenizer"""
    
    # Note: We don't need to re-add the decoder here since it was set
    # in the train_bpe_tokenizer function right before returning.
    
    test_texts = [
        "Once upon a time, there was a little girl.",
        "The cat and the dog played together.",
        "She was very happy and smiled.",
    ]
    
    print("\n" + "="*60)
    print("Testing Tokenizer")
    print("="*60)
    
    for text in test_texts:
        # 1. Encode the text.
        # This runs the full pipeline:
        # Normalization -> Pre-tokenization -> BPE Model -> Post-Processing.
        # The output (`encoding`) is an object containing the `.tokens` (human-readable)
        # and `.ids` (the numbers for the model).
        encoding = tokenizer.encode(text)
        
        print(f"\nOriginal: '{text}'")
        print(f"Tokens:   {encoding.tokens}")
        print(f"IDs:      {encoding.ids}")
        
        # 3. Decode the token IDs back into a string.
        # This runs the `.decoder` (ByteLevelDecoder) and also reverses the `post_processor`'s
        # work (e.g., by stripping the <|pad|> and <|endoftext|> tokens).
        decoded_text = tokenizer.decode(encoding.ids)
        print(f"Decoded:  '{decoded_text}'")
        
        # 4. Verification check.
        # A "round-trip" test ensures that `decode(encode(text))` gives us
        # back the original `text`. This is how we found our "prefix space" bug.
        if decoded_text == text:
            print("Status:   ✅ Round-trip successful")
        else:
            print("Status:   ❌ Round-trip FAILED")
            print(f"Expected: '{text}'")
            print(f"Got:      '{decoded_text}'")

def main():

    print("--- Starting Tokenizer Pipeline ---")
    train_bpe_tokenizer()
    print("---  Tokenizer Pipeline Ended ---")


if __name__ == "__main__":
    main()