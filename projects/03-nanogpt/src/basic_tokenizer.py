class BasicTokenizer:
    def __init__(self):
        self.merge_vocab = {}
        self.vocab = {idx:bytes([idx]) for idx in range(256)}

    def encoder(self, text):
        ids = list(text.encode('utf-8'))
        for pair, idx in sorted(self.merge_vocab.items(), key=lambda x: x[1]):
            ids = self.merge(ids, idx, pair)
        return ids
    
    def decoder(self, ids):
        "将ids进行解码"
        text_bytes = b''.join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode('utf-8',errors='replace')
        return text


    def train(self, text, vocab_size, verbose=False):
        "训练分词器，其中vocab_size是分词器词表的大小，text是原始文本"
        assert vocab_size > 256
        ids = self.encoder(text)
        loop_cnt = vocab_size - 256
        for i in range(loop_cnt):
            idx = 256 + i
            stats = self.get_stats(ids)
            pair = max(stats,key=stats.get)
            if not pair:
                break
            self.merge_vocab[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            ids = self.merge(ids,idx,pair)

    def get_stats(self,ids):
        freq_dict = {}
        for bigram in zip(ids,ids[1:]):
            freq_dict[bigram] = freq_dict.get(bigram, 0) + 1
        return freq_dict

    def merge(self, ids, idx, pair):
        #找到列表ids里的所有pair对，然后将其合并,标号为idx.
        # pair = max(counts,counts.get)
        newids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
bt = BasicTokenizer()
text = """This is a markdown-first, Cognee-backed knowledge wiki about Andrej Karpathy's public work. The wiki is designed to compound: sources become pages, queries become reusable synthesis, feedback becomes page and skill improvement.

The website has a live Ask Cognee tab when scripts/wiki_server.py is running: it recalls Cognee memory, searches the markdown wiki, and files reviewed query-synthesis pages automatically."""

text2 = """The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."""
bt.train(text,300)
print(bt.merge_vocab)
print(bt.vocab)
print(text2 == bt.decoder(bt.encoder(text2)))
