import argparse
import json
from transformers import GPT2Model, BertModel
from transformers import GPT2Tokenizer, BertTokenizer
import random
from collections import defaultdict
from common import *
import os
from tqdm import tqdm
from datasets import load_dataset
from stollen_prob_search import StolenProbabilitySearch, ExactAlgorithms, ApproxAlgorithms
from layer_norm_common import *
from FactorizedLayerNorm import FactorizedLayerNorm


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h):
        return h

class MODEL(Enum):
    BERT_NO_LN = auto()
    GPT2_W_SCALE = auto()
    GPT2_WO_SCALE = auto()

    @staticmethod
    def from_string(s):
        try:
            return MODEL[s]
        except KeyError:
            raise ValueError()

    def __str__(self):
        if self is MODEL.BERT_NO_LN:
            return "BERT_NO_LN"
        elif self is MODEL.GPT2_W_SCALE:
            return "GPT2_W_SCALE"
        elif self is MODEL.GPT2_WO_SCALE:
            return "GPT2_WO_SCALE"
        return "NA"

    def get_model(self):
        if self is MODEL.BERT_NO_LN:
            model = BertModel.from_pretrained('bert_no_ln/checkpoint-34000')
            model.base_model.embeddings.LayerNorm = Dummy()
            for block in model.base_model.encoder.layer:
                block.output.LayerNorm = Dummy()
                block.attention.output.LayerNorm = Dummy()
            return model
        elif self is MODEL.GPT2_W_SCALE:
            return GPT2Model.from_pretrained('gpt2_scale/checkpoint-34000')
        elif self is MODEL.GPT2_WO_SCALE:
            model = GPT2Model.from_pretrained('gpt2_no_scale/checkpoint-50000')
            # since it load the model as regular GPT2Model and put random weights on the layernorm,
            # we need manually to replace them with FactorizedLayerNorm modules
            for block in model.base_model.h:
                block.ln_1 = FactorizedLayerNorm(block.ln_1.normalized_shape, do_projection=True, do_scale=False,
                                                 elementwise_affine=False)
            return model

    def get_tokenizer(self, mode):
        if self is MODEL.BERT_NO_LN:
            return BertTokenizer.from_pretrained('bert_no_ln/checkpoint-50000')
        elif self is MODEL.GPT2_W_SCALE:
            return GPT2Tokenizer.from_pretrained('gpt2_scale/checkpoint-50000')
        elif self is MODEL.GPT2_WO_SCALE:
            return GPT2Tokenizer.from_pretrained('gpt2_no_scale/checkpoint-50000')

    def register_hooks(self, model, memory_dict):
        handles = []
        if self is MODEL.BERT_NO_LN:
            for layer_number, layer in enumerate(
                    [model.embeddings.LayerNorm] + [l.output.LayerNorm for l in model.encoder.layer]):
                def make_ln_unargmaxable_hook(ln):
                    def hook(module, i, o):
                        # (seq_len, hidden_size)
                        hidden_states_before = i[0][0].detach().numpy()
                        hidden_states_after = o[0].detach().numpy()

                        for hidden_states, s in [(hidden_states_before, 'before'), (hidden_states_after, 'after')]:
                            num_keys, dim = hidden_states.shape
                            sp_search = StolenProbabilitySearch(hidden_states)
                            results = sp_search.find_bounded_classes(class_list=tuple(range(num_keys)),
                                                                     exact_algorithm=ExactAlgorithms.default(),
                                                                     approx_algorithm=ApproxAlgorithms.default(),
                                                                     lb=-100,
                                                                     ub=100,
                                                                     patience=100,
                                                                     #  num_processes=mp.cpu_count() - 1,
                                                                     desc=f" layer {ln}")
                            bounded = [i for i, r in enumerate(results) if r['is_bounded']]
                            memory_dict[ln][f'unargmaxable_{s}_count'] = len(bounded)
                            memory_dict[ln][f'unargmaxable_{s}_unselectable_idx'] = bounded

                    return hook

                handles.append(layer.register_forward_hook(make_ln_unargmaxable_hook(layer_number)))
        elif self is MODEL.GPT2_W_SCALE or self is MODEL.GPT2_WO_SCALE:
            for layer_number, block in enumerate(model.h):
                def make_ln_unargmaxable_hook(ln):
                    def hook(module, i, o):
                        # (seq_len, hidden_size)
                        hidden_states_before = i[0][0].detach().numpy()
                        hidden_states_after = o[0].detach().numpy()

                        for hidden_states, s in [(hidden_states_before, 'before'), (hidden_states_after, 'after')]:
                            # for hidden_states, s in [(hidden_states_before, 'before')]:
                            num_keys, dim = hidden_states.shape
                            sp_search = StolenProbabilitySearch(hidden_states)
                            results = sp_search.find_bounded_classes(class_list=tuple(range(num_keys)),
                                                                     exact_algorithm=ExactAlgorithms.default(),
                                                                     approx_algorithm=ApproxAlgorithms.default(),
                                                                     lb=-100,
                                                                     ub=100,
                                                                     patience=100,
                                                                     #  num_processes=mp.cpu_count() - 1,
                                                                     desc=f" layer {ln}")
                            bounded = [i for i, r in enumerate(results) if r['is_bounded']]
                            memory_dict[ln][f'unargmaxable_{s}_count'] = len(bounded)
                            memory_dict[ln][f'unargmaxable_{s}_unselectable_idx'] = bounded
                    return hook

                handles.append(block.ln_1.register_forward_hook(make_ln_unargmaxable_hook(layer_number)))
        return handles

    def get_unargmaxable_rate(self, dataset, num_samples, min_sample_size, max_sample_size, out_dir):
        model = self.get_model()
        tokenizer = self.get_tokenizer()
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        # load previously report
        report_path = os.path.join(out_dir, 'report.json')
        unselectable_tokens_path = os.path.join(out_dir, 'unselectable_tokens.json')
        result_dict = defaultdict(lambda: defaultdict(list))
        unselectable_tokens = []
        generator = dataset.get_inputs(min_sample_size, max_sample_size, tokenizer, num_samples)
        for model_inputs, model_tokens in tqdm(generator, total=min(num_samples, len(dataset)), dynamic_ncols=True):
            unselectable_tokens.append({})
            unselectable_tokens[-1]['tokens'] = tokenizer.decode(tokenizer.convert_tokens_to_ids(model_tokens))
            unselectable_tokens[-1]["all"] = []
            # input -> layer -> stats
            memory_dict = defaultdict(dict)
            handles = self.register_hooks(model, memory_dict)
            model(**model_inputs)
            for handle in handles:
                handle.remove()
            for ln in memory_dict.keys():
                input_len = model_inputs['input_ids'].shape[-1]
                result_dict[ln]['before_percentage'].append(memory_dict[ln][f'unargmaxable_before_count'] / input_len)
                result_dict[ln]['after_percentage'].append(memory_dict[ln][f'unargmaxable_after_count'] / input_len)
                result_dict[ln]['after_percentage'].append(memory_dict[ln][f'unargmaxable_after_count'] / input_len)
                result_dict[ln]['before'].append(memory_dict[ln][f'unargmaxable_before_count'])
                result_dict[ln]['after'].append(memory_dict[ln][f'unargmaxable_after_count'])
                token_idx = memory_dict[ln][f'unargmaxable_after_unselectable_idx']
                unselectable_tokens[-1][ln] = tokenizer.convert_ids_to_tokens(
                    list(set(model_inputs.input_ids[0].gather(0, torch.tensor(token_idx)).tolist())))
            unselectable_all_layers = {t for t in unselectable_tokens[-1][ln]}
            for k, v in unselectable_tokens[-1].items():
                if k != 'tokens':
                    unselectable_all_layers = unselectable_all_layers.intersection(v)
            unselectable_tokens[-1]["all"] = list(unselectable_all_layers)

        for ln in result_dict.keys():
            result_dict[ln]['before_percentage'] = np.mean(result_dict[ln]['before_percentage'])
            result_dict[ln]['after_percentage'] = np.mean(result_dict[ln]['after_percentage'])
            result_dict[ln]['before'] = np.mean(result_dict[ln]['before'])
            result_dict[ln]['after'] = np.mean(result_dict[ln]['after'])

        with open(report_path, "w") as f:
            json.dump(result_dict, f, indent=4)
        with open(unselectable_tokens_path, "w") as f:
            json.dump(unselectable_tokens, f, indent=4)


class DATASET(Enum):
    SQuAD = load_dataset('squad', split="validation")
    WIKIPEDIA = load_dataset('wikipedia', "20220301.en", split="train")
    SST2 = load_dataset('glue', 'sst2', split="test")

    @staticmethod
    def from_string(s):
        try:
            return DATASET[s]
        except KeyError:
            raise ValueError()

    def __str__(self):
        if self is DATASET.SQuAD:
            return "SQuAD"
        if self is DATASET.WIKIPEDIA:
            return "wikipedia"
        if self is DATASET.SST2:
            return "SST2"
        return "NA"

    def get_field(self):
        if self is DATASET.SQuAD:
            return 'context'
        if self is DATASET.WIKIPEDIA:
            return 'text'
        if self is DATASET.SST2:
            return "sentence"

    def get_input(self, min_sample_size, max_sample_size, tokenizer, sample_idx):
        squad_dataset = self.value
        raw_inputs = squad_dataset[sample_idx][self.get_field()]
        model_inputs = tokenizer(raw_inputs, return_tensors="pt", max_length=max_sample_size, truncation=True)
        if model_inputs['input_ids'].shape[1] < min_sample_size:
            return None
        model_tokens = tuple(tokenizer.convert_ids_to_tokens(model_inputs['input_ids'][0]))
        return model_inputs, model_tokens

    def get_inputs(self, min_sample_size, max_sample_size, tokenizer, num_samples):
        squad_dataset = self.value
        if num_samples < len(self):
            for i in range(num_samples):
                result = None
                while result is None:
                    sample_idx = random.randint(0, len(self) - 1)
                    result = self.get_input(min_sample_size, max_sample_size, tokenizer, sample_idx)
                yield result
        else:
            for i in range(len(squad_dataset)):
                result = None
                while result is None:
                    result = self.get_input(min_sample_size, max_sample_size, tokenizer, i)
                yield result

    def __len__(self):
        return len(self.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", default=MODEL.GPT2_W_SCALE,
                        type=MODEL.from_string, choices=list(MODEL), required=False)
    parser.add_argument("--dataset", dest="dataset", default=DATASET.SQuAD,
                        type=DATASET.from_string, choices=list(DATASET), required=False)
    parser.add_argument("--num_samples", dest="num_samples", default=1000,
                        type=int, required=False)
    parser.add_argument("--min_sample_size", dest="min_sample_size", default=0,
                        type=int, required=False)
    parser.add_argument("--max_sample_size", dest="max_sample_size", default=128,
                        type=int, required=False)
    parser.add_argument("--out_dir", dest="out_dir", default="unselectable_layer",
                        type=str, required=False)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    out_dir = os.path.join(args.out_dir, str(args.dataset))
    args.model.get_unargmaxable_rate(args.dataset, args.num_samples, args.min_sample_size, args.max_sample_size,
                                     out_dir)