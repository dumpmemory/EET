import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, T5Config, T5Model
from eet.transformers.modeling_t5 import EETT5Model, EETT5ForConditionalGeneration
import time

min_length = 30
max_length = 100


def main():
    torch.set_grad_enabled(False)

    model_dir = "/root/data/models/test_ttg_t5/"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir, from_flax=False).to('cuda:0')
    eet_model = EETT5ForConditionalGeneration.from_pretrained(model_dir, max_batch=10, full_seq_len=100, data_type=torch.float16)

    input_str = "少侠路过码头时，一位船夫哽咽着向少侠求救：“少侠，我两年前出海时答应过儿子会给他带一把铁木剑，可船靠岸时那把剑不小心掉水里了，我沿着船舷上刻的记号找了许久也没找到， \
    # 这可如何是好？”少侠说：“帮你下水去捞。”<extra_id_0>船夫的儿子获得了铁木剑，十分开心。</s>"
    input_str = input_str.lower().replace(":", '：').replace('“', '"').replace('”', '"').replace('!', '！').replace('?',
                                                                                                                  '？')
    inputs = tokenizer([input_str] * 10, return_tensors='pt', add_special_tokens=False)
    input_ids = inputs['input_ids'].to(model.device)

    # ts
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    bot_outs = model.generate(
        input_ids,
        top_k=20,
        top_p=0.9,
        do_sample=True,
        temperature=0.88,
        repetition_penalty=1.0,
        no_repeat_ngram_size=7,
        min_length=min_length,
        max_length=max_length,
        bad_words_ids=[[tokenizer.unk_token_id]],
        eos_token_id=21129,
        return_dict_in_generate=True,
        output_scores=True,
        bos_token_id=0,
        decoder_start_token_id=0
    )  # "<extra_id_1>" 如果是150万就不需要这个，训练的时候没加
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_ts_full = t2 - t1

    gen_sequences_id = bot_outs.sequences
    # gen_sequences_id = bot_outs

    gen_sequences_id = gen_sequences_id[:, 1:]  # 去掉开始的pad
    lengths = torch.sum(gen_sequences_id != tokenizer.pad_token_id, -1)
    print('generate output length: ', gen_sequences_id.size(), '\t', lengths)
    gene_outs = tokenizer.batch_decode(gen_sequences_id, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)  # False
    print('generated results: ', gene_outs)

    scores = bot_outs.scores  # tuple of tensor :length, (num, vocab)
    probs = torch.stack(scores, dim=1).softmax(-1)
    gen_probs = torch.gather(probs, 2, gen_sequences_id[:, :, None]).squeeze(-1)  # pad的分数都是0
    gen_probs = gen_probs.sum(-1) / lengths
    print(gen_probs)

    # eet
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    eet_gen_sequences_id = eet_model.generate(
        input_ids,
        top_k=20,
        top_p=0.9,
        do_sample=True,
        temperature=0.88,
        repetition_penalty=1.0,
        no_repeat_ngram_size=7,
        min_length=min_length,
        max_length=max_length,
        bad_words_ids=[[tokenizer.unk_token_id]],
        eos_token_id=21129,
        return_dict_in_generate=True,
        output_scores=True,
        bos_token_id=0,
        decoder_start_token_id=0
    )  # "<extra_id_1>" 如果是150万就不需要这个，训练的时候没加

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_eet_full = t2 - t1
    # time
    print("eet full time: ", time_eet_full)
    print("ts full time: ", time_ts_full)
    print(input_str)

    eet_gen_sequences_id = eet_gen_sequences_id[:, 1:]  # 去掉开始的pad
    lengths = torch.sum(eet_gen_sequences_id != tokenizer.pad_token_id, -1)
    print('eet generate output length: ', eet_gen_sequences_id.size(), '\t', lengths)
    eet_gene_outs = tokenizer.batch_decode(eet_gen_sequences_id, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)  # False
    print(' eet generated results: ', eet_gene_outs)


if __name__ == "__main__":
    main()

