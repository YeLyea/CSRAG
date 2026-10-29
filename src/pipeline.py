"""Three-stage CSRAG inference using vLLM continuous batching."""
import argparse
import json
import re
import time
from pathlib import Path

from eval import evaluate
from prompts import (
    system_fact, user_fact, system_context_para, user_context_para,
    system_copa_response, user_copa_response,
)


def parse_list_output(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.split(r'<think>', text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('-'):
            items.append(line[1:].strip())
        elif '[PARAPHRASE]:' in line:
            items.append(line.split('[PARAPHRASE]:', 1)[1].strip())
    return [item for item in items if item] or ([text] if text else [])


def load_samples(path, limit=None):
    samples = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(samples, list) or not samples:
        raise ValueError('Data must be a nonempty JSON array.')
    samples = samples[:limit] if limit else samples
    seen = set()
    for sample in samples:
        for field in ('id', 'question', 'context', 'choices', 'answer'):
            if field not in sample:
                raise ValueError(f'Missing field: {field}')
        key = str(sample['id'])
        if key in seen:
            raise ValueError(f'Duplicate sample id: {key}')
        seen.add(key)
    return samples


def run(args):
    from vllm import LLM, SamplingParams
    from processors import collect_stopword_ids, build_logit_bias

    samples = load_samples(args.data, args.limit)
    llm = LLM(model=args.model, dtype='bfloat16', seed=args.seed,
              tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              trust_remote_code=args.trust_remote_code)
    tokenizer = llm.get_tokenizer()
    stop_ids = collect_stopword_ids(tokenizer)
    template_kwargs = {'enable_thinking': False} if 'qwen3' in args.model.lower() else {}

    def chat(system, users, max_tokens, biases=None):
        conversations = [[{'role': 'system', 'content': system},
                          {'role': 'user', 'content': user}] for user in users]
        for messages in conversations:
            ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                               add_generation_prompt=True, **template_kwargs)
            if len(ids) + max_tokens > args.max_model_len:
                raise ValueError('Input plus output budget exceeds --max-model-len; '
                                 'increase the window or shorten the input.')
        sampling = [SamplingParams(temperature=0, max_tokens=max_tokens,
                                   logit_bias=biases[i] if biases else None)
                    for i in range(len(users))]
        outputs = llm.chat(conversations, sampling, use_tqdm=True,
                           chat_template_kwargs=template_kwargs)
        if len(outputs) != len(users):
            raise RuntimeError('vLLM output count does not match input count.')
        return [output.outputs[0].text.strip() for output in outputs]

    start = time.perf_counter()
    # 1. Elicit query-related parametric facts without retrieved context.
    fact_texts = chat(system_fact,
                      [user_fact.format(question=s['question']) for s in samples],
                      args.fact_max_tokens)
    facts = [parse_list_output(text) for text in fact_texts]

    # 2. Paraphrase evidence; retain the research runner's concatenation rule.
    para_texts = chat(system_context_para,
                      [user_context_para.format(question=s['question'], context=s['context'])
                       for s in samples], args.para_max_tokens)
    contexts = [s['context'] + '\n'.join(parse_list_output(text))
                for s, text in zip(samples, para_texts)]

    # 3. Each request gets its own additive suppression/boost map.
    biases = [build_logit_bias(tokenizer, f, c, stop_ids, args.alpha, args.beta)
              for f, c in zip(facts, contexts)]
    answers = chat(system_copa_response,
                   [user_copa_response.format(question=s['question'], context=c,
                                              options='\n'.join(s['choices']))
                    for s, c in zip(samples, contexts)], args.final_max_tokens, biases)
    elapsed = time.perf_counter() - start
    predictions = {s['id']: answer for s, answer in zip(samples, answers)}
    scores = evaluate(samples, predictions, cot_format=True, detailed_output=True)
    results = [{'id': s['id'], 'parametric_facts': f, 'final_context': c,
                'final_response': answer}
               for s, f, c, answer in zip(samples, facts, contexts, answers)]
    summary = {key: value for key, value in scores.items() if key != 'details'}
    summary['generation_seconds'] = elapsed
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'config': vars(args), 'summary': summary,
                                  'details': scores['details'], 'results': results},
                                 ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Saved to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='results/csrag.json')
    parser.add_argument('--alpha', type=float, default=-1.0)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--fact-max-tokens', type=int, default=4196)
    parser.add_argument('--para-max-tokens', type=int, default=4196)
    parser.add_argument('--final-max-tokens', type=int, default=4196)
    parser.add_argument('--max-model-len', type=int, default=16384)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--trust-remote-code', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if min(args.fact_max_tokens, args.para_max_tokens, args.final_max_tokens,
           args.max_model_len, args.tensor_parallel_size) <= 0:
        parser.error('Token budgets, context window and tensor parallel size must be positive')
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error('--gpu-memory-utilization must be in (0, 1]')
    run(args)


if __name__ == '__main__':
    main()
