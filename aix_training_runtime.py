"""Full causal-LM LoRA training and actual generation; heavy imports stay optional."""
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random

from artifact_io import canonical_json, write_json
from aix_training_data import checkpoint_identity, load_bundle, render_prompt
from provider_adapters import ProviderResponse


def libraries():
    import torch
    import transformers
    import peft
    return torch, transformers, peft


def execution(config):
    torch, _, _ = libraries()
    requested = config['training']['device']
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if requested == 'auto' else requested
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable')
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    return device, dtype


def tokenizer_for(config):
    _, transformers, _ = libraries()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['base_model'], local_files_only=True, trust_remote_code=False)
    template_file = Path(config['base_model']) / 'chat_template.jinja'
    if not tokenizer.chat_template and template_file.exists():
        tokenizer.chat_template = template_file.read_text()
    if not tokenizer.chat_template:
        raise ValueError('checkpoint needs its native chat template')
    if tokenizer.eos_token_id is None:
        raise ValueError('checkpoint tokenizer needs an EOS token')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def config_for(config):
    _, transformers, _ = libraries()
    raw = json.loads((Path(config['base_model']) / 'config.json').read_text())
    loaded = transformers.AutoConfig.from_pretrained(config['base_model'], local_files_only=True, trust_remote_code=False)
    # The local AIX export uses Transformers 5's nested RoPE representation.
    # Preserve its actual theta when loading with our compatible 4.x stack.
    rope = raw.get('rope_parameters')
    if rope is not None:
        if rope.get('rope_type', 'default') != 'default':
            raise ValueError('this workflow only maps default nested RoPE; use a supported checkpoint config')
        theta = rope['rope_theta']
        if not math.isfinite(theta) or theta <= 0 or ('rope_theta' in raw and raw['rope_theta'] != theta):
            raise ValueError('invalid or conflicting RoPE settings')
        loaded.rope_theta = theta
        loaded.rope_scaling = None
    return loaded


def model_for(config):
    torch, transformers, _ = libraries()
    device, dtype = execution(config)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config['base_model'], config=config_for(config), torch_dtype=dtype, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, low_cpu_mem_usage=True, attn_implementation='eager')
    return model.to(device)


def encode_example(tokenizer, example, max_length, vocab_size):
    prompt = tokenizer.encode(render_prompt(tokenizer, example['input']), add_special_tokens=False)
    answer = tokenizer.encode(example['target'], add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prompt + answer
    if not prompt or not answer or max(ids) >= vocab_size or min(ids) < 0:
        raise ValueError('tokenizer/checkpoint vocabulary mismatch')
    if len(ids) > max_length:
        return None  # Never train against a silently truncated reference answer.
    return {'input_ids': ids, 'labels': [-100] * len(prompt) + answer, 'sample_id': example['sample_id'],
            'domain': example['input']['domain']}


def experiment_hash(config):
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def train_adapter(run_dir):
    root = Path(run_dir)
    config, examples, manifest = load_bundle(root)
    destination = root / 'adapter'
    if destination.exists() or (root / 'adapter.pending').exists() or (root / 'training-report.json').exists():
        raise ValueError('candidate already exists; use a fresh experiment directory')
    torch, transformers, peft = libraries()
    torch.manual_seed(config['training']['seed'])
    tokenizer = tokenizer_for(config)
    model_config = json.loads((Path(config['base_model']) / 'config.json').read_text())
    encoded = [encode_example(tokenizer, example, config['training']['max_seq_length'], model_config['vocab_size']) for example in examples]
    batches = [batch for batch in encoded if batch is not None]
    counts = dict(Counter(batch['domain'] for batch in batches))
    if any(not counts.get(domain) for domain in manifest['generation']['domains']):
        raise ValueError('a domain has no complete training examples that fit; increase --max-seq-length in a new run')
    domains = manifest['generation']['domains']
    if config['training']['steps'] < len(domains):
        raise ValueError('steps must cover every requested training domain')
    print(f'Training on {len(batches)} complete examples: {counts}; skipped overlength={len(examples) - len(batches)}', flush=True)
    print('Loading the full base checkpoint for LoRA training...', flush=True)
    model = model_for(config)
    last_n = config['training']['last_n_layers']
    layers = list(range(model.config.num_hidden_layers - last_n, model.config.num_hidden_layers))
    model = peft.get_peft_model(model, peft.LoraConfig(
        task_type='CAUSAL_LM', r=config['training']['rank'], lora_alpha=2 * config['training']['rank'],
        lora_dropout=0, target_modules=['q_proj', 'v_proj'], layers_to_transform=layers, bias='none'))
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any('lora_' not in name for name, _ in trainable):
        raise ValueError('expected only LoRA adapter weights to be trainable')
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=config['training']['learning_rate'])
    device, _ = execution(config)
    rng, losses, used, visits = random.Random(config['training']['seed']), [], set(), Counter()
    orders = {domain: [i for i, batch in enumerate(batches) if batch['domain'] == domain] for domain in domains}
    cursors = {domain: len(order) for domain, order in orders.items()}
    for step in range(config['training']['steps']):
        domain = domains[step % len(domains)]
        if cursors[domain] >= len(orders[domain]):
            rng.shuffle(orders[domain])
            cursors[domain] = 0
        index = orders[domain][cursors[domain]]
        cursors[domain] += 1
        batch = batches[index]
        used.add(index)
        visits[domain] += 1
        ids = torch.tensor([batch['input_ids']], device=device)
        labels = torch.tensor([batch['labels']], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss; no completed adapter was saved')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(f"Step {step + 1}/{config['training']['steps']} loss={losses[-1]:.6f}", flush=True)
    if checkpoint_identity(config['base_model']) != config['base_identity']:
        raise ValueError('base files changed during training')
    temporary = root / 'adapter.pending'
    if temporary.exists():
        raise ValueError('an interrupted adapter save exists; use a new experiment directory')
    model.save_pretrained(temporary, safe_serialization=True)
    temporary.rename(destination)
    report = {'experiment_sha256': experiment_hash(config), 'steps': len(losses), 'losses': losses,
              'training_examples_used': len(used), 'training_examples_eligible': len(batches),
              'training_examples_skipped_overlength': len(examples) - len(batches),
              'examples_eligible_by_domain': counts,
              'examples_used_by_domain': dict(Counter(batches[i]['domain'] for i in used)),
              'steps_by_domain': dict(visits), 'used_sample_ids': sorted(batches[i]['sample_id'] for i in used),
              'trainable_parameters': sum(p.numel() for _, p in trainable),
              'adapter_identity': checkpoint_identity(destination, adapter=True),
              'base_unchanged': True, 'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__}}
    write_json(report, root / 'training-report.json')
    return report


class LocalCheckpointProvider:
    def __init__(self, config, tokenizer, model, adapter_identity=None):
        torch, transformers, peft = libraries()
        self.tokenizer, self.loaded = tokenizer, model.eval()
        self.config = copy.deepcopy(config)
        device, dtype = execution(config)
        self.device = device
        base_sha = config['base_identity']['sha256']
        self.model = {'provider': 'hf-local', 'model_id': Path(config['base_model']).name,
                      'revision': hashlib.sha256((base_sha + (adapter_identity or '')).encode()).hexdigest(),
                      'adapter_version': 'aix-causal-json-1.0.0',
                      'parameters': {'base_sha256': base_sha, 'adapter_sha256': adapter_identity,
                                     'experiment_sha256': experiment_hash(config), 'generation': config['generation'],
                                     'device': device, 'dtype': str(dtype),
                                     'rope_theta': model.config.rope_theta,
                                     'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__}}}

    def generate(self, sample_id, task):
        torch, _, _ = libraries()
        print(f"Generating {sample_id} (limit {self.config['generation']['max_new_tokens']} new tokens)...", flush=True)
        prompt = self.tokenizer(render_prompt(self.tokenizer, task), add_special_tokens=False, return_tensors='pt')
        prompt = {key: value.to(self.device) for key, value in prompt.items() if key in ('input_ids', 'attention_mask')}
        if int(prompt['input_ids'].max()) >= self.loaded.get_input_embeddings().num_embeddings:
            raise ValueError('tokenizer/checkpoint vocabulary mismatch')
        length = prompt['input_ids'].shape[1]
        if length + self.config['generation']['max_new_tokens'] > self.loaded.config.max_position_embeddings:
            raise ValueError('generation exceeds model context; no silent input truncation')
        torch.manual_seed(self.config['training']['seed'])
        with torch.inference_mode():
            generated = self.loaded.generate(**prompt, do_sample=False,
                max_new_tokens=self.config['generation']['max_new_tokens'],
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
        output_ids = generated[0, length:].detach().cpu().tolist()
        print(f'Generated {len(output_ids)} tokens for {sample_id}', flush=True)
        return ProviderResponse(output=self.tokenizer.decode(output_ids, skip_special_tokens=True),
                                input_tokens=length, output_tokens=len(output_ids))


def evaluate_checkpoint(run_dir, which):
    from benchmark_runner import run_benchmark, save_result
    from run_benchmark import verisci_score
    root = Path(run_dir)
    output = root / (which + '.json')
    if output.exists():
        raise ValueError(f'{output} already exists; existing measurements are never overwritten')
    config, _, manifest = load_bundle(root)
    adapter_sha = None
    if which == 'candidate':
        training = json.loads((root / 'training-report.json').read_text())
        if training['experiment_sha256'] != experiment_hash(config):
            raise ValueError('experiment configuration differs from training')
        identity = checkpoint_identity(root / 'adapter', adapter=True)
        if identity != training['adapter_identity']:
            raise ValueError('adapter files changed')
        adapter_sha = identity['sha256']
    print(f'Loading {which} for actual generated-answer evaluation...', flush=True)
    tokenizer = tokenizer_for(config)
    model = model_for(config)
    if which == 'candidate':
        _, _, peft = libraries()
        model = peft.PeftModel.from_pretrained(model, root / 'adapter', is_trainable=False, local_files_only=True)
    provider = LocalCheckpointProvider(config, tokenizer, model, adapter_sha)
    result = run_benchmark(manifest['cases'], dataset=manifest['dataset'], provider=provider,
                           scorer=verisci_score, scorer_metadata=manifest['scorer'], manifest=manifest)
    if checkpoint_identity(config['base_model']) != config['base_identity']:
        raise ValueError('base files changed during evaluation')
    save_result(result, output)
    print(f"Saved {output}: mean={result['summary']['mean_score']:.6f}, errors={sum('error' in sample for sample in result['samples'])}", flush=True)
    return result
