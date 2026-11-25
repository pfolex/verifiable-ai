"""Small, isolated AI-coding experiment; never promotes or edits the daemon.

Outputs Python expressions, scored with a bounded AST interpreter (no eval/exec).
Development probes share templates with training: this is not a coding benchmark.
"""
import argparse
import ast
import hashlib
import json
import math
import operator
from pathlib import Path
import time


def curriculum(split):
    ranges = {'train': range(1, 9), 'development': range(9, 13)}
    if split not in ranges:
        raise ValueError('Unknown split')
    rows = []
    for k in ranges[split]:
        for skill, question, answer in (
            ('sgd', f'SGD: w minus {k} times lr times g.', f'w-{k}*lr*g'),
            ('linear', f'Linear: {k} times x plus b.', f'{k}*x+b'),
            ('lora', f'LoRA: base plus {k}/r times a times b.', f'base+{k}/r*a*b'),
        ):
            rows.append(dict(id=f'{split}-{skill}-{k}', skill=skill,
                             prompt='Q:Code:'+question+' Python expression:<ans>',
                             answer=answer))
    return rows


def interpret(source, values):
    if not isinstance(source, str) or len(source) > 128:
        raise ValueError('Expression too long')
    tree = ast.parse(source, mode='eval')
    if len(list(ast.walk(tree))) > 64:
        raise ValueError('Expression too complex')
    ops = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id in values:
            result = values[node.id]
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in ops:
            result = ops[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            result = -visit(node.operand)
        else:
            raise ValueError('Unsupported Python syntax')
        if not math.isfinite(result) or abs(result) > 1e12:
            raise ValueError('Unbounded arithmetic')
        return result
    return visit(tree)


def correct(source, reference):
    try:
        for i in range(1, 9):
            values = dict(w=i*.7, lr=.01*i, g=(-1)**i*(i+.3), x=i-4.5,
                          b=i*.13, base=2-i*.2, r=i+1, a=1.7-i*.3)
            if not math.isclose(interpret(source, values), interpret(reference, values),
                                rel_tol=1e-8, abs_tol=1e-8):
                return False
        return True
    except (ValueError, SyntaxError, ArithmeticError, TypeError, RecursionError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--champion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=.0001)
    parser.add_argument('--max-seconds', type=float, default=120)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000 or not 0 < args.learning_rate <= .01 or not 0 < args.max_seconds <= 3600:
        parser.error('Invalid training budget')
    # Exclusive directory creation prevents replacing previous evidence/checkpoints.
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from aix_zero.candidate_checkpoint import load_candidate_checkpoint, encode_candidate_checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    initial = args.champion.read_bytes()
    model, tok, checkpoint_format = load_candidate_checkpoint(initial)
    train, dev = curriculum('train'), curriculum('development')
    encoded = []
    for row in train:
        prefix = tok.encode(row['prompt'])
        target = tok.encode(row['answer']+'</ans>', add_bos=False)
        if tok.decode(target) != row['answer']+'</ans>':
            raise ValueError('Tokenizer cannot represent coding targets exactly')
        if len(prefix)+len(target) > model.config.max_seq_len:
            raise ValueError('Coding example exceeds checkpoint context')
        encoded.append((prefix+target, [0]*len(prefix)+target))
    width = max(len(x) for x, _ in encoded)
    inputs = torch.tensor([x+[0]*(width-len(x)) for x, _ in encoded])
    labels = torch.tensor([y+[0]*(width-len(y)) for _, y in encoded])
    def evaluate():
        model.eval()
        results = []
        with torch.no_grad():
            for row in dev:
                ids = tok.encode(row['prompt'])
                generated = []
                for _ in range(32):
                    if len(ids) >= model.config.max_seq_len:
                        break
                    token = int(model(torch.tensor([ids]))['logits'][0, -1].argmax())
                    ids.append(token)
                    if token in (tok.end_ans_id, tok.eos_id):
                        break
                    generated.append(token)
                answer = tok.decode(generated)
                results.append(dict(id=row['id'], skill=row['skill'], output=answer,
                                    correct=correct(answer, row['answer'])))
        return dict(correct=sum(r['correct'] for r in results), total=len(results), rows=results)
    before = evaluate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start = time.monotonic()
    losses = []
    model.train()
    for _ in range(args.steps):
        if time.monotonic()-start >= args.max_seconds:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs, labels=labels)['loss']
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    after = evaluate()
    candidate = encode_candidate_checkpoint(model, tok)
    (args.output/'candidate.pt').write_bytes(candidate)
    report = dict(task='AI scalar Python expressions; not general coding expertise',
                  assistance_mode='neural_only', automatic_promotion=False,
                  independent_final_test=False, split_note='Coefficient-disjoint, template-shared development',
                  champion_sha256=hashlib.sha256(initial).hexdigest(),
                  candidate_sha256=hashlib.sha256(candidate).hexdigest(),
                  checkpoint_format=checkpoint_format, physical_parameters=sum(p.numel() for p in model.parameters()),
                  seed=args.seed, learning_rate=args.learning_rate, steps_completed=len(losses),
                  training_budget_seconds=args.max_seconds, budget_mode='cooperative; evaluation excluded',
                  training_rows=len(train), losses=losses, before=before, after=after)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('steps_completed', 'physical_parameters', 'automatic_promotion')}))
    print(f"Development: {before['correct']}/{before['total']} -> {after['correct']}/{after['total']}")
    print('Report:', args.output/'report.json')


if __name__ == '__main__':
    main()
