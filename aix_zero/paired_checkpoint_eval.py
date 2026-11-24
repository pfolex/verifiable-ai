"""Offline paired neural evaluation on manifest-selected development or regression rows."""
import argparse
import hashlib
import json
from pathlib import Path

from aix_zero.experiment_splits import verify_manifest
from aix_zero.paired_evaluation import evaluate_pair


def evaluate_checkpoints(
    manifest,
    champion_path,
    candidate_path,
    *,
    split='development',
    max_new_tokens=36,
    candidate_cartridge_bank_dir=None,
):
    import torch
    from aix_zero.benchmark_baseline import load_v11
    from aix_zero.meta_ai_autopilot import verify_r1_reasoning_trace
    from aix_zero.eval_harness_bridge import score_rational_or_probability
    from aix_zero.routed_specialist_training import compute_model_state_sha256
    from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank
    if split not in ('development', 'regression'):
        raise ValueError('Only development and regression splits are available to this evaluator')
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 112:
        raise ValueError('Invalid generation budget')
    payload = verify_manifest(manifest)
    rows = [record['row'] for record in payload['splits'][split]]
    if not rows:
        raise ValueError('Selected evaluation split is empty')
    base = Path(__file__).resolve().parent / 'checkpoints/aix_zero_v11_kip.safetensors'
    models = {}; tokenizers = {}; identities = {}; state_digests = {}
    # Read once so the recorded digest identifies the exact bytes deserialized.
    import io
    for role, source in (('champion',champion_path),('candidate',candidate_path)):
        data = Path(source).read_bytes()
        model, tokenizer = load_v11(base)
        model.load_state_dict(torch.load(io.BytesIO(data),map_location='cpu',weights_only=True),strict=True)
        model.eval(); models[role]=model; tokenizers[role]=tokenizer
        identities[role]=hashlib.sha256(data).hexdigest()
        state_digests[role]=compute_model_state_sha256(model)
    if tokenizers['champion'].vocab_sha256 != tokenizers['candidate'].vocab_sha256:
        raise ValueError('Tokenizer identities differ')
    cand_bank = (
        SparseExpertCartridgeBank(Path(candidate_cartridge_bank_dir))
        if candidate_cartridge_bank_dir is not None
        else None
    )
    torch.set_num_threads(1)
    settings=dict(max_new_tokens=max_new_tokens,num_candidates=1,temperature=0.,inflight_alu=False)
    def adapter(role):
        def generate(prompt, settings):
            tokenizer=tokenizers[role];model=models[role]
            ids=tokenizer.encode(prompt,add_bos=True,add_eos=False)
            if len(ids)>=model.config.max_seq_len:
                raise ValueError('Prompt exceeds context budget')
            with torch.no_grad():
                if role == 'candidate' and cand_bank is not None:
                    generated,_,_ = cand_bank.execute_routed_request(
                        model,
                        prompt,
                        lambda m, _p: m.generate_verified(ids,eos_id=tokenizer.eos_id,tokenizer=tokenizer,**settings),
                        active_base_checkpoint_sha256=state_digests['candidate'],
                    )
                else:
                    generated,_,_=model.generate_verified(ids,eos_id=tokenizer.eos_id,tokenizer=tokenizer,**settings)
            new_ids=generated[len(ids):]
            if new_ids and new_ids[-1]==tokenizer.eos_id:
                termination_reason='eos'
            elif len(generated)>=model.config.max_seq_len:
                termination_reason='context_limit'
            elif len(new_ids)>=settings['max_new_tokens']:
                termination_reason='max_new_tokens'
            else:
                termination_reason='incomplete'
            return dict(
                completion=tokenizer.decode(new_ids,skip_special=False),
                prompt_tokens=len(ids),
                generated_tokens=len(new_ids),
                total_tokens=len(generated),
                termination_reason=termination_reason,
            )
        return generate
    report=evaluate_pair(rows,adapter('champion'),adapter('candidate'),identities=identities,settings=settings,
                         score_answer=lambda text,gold:score_rational_or_probability(text,gold,is_probability=False)['score']==1.,
                         score_trace=lambda text,gold:verify_r1_reasoning_trace(text,gold)['fully_verified'])
    report.update(split=split,split_manifest_sha256=manifest['sha256'],tokenizer_sha256=tokenizers['champion'].vocab_sha256,
                  assistance_mode='neural_only',
                  routing_assistance_mode='predicted_prefix_cartridge' if cand_bank is not None else 'none',
                  checkpoint_format='v11-compatible torch state_dict',
                  context_limit=models['champion'].config.max_seq_len)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--champion',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=('development','regression'),default='development')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing report')
    report=evaluate_checkpoints(json.loads(args.manifest.read_text()),args.champion,args.candidate,split=args.split)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(complete=report['comparison_complete'],paired=report['paired'],output=str(args.output))))


if __name__=='__main__':main()
