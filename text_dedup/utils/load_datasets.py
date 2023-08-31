from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk

def load_dataset(args):
    if args.local:
        return load_from_disk(args.path)
    else:
        return hf_load_dataset(
            path=args.path,
            name=args.name,
            data_dir=args.data_dir,
            data_files=args.data_files,
            split=args.split,
            revision=args.revision,
            cache_dir=args.cache_dir,
            use_auth_token=args.use_auth_token,
            num_proc=10
        )