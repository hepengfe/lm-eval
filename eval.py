from arguments import TrainerArguments
from soft_trainer import SoftTrainer

import os
import argparse
import datetime
EXPR_DIR = "/media/NAS/NLP/tmp/"

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument("--config", type=str, default="configs/gpt2.yaml")
    arg_parser.add_argument("--dataset", type=str, default="wikitext")
    arg_parser.add_argument("--models", type=str, default="gpt2")
    arg_parser.add_argument("--mode", type=str, default="fine_tuning")
    arg_parser.add_argument("--dev", action="store_true")
    arg_parser.add_argument("--max_steps", type=int, default=30000)
    arg_parser.add_argument("--eval_save_steps", type=int, default=300)
    arg_parser.add_argument("--per_device_train_batch_size", type=int, default=6)
    arg_parser.add_argument("--per_device_eval_batch_size", type=int, default=6)
    arg_parser.add_argument("--lr", type=float, default=0.3)
    arg_parser.add_argument("--num_soft_tokens", type=int, default=5)
    arg_parser.add_argument("--early_stop", action="store_true")
    args = arg_parser.parse_args()
    
    early_stopping_patience = 4
    predict_with_generate = False
    dataset_names = ("super_glue","boolq")
    if args.dev:
        # get current time
        
        EXPR_DIR = "~/tmp/" # local nvme loading is faster but it takes longer to download model for the first time
        # lr = 3e-7
        
        # lr = 1e-5
        # lr = 0.3 # for embedding tuning

        args.max_steps = 30000
        # mode = "fine_tuning"
        # mode = "embedding_tuning"
        args.mode = "prompt_tuning"
        per_device_train_batch_size = 12
        per_device_eval_batch_size = 12
        predict_with_generate = False
        # predict_with_generate = True
        
        default_optimizer_n_scheduler = False
        include_labels_in_input = False
        include_labels_in_input = True



    
    if args.mode == "fine_tuning":
        args.lr = 5e-5  # for regular fine-tuning


    args.models = args.models.split(",")
    args.models = [m.strip() for m in args.models]

    # EXPR_DIR = "~/tmp/"
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(EXPR_DIR, time)
    cache_path = os.path.join(EXPR_DIR, "cache")
    
    train_args = TrainerArguments(
        # train t5-base
        # model_names_or_paths=["google/t5-xxl-lm-adapt"],
        # model_names_or_paths=["google/t5-xl-lm-adapt"],
        # model_names_or_paths= ["google/t5-large-lm-adapt"],
        model_names_or_paths= args.models,
        # model_names_or_paths=["google/t5-xl-lm-adapt"],
        # model_names_or_paths = ["bigscience/bloomz-560m"],
        # model_names_or_paths = ["bigscience/bloomz-7b1"],
        # model_names_or_paths = ["facebook/opt-6.7b"],
        model_parallel_gpus = 3,


        output_dir = output_path,
        cache_dir=cache_path,
        overwrite_output_dir=True,
           
        dataset_name = dataset_names[0],
        dataset_config_name = dataset_names[1],

        # dataset_name = "super_glue",
        # dataset_config_name = "boolq",
        
        # dataset_name = "super_glue",
        # dataset_config_name = "wic",

        # dataset_name = "super_glue",
        # dataset_config_name = "multirc",

        max_source_length=512,
        max_target_length=8,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        predict_with_generate=predict_with_generate,
        # warmup_ratio=0.1,
        # warmup_stpes = 100,
        # lr_scheduler_type="linear",
        # lr_scheduler_type="constant",
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps = args.eval_save_steps,
        save_strategy="steps",
        save_steps = args.eval_save_steps,
        save_total_limit=1,
        learning_rate=args.lr,
        # fp16=True, -> zero loss
        # eval_data_preprocess=False,
        # label_names = ["class_ids", "labels", "eval_class_ids"],
        # label_names = ["labels", "class_ids"], # labels is for training, class_ids is for eval 
        label_names = ["labels"],
        load_best_model_at_end=True,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if args.early_stop else None,
        metric_for_best_model = "eval_avg_acc",
        report_to = "wandb",
        # verbalize_labels = True,
        pad_to_max_length = True,
        # pad_to_max_length = False,
        prompt_init_method = "verbalizer",
        # prompt_init_method = "random",
        num_soft_tokens=args.num_soft_tokens,
        dev = args.dev,
        mode = args.mode,
        include_labels_in_input = include_labels_in_input,
    )

    trainer = SoftTrainer(train_args)
    # results = trainer.evaluate()
    # print("pre-prompting tuning results: ", results)
    trainer.train()
    