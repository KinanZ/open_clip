import argparse


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN18", "RN18_pretrained", "RN18_256", "RN18ish", "RN18ish_256", "RN50", "RN101", "RN50x4"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name == "ViT-B/32":
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to csv filewith training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to csv file with validation data",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/outputs_2/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument("--use-bn-sync",
                        default=False,
                        action="store_true",
                        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Specify a single GPU to run the code on for debugging."
             "Leave at None to use all available GPUs.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--regression-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        choices=["RN18", "RN18_pretrained", "RN18_256", "RN18ish", "RN18ish_256", "RN50", "RN101", "RN50x4", "ViT-B/32"],
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--openai-pretrained",
        default=False,
        action='store_true',
        help="Use the openai pretrained models.",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:6100",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--skip-aggregate",
        default=False,
        action="store_true",
        help="whether to aggregate features across gpus before computing the loss"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--eval-train",
        default=False,
        action="store_true",
        help="an option to evaluate the train set"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--C", type=float, default=3.16, help="inverse regularizer for logistic reg."
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there."
    )
    parser.add_argument(
        "--dp",
        default=False,
        action="store_true",
        help="Use DP instead of DDP."
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="In DP, which GPUs to use for multigpu training",
    )
    # My params
    parser.add_argument(
        "--set-aug-text",
        default=False,
        action="store_true",
        help="Whether to use the set augment function on the captions"
    )
    parser.add_argument(
        "--hflip-aug",
        default=False,
        action="store_true",
        help="Whether to use the horizontal flip function on image and caption"
    )
    parser.add_argument(
        "--negative-aug-text",
        default=False,
        action="store_true",
        help="Whether to use the augment negative on the captions"
    )
    parser.add_argument(
        "--positive-aug-text",
        default=False,
        action="store_true",
        help="Whether to use the augment positive on the captions"
    )
    parser.add_argument(
        "--skip-aug-text",
        default=False,
        action="store_true",
        help="Whether to use the augment skipping some words and replacing them with _ "
    )
    parser.add_argument(
        "--csv-label-key",
        type=str,
        default="labels",
        help="For csv-like datasets, the name of the key for the labels."
    )
    parser.add_argument(
        "--csv-bbox-key",
        type=str,
        default="bboxes",
        help="For csv-like datasets, the name of the key for the bboxes."
    )
    parser.add_argument(
        "--default-loss",
        default=False,
        action="store_true",
        help="Whether to use default clip loss"
    )
    parser.add_argument(
        "--Label-grouped",
        default=False,
        action="store_true",
        help="class based loss: to group the samples with the same label in the batch together"
    )
    parser.add_argument(
        "--Healthy-grouped",
        default=False,
        action="store_true",
        help="class based loss: to group the samples with label 0 (Healthy) in the batch together"
    )
    parser.add_argument(
        "--Healthy-Caption-grouped",
        default=False,
        action="store_true",
        help="class based loss: to group the samples with label 0 (Healthy) in the batch together. Additionally group the samples which belong to a deseased class and have the same caption together"
    )
    parser.add_argument(
        "--Caption-grouped",
        default=False,
        action="store_true",
        help="class based loss: to group the samples which belong to a deseased class and have the same caption together"
    )
    parser.add_argument(
        "--custom-eval",
        default=False,
        action="store_true",
        help="Updated evaluations"
    )
    parser.add_argument(
        "--custom-eval-no-healthy",
        default=False,
        action="store_true",
        help="Updated evaluations"
    )
    parser.add_argument(
        "--seed", type=int, default=101, help="random seed"
    )
    parser.add_argument(
        "--t-sne",
        default=False,
        action="store_true",
        help="log feature embedding?"
    )
    parser.add_argument(
        "--default-aug-img",
        default=False,
        action="store_true",
        help="Whether to use the default clip transforms"
    )
    parser.add_argument(
        "--custom-aug-img",
        default=False,
        action="store_true",
        help="Whether to use our custom transforms"
    )
    parser.add_argument(
        "--bbox-aug-img",
        default=False,
        action="store_true",
        help="Whether to use bbox aug on the images"
    )
    parser.add_argument(
        "--use-de-tokenizer",
        default=False,
        action="store_true",
        help="Whether to use the new german tokenizer"
    )
    parser.add_argument(
        "--new-model",
        default=False,
        action="store_true",
        help="use the new way to build the model."
    )
    parser.add_argument(
        "--embid-dim", type=int, default=512, help="output feature dimension for both models"
    )
    parser.add_argument(
        "--IN-pretrained",
        default=False,
        action='store_true',
        help="Use the imagenet pretrained model.",
    )
    parser.add_argument(
        "--freeze-vision-model",
        default=False,
        action='store_true',
        help="freeze the backbone of the vision model",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default='bert-base-german-cased',
        help="name of the pretrained text model",
    )
    parser.add_argument(
        "--transformer-dim", type=int, default=768, help="transformer width"
    )
    parser.add_argument(
        "--freeze-text-model",
        default=False,
        action='store_true',
        help="freeze the backbone of the text model",
    )
    parser.add_argument(
        "--use-weights-1",
        default=False,
        action='store_true',
        help="use class weights during pretraining where the weight of a class is: 1 - (class_count / total_count)",
    )
    parser.add_argument(
        "--use-weights-2",
        default=False,
        action='store_true',
        help="use class weights during pretraining where the weight of a class is: total_count - (num_of_classes / class_count)",
    )
    args = parser.parse_args()
    args.aggregate = not args.skip_aggregate

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
