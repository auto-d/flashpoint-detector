#!/usr/bin/env python3

import argparse 
import os
import tempfile
import glob
from . import naive
from . import classic 
from . import nn
from fed.dataset import FlashpointsDataset, FlashpointsTorchDataset
from fed.process import run_subprocess
from . import demo

def deploy(share=False, data_tag="test"): 
    """
    Deploy, optionally pushing to the cloud (share = True)
    """
    demo(share, data_tag)

def readable_file(path):
    """
    Test for a readable file
    NOTE: reused from NLP project
    """
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"'{path}' doesn't exist.")
    return path

def nonexistent_file(path):
    """
    Test for a non-existent file (to help avoid overwriting important stuff)
    NOTE: reused from NLP project
    """
    if os.path.exists(path):
        raise argparse.ArgumentTypeError(f"'{path}' already exists.")
    return path

def readable_dir(path):
    """
    Test for a readable dir
    NOTE: reused from NLP project
    """
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory.")
    
def nonexistent_dir(path): 
    """
    Test to ensure directory doesn't exist
    NOTE: reused from NLP project
    """    
    if os.path.exists(path):
        if os.path.isdir(path):
            raise argparse.ArgumentTypeError(f"Directory '{path}' already exists.")
        else:
            raise argparse.ArgumentTypeError(f"Path '{path}' exists and is not a directory.")
    return path

def build_parser(): 
    """
    Apply a command-line schema, returning a parser

    NOTE: Parser setup based on work from prior assignments 
    """
    parser = argparse.ArgumentParser("fed", description="Flashpoints Ukraine event predictor")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Build mode 
    build_parser = subparsers.add_parser("build") 
    build_parser.add_argument("--sample-n", type=int, help="Number of detections to sample", default=10000, required=False)
    build_parser.add_argument("--output-dir", type=readable_dir, help="Directory to write resulting dataset to", default="data/", required=False)
    build_parser.add_argument("--tag", type=str, help="Friendly name to tag dataset names with", required=True)

    # Train mode 
    train_parser = subparsers.add_parser("train") 
    train_parser.add_argument("--data-dir", type=readable_dir, help="Directory to look for tagged dataset", default="data/", required=False)
    train_parser.add_argument("--data-tag", type=str, help="Dataset tag to look for (set during creation)", required=True)
    train_parser.add_argument("--model-dir", help="Directory to write resulting model to", default="models")
    train_parser.add_argument("--nn-epochs", type=int, default=1)
    train_parser.add_argument("--nn-batch", type=int, default=1)
    train_parser.add_argument("--type", choices=['naive', 'classic', 'neural'], default='neural')

    # Test mode 
    test_parser = subparsers.add_parser("test") 
    test_parser.add_argument("--model_dir", type=readable_dir, help="Directory to load model from", default="models")
    test_parser.add_argument("--data-dir", type=readable_dir, help="Directory to look for tagged dataset", default="data/", required=False)    
    test_parser.add_argument("--data-tag", type=str, help="Dataset tag to look for (set during creation)", required=True)
    test_parser.add_argument("--type", choices=['naive', 'classic', 'neural'], default='neural')

    # Deploy mode 
    deploy_parser = subparsers.add_parser("deploy")
    deploy_parser.add_argument("--data-tag", type=str, help="Data tag associated with the ref data")
    deploy_parser.add_argument("--share", action='store_true', default=False, help="Whether or not to deploy to gradio hosting")
    
    return parser
    
def router(): 
    """
    Argument processor and router

    NOTE: Argparsing with help from chatgpt: https://chatgpt.com/share/685ee2c0-76c8-8013-abae-304aa04b0eb1
    NOTE: arg parsing logic incorporates work from prior 540 assignments
    """

    parser = build_parser() 
    args = parser.parse_args()    
    
    match(args.mode):
        case "build":
            dataset = FlashpointsDataset(args.tag)
            dataset.load()
            dataset.store(args.output_dir)

        case "train":
            dataset = FlashpointsDataset(args.data_tag)
            dataset.load()
            dataset.split()

            match(args.type): 
                case 'naive':
                    model = naive.train(dataset.train, dataset.val)
                    naive.save_model(model, args.model_dir)
                case 'classic':
                    model = classic.train(dataset.train, dataset.val) 
                    classic.save_model(model, args.model_dir)
                case 'neural': 
                    torch_dataset = FlashpointsTorchDataset(matrix=dataset.train, batch_size=args.nn_batch)
                    model = nn.train(torch_dataset, args.nn_epochs, dataset.val)
                    nn.save_model(model, args.model_dir)

        case  "test":
            dataset = FlashpointsDataset(args.data_tag)
            dataset.load(args.data_dir)
            dataset.split() 
            match (args.type): 
                case 'naive':
                    model = naive.load_model(args.model_dir)
                    naive.test(model, dataset.test)
                case 'classic':
                    model = classic.load_model(args.model_dir)
                    classic.test(model, dataset.test) 
                case 'neural': 
                    model = nn.load_model(args.model_dir)
                    torch_dataset = FlashpointsTorchDataset(dataset)
                    nn.test(model, torch_dataset, dataset.test)

        case "deploy":
            deploy(args.share, args.data_tag)
        case _:
            parser.print_help()