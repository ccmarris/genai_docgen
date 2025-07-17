#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to generate responses from ChatGPT for a set of prompts and save them to a Word document.
"""
__author__ = "Chris Marrison"
__copyright__ = "Copyright 2025, Chris Marrison / Infoblox"
__license__ = "BSD2"
__version__ = "0.2.6"
__email__ = "chris@infoblox.com"

import logging
import argparse
import time
import clients.genai_client as genai_client
from utils.docgen import save_responses
from prompts import PROMPTS
from rich import print
from tqdm import tqdm
from platformdirs import user_config_dir, user_cache_dir

_logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ini", 
                        type=str, 
                        help="Path to the ini file with API key and model settings",
                        required= False)
    parser.add_argument(
        "-P", "--provider",
        type=str,
        default="openai",
        help="GenAI provider to use (default: openai). Other options: 'anthropic', 'azure', 'google', etc.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.docx",
        help="Output filename (default: output.docx)")
    parser.add_argument(
        "-f", "--output-format",
        choices=["docx", "txt", "md", "stdout"],
        default="docx",
        help="Output format: docx (default), txt, md or stdout")
    parser.add_argument(
        "-t", "--title",
        type=str,
        help="Title for the document (default: 'Generated Responses'")
    parser.add_argument(
        "-a", "--asprompts",
        action="store_true",
        default=False,
        help="Output responses as prompts (no prompts included in output)")
    parser.add_argument(
        "-p", "--prompt-file",
        type=str,
        help="Path to a file containing prompts (one per line). If not set, uses PROMPTS from prompts.py")
    parser.add_argument(
        "--ignore_cache",
        action="store_false",
        help="Ignore cached responses and regenerate all prompts")
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Clear the cache for the specified provider before generating responses")
    parser.add_argument(
        "-s", "--sleep",
        type=int,
        default=1,
        help="Sleep time in seconds between requests to avoid hitting rate limits (default: 1 second)")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging")
    # Add arguments for custom weightings
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    args = parser.parse_args()
    return args


def setup_logging(debug):
    '''
     Set up logging

     Parameters:
        debug (bool): True or False.

     Returns:
        None.

    '''
    # Set debug level
    if debug:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='%(asctime)s %(levelname)s: %(message)s')
    return


def load_prompts(prompt_file=None):
    '''
    Load prompts from a file or use the default PROMPTS list.
    Parameters:
        prompt_file (str): Path to a file containing prompts (one per line). If None, uses the default PROMPTS.
    Returns:
        list: A list of prompts.
    '''
    prompts = []
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = PROMPTS
    return prompts


def main():
    """
    Main function to generate responses from ChatGPT for a set of prompts and save them to a Word document.
    This function parses command-line arguments, sets up logging, loads prompts, initializes the ChatGPT client,
    generates responses for each prompt, and saves the responses to the specified output format.
    Returns:
        None
    """
    prompt_response_pairs:list = []
    total_tokens:int = 0
    args = parse_args()

    CONFIGDIR = user_config_dir(f"{args.provider}_docgen")
    CONFIGFILE = f"{args.provider}_docgen.ini"

    temperature = None
    top_p = None
    frequency_penalty = None
    presence_penalty = None

    if args.temperature is not None:
        temperature = args.temperature
    if args.top_p is not None:
        top_p = args.top_p
    if args.frequency_penalty is not None:
        frequency_penalty = args.frequency_penalty
    if args.presence_penalty is not None:
        presence_penalty = args.presence_penalty
    
    setup_logging(debug=args.debug)  # Set debug to True for verbose logging

    _logger.debug(
        f"Using parameters: temperature={temperature}, top_p={top_p}, frequency_penalty={frequency_penalty}, presence_penalty={presence_penalty}"
    )

    # Load prompts
    prompts = load_prompts(args.prompt_file)

    # Initialize ChatGPT client
    if not args.ini:
        filename = f"{CONFIGDIR}/{CONFIGFILE}"
        if os.path.exists(filename):
            args.ini = filename
            _logger.debug(f"No ini file specified, using default: {args.ini}")
        else:
            args.ini = None

    if args.clear_cache:
        _logger.info(f"Clearing cache for provider: {args.provider}")
        if genai_client.clear_cache(provider=args.provider):
            _logger.info("Cache cleared successfully.")
    
    client = genai_client.get_llm_client(provider=args.provider,
                                         inifile=args.ini, 
                                         use_cache=args.ignore_cache)

    # Generate responses for each prompt
    try:
        for prompt in tqdm(prompts, desc="Processing prompts"):
            _logger.debug(f"Sending prompt: {prompt}")

            response = client.get_response(
                prompt,
                model=client.model,
                temperature=temperature,
                top_p=top_p
                # frequency_penalty=frequency_penalty,
                # presence_penalty=presence_penalty,
            )
            # Check token usage
            total_tokens += client.last_usage.get('total_tokens', 0)

            if response == 'Success':
                prompt_response_pairs.append((prompt, client.last_response))
                time.sleep(args.sleep)
            else:
                _logger.error(f"Error generating response for prompt: {prompt}")
                _logger.error(f"Exiting due to error: {response}")
                print(f"[red]Error generating response for prompt: {prompt}[/red]")
                print(f"[red]Exiting due to error:[/red] {response}")
                break
    except KeyboardInterrupt:
        _logger.warning("Interrupted by user. Saving partial results...")

    if prompt_response_pairs:
        _logger.info(f"Saving responses to {args.output} (format: {args.output_format})")
        # Save responses to the specified output format
        save_responses(prompt_response_pairs, 
                       filename=args.output, 
                       output_format=args.output_format,
                       generate_title=False if args.asprompts else True,
                       title=args.title if args.title else "Generated Responses",
                       generate_as_prompts=args.asprompts)
    else:
        _logger.warning("No responses to save.")

    _logger.warning(f"Total tokens used: {total_tokens}")
    print(f"[yellow]Total tokens used: {total_tokens}[/yellow]")

    return

if __name__ == "__main__":
    main()